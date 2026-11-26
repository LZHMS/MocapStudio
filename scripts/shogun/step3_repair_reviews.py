"""Repair old NEEDS_REVIEW checkpoints without rebuilding or replacing their sources."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import uuid

from step1_process_takes import discover_shogun, load_manifest, run_jobs, write_json
from pipeline_review_plan import plan_review_repairs
from pipeline_review_trim import CAMERA_EXEMPTIONS, DEFAULT_REVIEW_POLICY
from pipeline_run_registry import register_run
from pipeline_layout import initialize_layout, load_layout


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("-SourceOutput", required=True, help="Existing completed pipeline output (read-only source)")
    result.add_argument("-OutputDir", help="Separate repair output; default SourceOutput/_repair_runs/<run_id>")
    result.add_argument("-ShogunCL", help="Installed ShogunPostCL.exe")
    result.add_argument("-SubjectManifest", help="Optional explicit capture roster overrides in the existing manifest format")
    result.add_argument("-MaxWorkers", type=int, default=2)
    result.add_argument("-Limit", type=int, help="Limit the sparse-first pending selection, without changing the full plan")
    result.add_argument("-StartupTimeout", type=float, default=120.0)
    result.add_argument("-PropJitterMaxCorrection", type=float, default=DEFAULT_REVIEW_POLICY["prop_jitter_max_correction"], help="Maximum model-derived correction in mm, up to 5")
    result.add_argument("-PropJitterMaxGap", type=int, default=DEFAULT_REVIEW_POLICY["prop_jitter_max_gap"])
    result.add_argument("-PropJitterWindow", type=int, default=DEFAULT_REVIEW_POLICY["prop_jitter_window"])
    result.add_argument("-CameraExemptions", default=",".join(DEFAULT_REVIEW_POLICY["camera_exemptions"]), help="Only these exact cameras can be waived after repair")
    result.add_argument("-NoAutoTrim", action="store_true")
    result.add_argument("-TrimMaxLossFraction", type=float, default=DEFAULT_REVIEW_POLICY["trim_max_loss_fraction"])
    result.add_argument("-TrimMinFrames", type=int, default=DEFAULT_REVIEW_POLICY["trim_min_frames"])
    result.add_argument("-DryRun", action="store_true", help="Plan only; no SDK, native process or output writes")
    return result


def _positive(value, label):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")


def _policy(options):
    _positive(options.StartupTimeout, "StartupTimeout")
    _positive(options.PropJitterMaxCorrection, "PropJitterMaxCorrection")
    for label in ("MaxWorkers", "PropJitterMaxGap", "PropJitterWindow", "TrimMinFrames"):
        if getattr(options, label) <= 0:
            raise ValueError(f"{label} must be positive")
    if options.TrimMinFrames < 120:
        raise ValueError("TrimMinFrames must be at least 120")
    if options.PropJitterMaxCorrection > 5 or options.PropJitterMaxGap > 5:
        raise ValueError("Review jitter is bounded to at most 5 mm and 5 frames")
    if options.Limit is not None and options.Limit <= 0:
        raise ValueError("Limit must be positive")
    if not math.isfinite(options.TrimMaxLossFraction) or not 0 <= options.TrimMaxLossFraction <= 0.10:
        raise ValueError("TrimMaxLossFraction must be between zero and 0.10; larger data loss requires manual review")
    cameras = [name.strip() for name in options.CameraExemptions.split(",") if name.strip()]
    if len(set(cameras)) != len(cameras) or set(cameras) - CAMERA_EXEMPTIONS:
        raise ValueError("Only unique camera_1/camera_2/camera_3 exemptions are allowed")
    return {
        "review_repair": True,
        "repair_subject": None,
        "ignore_labeling_clusters": False,
        "min_coverage": 1.0,
        "core_markers": {},
        "prop_tolerance": 2.0,
        "prop_max_step": 200.0,
        "prop_jitter_max_correction": options.PropJitterMaxCorrection,
        "prop_jitter_max_gap": options.PropJitterMaxGap,
        "prop_jitter_window": options.PropJitterWindow,
        "camera_exemptions": cameras,
        "auto_trim": not options.NoAutoTrim,
        "trim_max_loss_fraction": options.TrimMaxLossFraction,
        "trim_min_frames": options.TrimMinFrames,
    }


def _plan_snapshot(plan):
    fields = (
        "job_id", "input_path", "relative_path", "checkpoint_vdf", "checkpoint_fingerprint",
        "source_output", "source_report", "source_report_sha256", "checkpoint_completion",
        "source_timecode", "expected_subjects",
        "expected_props", "expected_human_count", "confirmed_subject_roster", "roster_resolution",
        "review_policy", "priority",
    )
    return {
        "counts": plan["counts"],
        "jobs": [{key: job[key] for key in fields if key in job} for job in plan["jobs"]],
        "skipped": plan["skipped"],
        "roster_pending": plan["roster_pending"],
    }


def _print_plan(plan, output):
    print("[REPAIR-PLAN] " + json.dumps(plan["counts"], ensure_ascii=False), flush=True)
    print(f"[REPAIR-PLAN] output={output}; source files/reports are retained", flush=True)
    for job in plan["jobs"][:10]:
        priority = job["priority"]
        print(f"[PLAN] {job['relative_path']}; expected_count={job.get('expected_human_count')}; priority={priority}", flush=True)
    print("[INFO] Completed native checkpoint -> prop association/jitter -> prop solve -> final quality -> optional verified small trim", flush=True)
    print("[INFO] Only permitted failed cameras may be waived; chairs/desks/other props remain required. No stitching across gaps.", flush=True)


def main(argv=None):
    options = parser().parse_args(argv)
    policy = _policy(options)
    source = Path(options.SourceOutput).expanduser().resolve()
    source_layout = load_layout(source)
    if not source.is_dir() or (source_layout is None and not (source / "takes").is_dir()):
        raise ValueError(f"SourceOutput has no canonical takes reports or sequence layout: {source}")
    if (source / ".pipeline.lock").exists():
        raise RuntimeError("Source batch has a lock; finish or safely recover that batch before repairing its outputs")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-repair-" + uuid.uuid4().hex[:8]
    output = Path(options.OutputDir).expanduser().resolve() if options.OutputDir else source / "_repair_runs" / run_id
    if (output / ".pipeline.lock").exists():
        raise RuntimeError("Repair output has an existing lock; inspect its owning process before resuming")
    manifest = load_manifest(options.SubjectManifest)
    if source_layout and not options.DryRun:
        initialize_layout(output, source_layout["data_root"])
    plan = plan_review_repairs(source, output, manifest=manifest, limit=options.Limit)
    _print_plan(plan, output)
    if options.DryRun:
        print("[DRY-RUN] No files written; no Shogun SDK imported or native process started.", flush=True)
        return 2 if plan["skipped"] else 0
    if not plan["jobs"]:
        print("[INFO] No pending eligible old review checkpoints; inspect preserved/skipped counts.", flush=True)
        return 2 if plan["skipped"] else 0
    executable = Path(options.ShogunCL).expanduser().resolve() if options.ShogunCL else discover_shogun()
    if not executable or not executable.is_file() or executable.name.casefold() != "shogunpostcl.exe":
        raise ValueError("Pass an existing ShogunPostCL.exe with -ShogunCL")
    from pipeline_session import check_sdk
    check_sdk(str(executable))
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".pipeline.lock"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError(f"Another repair or stale lock exists: {lock_path}; verify its owning process") from None
    try:
        with lock:
            json.dump({"pid": os.getpid(), "run_id": run_id, "source_output": str(source)}, lock)
            lock.flush()
            if (source / ".pipeline.lock").exists():
                raise RuntimeError("Source batch became active during planning; no repair workers dispatched")
            plan = plan_review_repairs(source, output, manifest=manifest, limit=options.Limit)
            write_json(output / "repair_plan.json", dict(_plan_snapshot(plan), run_id=run_id,
                       source_output=str(source), output_dir=str(output), policy=policy))
            write_json(output / "roster_pending.json", {"run_id": run_id, "items": plan["roster_pending"],
                       "note": "Unconfirmed rosters are not inferred from filenames or reconstructed person counts"})
            launch = {"run_id": run_id, "scheduler_pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
                      "source_output": str(source), "output_dir": str(output), "max_workers": options.MaxWorkers,
                      "command": [sys.executable, *sys.argv], "policy": policy, "plan_counts": plan["counts"],
                      "log": str(output / "production.log"), "kind": "review_repair", "planned_total": len(plan["jobs"])}
            write_json(output / "production_launch.json", launch)
            write_json(output / "launch_history" / (run_id + ".json"), launch)
            register_run(output, kind=launch["kind"], run_id=run_id, source_output=source,
                         started_at=launch["started_at"], scheduler_pid=launch["scheduler_pid"],
                         planned_total=launch["planned_total"])
            results = run_jobs(plan["jobs"], output, executable, options.MaxWorkers, None,
                               options.StartupTimeout, policy, False)
            rows = [{"job_id": row["job_id"], "status": row["status"]} for row in results]
            dispatched = {row["job_id"] for row in rows}
            for job in plan.get("preserved", []):
                previous = json.loads(Path(job["report_path"]).read_text(encoding="utf-8"))
                if previous["job_id"] not in dispatched:
                    rows.append({"job_id": previous["job_id"], "status": previous["status"]})
            counts = {status: sum(row["status"] == status for row in rows)
                      for status in ("SUCCESS", "NEEDS_REVIEW", "FAILED")}
            summary = {"schema_version": 1, "run_id": run_id, "source_output": str(source),
                       "counts": counts, "jobs": rows, "plan_counts": plan["counts"], "max_workers": options.MaxWorkers,
                       "policy": policy, "shogun": str(executable), "identity_validated": False,
                       "finished_at": datetime.now(timezone.utc).isoformat(),
                       "scope": "This separate repair output only; source reports and artifacts were not replaced"}
            write_json(output / ("batch-" + run_id + ".json"), summary)
            write_json(output / "batch_report.json", summary)
            print("[SUMMARY] " + " ".join(f"{key}={value}" for key, value in counts.items()), flush=True)
            return 1 if counts["FAILED"] else 2 if counts["NEEDS_REVIEW"] or plan["skipped"] else 0
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    mp.freeze_support()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[INTERRUPTED] Owned repair workers stopped; original outputs retained.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        raise SystemExit(1)
