"""Repair all reusable processed takes into a separate, resumable native output."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import re
import sys
import uuid

from step1_process_takes import discover_shogun, inside, load_manifest, names, publish, run_jobs, write_json
from pipeline_resume import select_resume_jobs
from pipeline_review_plan import _make_job, _resume
from recheck_worker import run_isolated_recheck
from pipeline_run_registry import register_run
from pipeline_layout import in_run, initialize_layout, load_layout, report_paths, report_relative, source_runs, take_paths


_COMPLETED = {"SUCCESS", "NEEDS_REVIEW"}
_RECHECK_FAILURE = "independent_recheck_failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _artifact_hashes(report: dict) -> dict:
    """Bind cached inspection evidence to every delivered artifact's exact bytes."""
    hashes, by_path = {}, {}
    products = report.get("artifacts")
    if not isinstance(products, dict) or not products:
        raise ValueError("Completed report has no artifact set")
    entries = [(f"artifact:{role}", value) for role, value in products.items()]
    exports = report.get("subject_exports", [])
    if not isinstance(exports, list):
        raise ValueError("Invalid subject exports")
    entries.extend((f"subject_export:{index}", row.get("path"))
                   for index, row in enumerate(exports))
    for key, value in entries:
        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid artifact path: {key}")
        path = Path(value).resolve()
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty artifact: {path}")
        normalized = str(path)
        if normalized not in by_path:
            by_path[normalized] = _sha(path)
        hashes[key] = {"path": normalized, "sha256": by_path[normalized]}
    return hashes


def _source_roots(source: Path, output: Path) -> list[Path]:
    return [root for root in source_runs(source) if root != output]


def _assert_sources_idle(roots: list[Path]) -> None:
    for root in roots:
        if (root / ".pipeline.lock").exists():
            raise RuntimeError(f"Source batch has a lock; inspect its owner before repairing: {root}")


def _prepare_job(report: dict, path: Path, payload: bytes, source: Path,
                 output: Path, manifest: dict) -> dict:
    """Keep the established checkpoint proof, not the old driver's VDF fallback."""
    job = _make_job(report, path, payload, source, output, manifest)
    resolution = job["roster_resolution"]
    if not resolution["confirmed"]:
        # Detected/effective names remain evidence, never the expected roster.
        job["expected_subjects"] = None
        resolution["pending"] = True
    relative = job["relative_path"].casefold()
    take_props = {key.replace("\\", "/").casefold(): value
                  for key, value in manifest.get("prop_takes", {}).items()}
    if relative in take_props:
        props = take_props[relative]
    elif manifest.get("expected_props") is not None:
        props = manifest["expected_props"]
    elif report.get("expected_props") is not None:
        props = report["expected_props"]
    else:
        props = report.get("effective_props")
    if props is not None:
        job["expected_props"] = names(props, "expected props", allow_empty=True)
    job["split_subjects"] = True
    job["repair_checkpoint_props"] = False
    # Preserve the preferred checkpoint's camera/jitter/quality scope. Do not
    # introduce the review tool's default camera waivers or shorten more data.
    job["review_policy"] = dict(report.get("policy") or {})
    job["review_policy"].update(repair_subject=None, review_repair=True, auto_trim=False)
    job["review_policy"].setdefault("camera_exemptions", [])
    job["review_policy"].setdefault("prop_jitter_max_correction", 0.0)
    return job


def plan_human_repairs(source_output: Path, output: Path, *, manifest: dict | None = None,
                       limit: int | None = None, relative_path: str | None = None,
                       job_id: str | None = None) -> dict:
    """Read-only full inventory; select newest valid checkpoint per raw identity.

    Report mtime (nanoseconds), then canonical path, deterministically orders
    valid duplicates. Invalid/newer reports never hide an older proven checkpoint.
    Every canonical input report has one compact inventory row, including failures.
    """
    source, output = Path(source_output).resolve(), Path(output).resolve()
    if not source.is_dir() or (load_layout(source) is None and not (source / "takes").is_dir()):
        raise ValueError("Source output must contain canonical takes reports or a sequence layout")
    if inside(source, output):
        raise ValueError("Output must not equal or contain source output")
    if inside(output, source):
        parts = output.relative_to(source).parts
        if len(parts) != 2 or parts[0] != "_repair_runs":
            raise ValueError("Nested output must be source/_repair_runs/<runid>")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be positive")
    for bucket in ("takes", "_needs_review", ".work", "_failed", "logs"):
        if not inside((output / bucket).resolve(), output):
            raise ValueError(f"Output bucket escapes output root: {bucket}")
    roots = _source_roots(source, output)
    _assert_sources_idle(roots)
    inventory, winners, identities = [], {}, {}
    for root in roots:
        for path in sorted(report_paths(root)):
            row = {"source_output": str(root), "source_report": str(path)}
            inventory.append(row)
            try:
                report_relative(root, path)
                payload = path.read_bytes()
                row["source_report_sha256"] = hashlib.sha256(payload).hexdigest()
                report = json.loads(payload.decode("utf-8-sig"))
                if not isinstance(report, dict):
                    raise ValueError("Canonical report must be a JSON object")
                row.update(status=report.get("status"), job_id=report.get("job_id"),
                           relative_path=report.get("relative_path"), source_error=report.get("error"))
                rel = report.get("relative_path") or ""
                if re.search(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.mcp$", Path(rel).name, re.IGNORECASE):
                    row.update(disposition="skipped_informal_name",
                               reason="Take filename is an informal timestamp-named trial; excluded from batch repair",
                               checkpoint_vdf=report.get("artifacts", {}).get("vdf"))
                    continue
                if report.get("status") == "FAILED":
                    row.update(disposition="source_failed", reason="Failed processing is inventoried, not claimed repaired",
                               checkpoint_vdf=report.get("artifacts", {}).get("vdf"))
                    continue
                if report.get("status") not in _COMPLETED:
                    raise ValueError(f"Unsupported canonical status: {report.get('status')!r}")
                if report.get("quality", {}).get("basis") != "post_processing":
                    raise ValueError("Checkpoint quality lacks post-processing evidence")
                job = _prepare_job(report, path, payload, root, output, manifest or {})
                key = job["job_id"]
                raw = str(Path(job["input_path"]).resolve()).casefold()
                identities.setdefault(key, set()).add(raw)
                row.update(disposition="duplicate", checkpoint_vdf=job["checkpoint_vdf"],
                           checkpoint_completion=job["checkpoint_completion"]["basis"])
                row["report_mtime_ns"] = path.stat().st_mtime_ns
                rank = (row["report_mtime_ns"], str(path).casefold())
                if key not in winners or rank > winners[key][0]:
                    winners[key] = (rank, job, row)
            except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
                row.update(disposition="invalid_checkpoint", reason=f"{type(error).__name__}: {error}")
    jobs, roster_pending = [], []
    for key in sorted(winners, key=lambda key: winners[key][1]["relative_path"].casefold()):
        _, job, row = winners[key]
        if len(identities[key]) != 1:
            for item in inventory:
                if item.get("job_id") == key and item["disposition"] == "duplicate":
                    item.update(disposition="identity_conflict", reason="Same take identity refers to different raw MCP inputs")
            continue
        row["disposition"] = "selected"
        if job["roster_resolution"]["pending"]:
            roster_pending.append({"job_id": key, "relative_path": job["relative_path"],
                                   "roster_resolution": job["roster_resolution"]})
        jobs.append(job)
    for row in inventory:
        key = row.get("job_id")
        winner = winners.get(key) if isinstance(key, str) else None
        if row["disposition"] == "duplicate" and winner:
            row["preferred_source_report"] = winner[1]["source_report"]
    eligible = len(jobs)
    if relative_path is not None:
        selected = relative_path.replace("\\", "/").casefold()
        jobs = [job for job in jobs if job["relative_path"].casefold() == selected]
    if job_id is not None:
        jobs = [job for job in jobs if job["job_id"] == job_id]
    if (relative_path is not None or job_id is not None) and len(jobs) != 1:
        raise ValueError("Take selector must match exactly one reusable checkpoint; inspect the full inventory")
    matching = len(jobs)
    if limit is not None:
        jobs = jobs[:limit]
    selected_ids = {job["job_id"] for job in jobs}
    for row in inventory:
        if row["disposition"] == "selected" and row.get("job_id") not in selected_ids:
            row["disposition"] = "deferred"
    counts = {"reports_scanned": len(inventory), "eligible_identities": eligible,
              "selected": len(jobs), "deferred": eligible - len(jobs),
              "limited": matching - len(jobs), "roster_pending": len(roster_pending)}
    inventoried = {row["job_id"] for row in inventory
                   if isinstance(row.get("job_id"), str) and row["job_id"]}
    counts["inventoried_identities"] = len(inventoried)
    counts["identities_without_reusable_checkpoint"] = len(inventoried) - eligible
    for disposition in ("source_failed", "invalid_checkpoint", "identity_conflict", "duplicate", "skipped_informal_name"):
        counts[disposition] = sum(row["disposition"] == disposition for row in inventory)
    return {"source_roots": [str(root) for root in roots], "jobs": jobs, "inventory": inventory,
            "roster_pending": roster_pending, "counts": counts}


def _resume_selection(jobs: list[dict], output: Path) -> dict:
    selection = select_resume_jobs(jobs, output)
    for job in selection["preserved"]:
        _resume(job, output)  # Also verify source SHA and per-take artifact containment.
    for job in jobs:
        path = Path(job["report_path"])
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            previous = report.get("source_report_sha256")
            if previous is not None and previous != job["source_report_sha256"]:
                raise ValueError(f"Output has different source provenance: {path}")
    for job, stage in selection["recoverable"]:
        report = json.loads((stage / ".control" / "result.json").read_text(encoding="utf-8"))
        if (report.get("input_path") != job["input_path"]
                or report.get("source_report_sha256") != job["source_report_sha256"]):
            raise ValueError(f"Recoverable stage has different input/source provenance: {stage}")
    return selection


def _recheck_saved(job: dict, output: Path, executable: Path, timeout: float) -> dict:
    path = Path(job["report_path"])
    relative = report_relative(output, path)
    if relative.as_posix() != job["relative_path"]:
        raise ValueError("Independent recheck report belongs to a different take")
    state_path = path.with_name("independent_recheck.json")
    if not in_run(output, state_path):
        raise ValueError("Independent recheck sidecar escapes this run")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") not in _COMPLETED:
        return {"job_id": job["job_id"], "status": "NOT_PRODUCED"}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (state.get("status") == "PASSED" and state.get("report_sha256") == _sha(path)
                and state.get("source_report_sha256") == job["source_report_sha256"]):
            try:
                current_hashes = _artifact_hashes(report)
            except (OSError, ValueError, TypeError, AttributeError):
                current_hashes = None
            if current_hashes is not None and state.get("artifact_sha256") == current_hashes:
                return state
    state = {"job_id": job["job_id"], "relative_path": job["relative_path"],
             "source_report_sha256": job["source_report_sha256"], "status": "RUNNING", "started_at": _now()}
    # Remove only our previous failure gate; the native checker rebuilds quality.
    report["review_reasons"] = [reason for reason in report.get("review_reasons", [])
                                if reason != _RECHECK_FAILURE]
    report.setdefault("quality", {})["usable"] = False
    report["status"] = "NEEDS_REVIEW"
    report["independent_recheck"] = {"status": "RUNNING", "started_at": state["started_at"]}
    for entry in report.get("subject_exports", []):
        entry.setdefault("quality", {})["usable"] = False
    write_json(path, report)
    write_json(state_path, state)
    try:
        result = run_isolated_recheck(path, output, str(executable), False, timeout=timeout)
        report = json.loads(path.read_text(encoding="utf-8"))
        artifact_hashes = _artifact_hashes(report)
    except Exception as error:
        report = json.loads(path.read_text(encoding="utf-8"))
        report["status"] = "NEEDS_REVIEW"
        report.setdefault("quality", {})["usable"] = False
        report["review_reasons"] = list(dict.fromkeys([*report.get("review_reasons", []), _RECHECK_FAILURE]))
        for entry in report.get("subject_exports", []):
            entry.setdefault("quality", {})["usable"] = False
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}", finished_at=_now())
    else:
        state.update(status="PASSED", result=result, artifact_sha256=artifact_hashes, finished_at=_now())
    report["independent_recheck"] = {key: state[key] for key in
                                    ("status", "finished_at", "error", "result") if key in state}
    write_json(path, report)
    state["report_sha256"] = _sha(path)
    write_json(state_path, state)
    return state


def summarize_outputs(jobs: list[dict], rechecks: list[dict], inventory_counts: dict) -> dict:
    """JSON aggregation only; actual native rechecking happens in _recheck_saved."""
    rows = []
    for job in jobs:
        path = Path(job["report_path"])
        report = json.loads(path.read_text(encoding="utf-8"))
        humans = [{"name": row.get("name"), "usable": row.get("quality", {}).get("usable", False),
                   "review_reasons": row.get("review_reasons", [])}
                  for row in report.get("subjects", []) if row.get("processing_target")]
        rows.append({"job_id": job["job_id"], "relative_path": job["relative_path"],
                     "status": report["status"], "quality": report.get("quality"), "humans": humans,
                     "independent_recheck": report.get("independent_recheck", {}).get("status"),
                     "review_reasons": report.get("review_reasons", []),
                     "error": report.get("error"), "report_path": str(path)})
    return {"counts": {status: sum(row["status"] == status for row in rows)
                       for status in ("SUCCESS", "NEEDS_REVIEW", "FAILED")},
            "recheck_counts": {status: sum(row["status"] == status for row in rechecks)
                               for status in ("PASSED", "FAILED", "NOT_PRODUCED")},
            "human_quality_counts": {
                "all_humans_usable_and_rechecked": sum(
                    row["independent_recheck"] == "PASSED" and bool(row["humans"])
                    and all(human["usable"] for human in row["humans"]) for row in rows),
                "human_issues_or_not_rechecked": sum(
                    row["independent_recheck"] != "PASSED" or not row["humans"]
                    or not all(human["usable"] for human in row["humans"]) for row in rows)},
            "inventory_counts": inventory_counts, "jobs": rows, "rechecks": rechecks,
            "scope": "Selected new outputs only. Source failures, invalid checkpoints and deferred takes remain in repair_plan.json; PASSED recheck means inspected, not necessarily usable.",
            "finished_at": _now(), "identity_validated": False}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-output", default="H:/MocapData_processed")
    result.add_argument("--output-dir", required=True, help="Independent output; reuse with --resume after inspecting any lock")
    result.add_argument("--shogun-cl", help="Installed ShogunPostCL.exe (otherwise discover installation)")
    result.add_argument("--subject-manifest", help="Existing validated subject/prop manifest format")
    result.add_argument("--max-workers", type=int, default=2)
    result.add_argument("--limit", type=int)
    selector = result.add_mutually_exclusive_group()
    selector.add_argument("--relative-path", help="Exact raw relative take identity, including .mcp")
    selector.add_argument("--job-id", help="Exact canonical job ID")
    result.add_argument("--startup-timeout", type=float, default=120.0)
    result.add_argument("--recheck-timeout", type=float, default=900.0,
                        help="Independent recheck process deadline; long takes may require more")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true", help="Full read-only inventory; no writes or native processes")
    return result


def main(argv=None) -> int:
    options = parser().parse_args(argv)
    if options.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    if options.limit is not None and options.limit <= 0:
        raise ValueError("--limit must be positive")
    for label in ("startup_timeout", "recheck_timeout"):
        if not math.isfinite(getattr(options, label)) or getattr(options, label) <= 0:
            raise ValueError(f"--{label.replace('_', '-')} must be finite and positive")
    source = Path(options.source_output).expanduser().resolve()
    output = Path(options.output_dir).expanduser().resolve()
    manifest = load_manifest(options.subject_manifest)
    if output.exists() and any(output.iterdir()) and not options.resume and not options.dry_run:
        raise ValueError("Nonempty output requires --resume; original outputs are never overwritten")
    source_layout = load_layout(source)
    if source_layout and not options.dry_run:
        initialize_layout(output, source_layout["data_root"])
    plan = plan_human_repairs(source, output, manifest=manifest, limit=options.limit,
                              relative_path=options.relative_path, job_id=options.job_id)
    print("[PLAN] " + json.dumps(plan["counts"], ensure_ascii=False), flush=True)
    for row in plan["inventory"]:
        if row["disposition"] in {"source_failed", "invalid_checkpoint", "identity_conflict", "skipped_informal_name"}:
            print(f"[INVENTORY] {row['source_report']}: {row['disposition']}: {row.get('reason')}; {row.get('source_error')}", flush=True)
    if options.dry_run:
        for job in plan["jobs"]:
            print(f"[SELECTED] {job['relative_path']} <- {job['source_report']}", flush=True)
        print("[DRY-RUN] No writes, SDK imports or native processes.", flush=True)
        return 2 if any(plan["counts"][key] for key in ("source_failed", "invalid_checkpoint", "identity_conflict")) else 0
    roots = [Path(root) for root in plan["source_roots"]]
    _assert_sources_idle(roots)
    executable = Path(options.shogun_cl).expanduser().resolve() if options.shogun_cl else discover_shogun()
    if not executable or not executable.is_file() or executable.name.casefold() != "shogunpostcl.exe":
        raise ValueError("Pass an existing ShogunPostCL.exe with --shogun-cl")
    from pipeline_session import check_sdk
    check_sdk(str(executable))
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".pipeline.lock"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-humans-" + uuid.uuid4().hex[:8]
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError(f"Existing scheduler/stale lock; inspect its owner: {lock_path}") from None
    try:
        with lock:
            json.dump({"pid": os.getpid(), "run_id": run_id, "source_output": str(source)}, lock)
            lock.flush()
            _assert_sources_idle(roots)
            plan = plan_human_repairs(source, output, manifest=manifest, limit=options.limit,
                                      relative_path=options.relative_path, job_id=options.job_id)
            _assert_sources_idle([Path(root) for root in plan["source_roots"]])
            fields = ("job_id", "relative_path", "input_path", "source_report", "source_report_sha256",
                      "checkpoint_vdf", "checkpoint_completion", "expected_subjects", "expected_props",
                      "expected_human_count", "roster_resolution", "review_policy", "repair_checkpoint_props")
            snapshot = {key: plan[key] for key in ("counts", "inventory", "source_roots", "roster_pending")}
            snapshot["jobs"] = [{key: job[key] for key in fields} for job in plan["jobs"]]
            write_json(output / "repair_plan.json", dict(snapshot, run_id=run_id))
            write_json(output / "roster_pending.json", {"items": plan["roster_pending"], "run_id": run_id})
            launch = {"run_id": run_id, "scheduler_pid": os.getpid(), "started_at": _now(),
                      "source_output": str(source), "output_dir": str(output), "max_workers": options.max_workers,
                      "command": [sys.executable, *sys.argv], "plan_counts": plan["counts"],
                      "log": str(output / "production.log"), "recheck_timeout": options.recheck_timeout,
                      "kind": "human_repair", "planned_total": len(plan["jobs"])}
            write_json(output / "production_launch.json", launch)
            write_json(output / "launch_history" / (run_id + ".json"), launch)
            register_run(output, kind=launch["kind"], run_id=run_id, source_output=source,
                         started_at=launch["started_at"], scheduler_pid=launch["scheduler_pid"],
                         planned_total=launch["planned_total"])
            selection = _resume_selection(plan["jobs"], output)
            for job, stage in selection["recoverable"]:
                publish(job, stage, output, False)
            selection = _resume_selection(plan["jobs"], output)
            for item in selection["retry_reports"]:
                report = Path(item["report_path"])
                identity = report_relative(output, report)
                paths = take_paths(output, identity)
                relative = report.relative_to(paths["take_root"]) if load_layout(output) else report.relative_to(output)
                write_json(paths["history_root"] / run_id / relative, item["report_data"])
                sidecar = report.with_name("independent_recheck.json")
                if sidecar.is_file():
                    if not in_run(output, sidecar):
                        raise ValueError("Independent recheck sidecar escapes this run")
                    shutil.copy2(sidecar, paths["history_root"] / run_id / relative.with_name(sidecar.name))
            for job in selection["pending"]:
                for field in ("log_path", "native_log_path"):
                    old = Path(job[field])
                    if old.is_file():
                        paths = take_paths(output, job["relative_path"])
                        root = paths["take_root"] if load_layout(output) else output
                        archived = paths["history_root"] / run_id / old.relative_to(root)
                        archived.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(old, archived)
            rechecks = []
            for job in selection["preserved"]:
                rechecks.append(_recheck_saved(job, output, executable, options.recheck_timeout))
            # Same-policy batches reuse the mature scheduler without changing its
            # worker API. Drain each batch before rechecking: at most max_workers
            # repair natives OR one fresh recheck native, never competing pools.
            groups = {}
            for job in selection["pending"]:
                key = json.dumps(job["review_policy"], sort_keys=True, allow_nan=False)
                groups.setdefault(key, []).append(job)
            for group in groups.values():
                for offset in range(0, len(group), options.max_workers):
                    chunk = group[offset:offset + options.max_workers]
                    run_jobs(chunk, output, executable, options.max_workers, None,
                             options.startup_timeout, chunk[0]["review_policy"], False)
                    for job in chunk:
                        rechecks.append(_recheck_saved(job, output, executable, options.recheck_timeout))
            summary = summarize_outputs(plan["jobs"], rechecks, plan["counts"])
            summary.update(run_id=run_id, source_output=str(source), max_workers=options.max_workers)
            write_json(output / ("batch-" + run_id + ".json"), summary)
            write_json(output / "batch_report.json", summary)
            write_json(output / "batch_human_repair_summary.json", summary)
            print("[SUMMARY] " + json.dumps({key: summary[key] for key in
                                              ("counts", "recheck_counts", "inventory_counts")}), flush=True)
            if summary["counts"]["FAILED"] or summary["recheck_counts"]["FAILED"]:
                return 1
            return 2 if summary["counts"]["NEEDS_REVIEW"] or any(plan["counts"][key] for key in
                        ("source_failed", "invalid_checkpoint", "identity_conflict", "skipped_informal_name", "deferred")) else 0
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    mp.freeze_support()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[INTERRUPTED] Owned workers stopped; checkpoints/logs retained for --resume.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
