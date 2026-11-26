"""Read-only, provenance-checked planning for completed native review repairs."""
from __future__ import annotations

import configparser
import hashlib
import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path, PurePosixPath

from pipeline_review_trim import DEFAULT_REVIEW_POLICY
from pipeline_layout import artifact_roots, in_run, load_layout, report_paths, report_relative, take_paths


_CAMERAS = frozenset({"camera_1", "camera_2", "camera_3"})
_COMPLETED = frozenset({"fix_occlusion", "solve_labeling", "solve_solving"})


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _identity(path: Path) -> str:
    return str(path.resolve()).casefold()


def _relative(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("Missing relative MCP identity")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ":" in normalized or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError("Unsafe relative MCP identity")
    if path.suffix.casefold() != ".mcp":
        raise ValueError("Review checkpoint must retain raw MCP identity")
    return path


def _nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Missing or empty file: {path}")


def _artifact(value: object, root: Path) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError("Artifact reference must be an absolute path")
    path = Path(value).resolve()
    if not in_run(root, Path(value)):
        raise ValueError(f"Artifact escapes declared root: {value}")
    _nonempty(path)
    return path


def _names(value: object, label: str, *, empty: bool = False) -> list[str] | None:
    if value is None:
        return None
    if (not isinstance(value, list) or (not value and not empty)
            or any(not isinstance(name, str) or not name.strip() for name in value)
            or len(value) != len(set(value))):
        raise ValueError(f"Invalid {label} roster")
    return list(value)


def _count(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError("Invalid expected human count")
    return value


def _capture_roster(report: dict, raw: Path) -> tuple[list[str] | None, dict]:
    """Use capture bookkeeping only when every name has an explicit rig class."""
    audit = {"status": "missing", "evidence": []}
    candidates = sorted(path for path in raw.parent.iterdir()
                        if path.suffix.casefold() == ".enf"
                        and (path.stem.casefold() == raw.stem.casefold()
                             or path.name.casefold().startswith(raw.stem.casefold() + ".")))
    if not candidates:
        return None, audit
    humans = {row["name"] for row in report.get("subjects", [])
              if row.get("rig_class") == "HUMAN_RIG"}
    props = {row["name"] for row in report.get("props", [])}
    props.update(row["name"] for row in report.get("subjects", [])
                 if row.get("rig_class") == "PROP_OR_NONHUMAN")
    # MP intentionally removes these typed nonhuman rigs before reconstruction.
    props.update(_names((report.get("dataset") or {}).get("excluded_props"),
                        "excluded props", empty=True) or [])
    # These exact names are explicitly identified as cameras by the repair policy.
    props.update(_CAMERAS)
    ambiguous = humans & props
    incomplete = {row["name"] for row in report.get("subjects", [])
                  if row.get("rig_class") == "INCOMPLETE_HUMAN_RIG"}
    props -= incomplete
    captured = None
    for path in candidates:
        try:
            if not _inside(path.resolve(), raw.parent) or path.stat().st_size > 1024 * 1024:
                raise ValueError("Unsafe capture metadata path or size")
            payload = path.read_bytes()
            parser = configparser.ConfigParser(interpolation=None)
            parser.read_string(payload.decode("utf-8-sig"))
            if (parser.get("Node Information", "TYPE").strip().casefold() != "trial"
                    or parser.get("Node Information", "NAME").strip().casefold() != raw.stem.casefold()):
                raise ValueError("Capture metadata NAME/TYPE does not match raw MCP")
            names = [name.strip() for name in parser.get("TRIAL_INFO", "SUBJECTS").split(",")]
            _names(names, "capture subjects")
            audit["evidence"].append({"path": str(path.resolve()),
                                      "sha256": hashlib.sha256(payload).hexdigest(),
                                      "capture_subjects": names})
            if captured is not None and set(captured) != set(names):
                audit.update(status="inconsistent", reason="Capture metadata rosters disagree")
                return None, audit
            captured = names
        except (OSError, ValueError, configparser.Error) as error:
            audit.update(status="invalid", reason=str(error))
            return None, audit
    unknown = [name for name in captured if name not in humans | props or name in ambiguous]
    human_roster = [name for name in captured if name in humans and name not in ambiguous]
    audit["capture_subjects"] = captured
    if unknown or not human_roster:
        audit.update(status="unclassified", unknown_subjects=unknown,
                     reason="Capture subjects are not all unambiguously classified")
        return None, audit
    audit["status"] = "confirmed"
    return human_roster, audit


def _roster(report: dict, relative: str, manifest: dict) -> tuple[list[str] | None, int | None, dict]:
    dataset = report.get("dataset") or {}
    original_count = dataset.get("expected_human_count")
    original = _count(report.get("expected_human_count") if original_count is None else original_count)
    detected = _names(dataset.get("detected_humans"), "detected humans", empty=True)
    if detected is None:
        detected = [row["name"] for row in report.get("subjects", [])
                    if row.get("processing_target") and isinstance(row.get("name"), str)
                    and row.get("classification") != "MISSING_SETUP"]
    take_rosters = {key.replace("\\", "/").casefold(): value
                    for key, value in manifest.get("takes", {}).items()}
    roster = take_rosters.get(relative.casefold())
    source = "manifest_take"
    if roster is None:
        roster = manifest.get("expected_subjects")
        source = "manifest_global"
    capture = None
    if roster is None:
        roster, capture = _capture_roster(report, Path(report["input_path"]).resolve())
        source = "capture_enf"
    if roster is None and capture["status"] == "missing":
        # Unlike effective_subjects, the original scheduler's expected roster
        # is explicit, provided its completeness audit does not say otherwise.
        roster = report.get("expected_subjects")
        source = "source_expected_subjects"
        if report.get("limits", {}).get("roster_completeness") in {"unknown", "template_derived"}:
            roster = None
    confirmed = roster is not None
    if confirmed:
        roster = _names(roster, source)
        expected_count = len(roster)
    else:
        source = "unconfirmed_source_count"
        expected_count = original
        roster = _names(report.get("effective_subjects"), "effective subjects") or detected or None
    mismatch = expected_count is None or len(detected) != expected_count
    pending = not confirmed and (mismatch or expected_count == 1
                                 or capture["status"] != "missing")
    resolution = {
        "source": source, "confirmed": confirmed,
        "source_expected_human_count": original, "expected_human_count": expected_count,
        "detected_humans": detected, "mismatch": mismatch, "pending": pending,
    }
    if capture is not None:
        resolution["capture"] = capture
    return roster, expected_count, resolution


def _range(report: dict) -> tuple[int, int]:
    ranges = report.get("range")
    keys = ("animation_start", "animation_end", "play_start", "play_end")
    if not isinstance(ranges, dict) or any(type(ranges.get(key)) is not int for key in keys):
        raise ValueError("Missing inclusive checkpoint range evidence")
    a, b, first, last = (ranges[key] for key in keys)
    if not a <= first <= last <= b:
        raise ValueError("Invalid inclusive checkpoint range")
    for key, expected in (("frame_count", last - first + 1), ("animation_frame_count", b - a + 1)):
        if key in ranges and (type(ranges[key]) is not int or ranges[key] != expected):
            raise ValueError(f"Inconsistent inclusive {key}")
    return first, last


def _priority(report: dict, first: int, last: int, roster: dict, props: list[str]) -> dict:
    intervals = []
    considered = []
    for row in report.get("props", []):
        name = row.get("name", "")
        if name.casefold() in _CAMERAS:
            continue
        considered.append(name)
        final = row.get("unresolved_intervals")
        bad_without_interval = (row.get("classification") != "PRESENT" or bool(row.get("review_reasons")))
        if final is None or (not final and bad_without_interval):
            intervals.append((first, last))
            continue
        if not isinstance(final, list):
            raise ValueError("Invalid final prop interval evidence")
        for interval in final:
            if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                    or any(type(value) is not int for value in interval)
                    or not first <= interval[0] <= interval[1] <= last):
                raise ValueError("Invalid final prop interval evidence")
            intervals.append(tuple(interval))
    for name in props:
        if name.casefold() not in _CAMERAS and name not in considered:
            considered.append(name)
            intervals.append((first, last))
    merged = []
    for begin, end in sorted(intervals):
        if merged and begin <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    subjects = [row for row in report.get("subjects", []) if row.get("processing_target")]
    human_pass = bool(subjects) and all(
        row.get("quality", {}).get("basis") == "post_processing"
        and row.get("quality", {}).get("usable") is True for row in subjects)
    bad = sum(end - begin + 1 for begin, end in merged)
    longest = max((end - begin + 1 for begin, end in merged), default=0)
    fraction = bad / (last - first + 1)
    sparse = 0 < fraction <= .01
    tier = 0 if sparse else 1 if bad == 0 else 2
    if not human_pass:
        tier = 3
    return {"tier": tier,
            "human_pass": human_pass, "sparse": sparse, "bad_frames": bad,
            "human_count_matches": not roster["mismatch"], "roster_pending": roster["pending"],
            "bad_fraction": fraction, "longest_bad_run": longest,
            "bad_intervals": merged, "non_camera_props": considered,
            "basis": "final_non_camera_prop_interval_union_and_post_processing_humans"}


def _review_checkpoint_completion(report: dict, source: Path, checkpoint: Path,
                                  completed: set[str]) -> dict:
    """Follow immutable report lineage without narrowing the original range."""
    source = Path(source).resolve()
    ancestor_value = report.get("source_output")
    if not isinstance(ancestor_value, str) or not Path(ancestor_value).is_absolute():
        raise ValueError("Review checkpoint lacks an absolute source output")
    ancestor = Path(ancestor_value).resolve()
    if ancestor == source or not _inside(source, ancestor / "_repair_runs"):
        raise ValueError("Review checkpoint lineage must point to an ancestor repair root")
    relative = _relative(report.get("relative_path")).as_posix()
    report_path = _artifact(report.get("source_report"), ancestor)
    if report_relative(ancestor, report_path).as_posix() != relative:
        raise ValueError("Review source report belongs to a different take")
    payload = report_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != report.get("source_report_sha256"):
        raise ValueError("Review source report checksum changed")
    previous = json.loads(payload.decode("utf-8-sig"))
    if (previous.get("job_id") != report.get("job_id")
            or previous.get("relative_path") != report.get("relative_path")
            or previous.get("input_kind") != "mcp"
            or _identity(Path(previous.get("input_path", "")))
            != _identity(Path(report["input_path"]))):
        raise ValueError("Review source report belongs to a different input")
    _range(report)
    _range(previous)
    for key in ("animation_start", "animation_end", "play_start", "play_end"):
        if report["range"][key] != previous["range"][key]:
            raise ValueError("Review checkpoint range differs; use the full source checkpoint")
    original = _artifact(previous.get("artifacts", {}).get("vdf"), ancestor)
    if not any(_inside(original, root.resolve()) for root in artifact_roots(ancestor, relative)):
        raise ValueError("Review source VDF belongs to a different take")
    recovery = report["processing_source"]
    if _identity(Path(recovery.get("path", ""))) != _identity(original):
        raise ValueError("Review checkpoint source VDF differs from its source report")
    fingerprint = recovery.get("checkpoint_fingerprint") or {}
    stat = original.stat()
    if fingerprint != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
        raise ValueError("Review source checkpoint fingerprint changed")
    if report.get("status") not in ("SUCCESS", "NEEDS_REVIEW") or "save_native" not in completed:
        raise ValueError("Review checkpoint was not successfully saved")
    if _artifact(report.get("artifacts", {}).get("vdf"), source) != Path(checkpoint).resolve():
        raise ValueError("Review completion evidence points to a different VDF")
    if not any(_inside(Path(checkpoint).resolve(), root.resolve()) for root in artifact_roots(source, relative)):
        raise ValueError("Review completion VDF belongs to a different take")
    inherited = checkpoint_completion(previous, ancestor, original)
    return {"basis": "validated_review_lineage", "completed_steps": sorted(_COMPLETED),
            "source_report": str(report_path), "source_report_sha256": digest,
            "source_completion": inherited}


def checkpoint_completion(report: dict, source: Path, checkpoint: Path) -> dict:
    """Prove native completion, byte-identical recovery, or validated review lineage."""
    completed = {entry.get("name") for entry in report.get("stages", [])
                 if isinstance(entry, dict) and entry.get("status") == "SUCCESS"}
    if _COMPLETED <= completed:
        return {"basis": "successful_native_stages", "completed_steps": sorted(_COMPLETED)}
    recovery = report.get("processing_source") or {}
    if (report.get("review_repair") is True and report.get("source_report") is not None
            and recovery.get("checkpoint_resume_without_fill") is True
            and _COMPLETED <= set(recovery.get("completed_steps_reused", []))):
        return _review_checkpoint_completion(report, source, checkpoint, completed)
    if (recovery.get("checkpoint_resume_without_fill") is not True
            or recovery.get("native_bytes_reused") is not True
            or not _COMPLETED <= set(recovery.get("completed_steps_reused", []))):
        raise ValueError("Missing successful fill and both native solve stages")
    original_path = recovery.get("path")
    if not isinstance(original_path, str) or not Path(original_path).is_absolute():
        raise ValueError("Recovered checkpoint has no absolute original source")
    source, checkpoint = Path(source).resolve(), Path(checkpoint).resolve()
    original = Path(original_path).resolve()
    failed_root = take_paths(source, _relative(report.get("relative_path")).as_posix())["failed_root"].resolve()
    if not in_run(source, original) or not _inside(original, failed_root) or original == checkpoint:
        raise ValueError("Recovered checkpoint original is not an independent failed-stage artifact")
    _nonempty(original)
    control = (original.parent / ".control" / "result.json").resolve()
    if not in_run(source, control) or not _inside(control, failed_root):
        raise ValueError("Recovered checkpoint control escapes failed-stage evidence")
    payload = control.read_bytes()
    previous = json.loads(payload.decode("utf-8-sig"))
    if (previous.get("job_id") != report.get("job_id")
            or _identity(Path(previous.get("input_path", ""))) != _identity(Path(report["input_path"]))):
        raise ValueError("Recovered checkpoint evidence belongs to another input")
    previous_stages = {entry.get("name") for entry in previous.get("stages", [])
                       if isinstance(entry, dict) and entry.get("status") == "SUCCESS"}
    if not _COMPLETED <= previous_stages:
        raise ValueError("Original failed-stage evidence did not complete both solves and fill")
    for key in ("animation_start", "animation_end", "play_start", "play_end"):
        if previous.get("range", {}).get(key) != report.get("range", {}).get(key):
            raise ValueError("Recovered checkpoint range differs from completed evidence")
    value = previous.get("artifacts", {}).get("vdf")
    if not isinstance(value, str):
        raise ValueError("Original completion evidence lacks its VDF")
    declared = Path(value)
    if (declared if declared.is_absolute() else original.parent / declared).resolve() != original:
        raise ValueError("Original completion evidence points to a different VDF")
    if checkpoint.stat().st_size != original.stat().st_size:
        raise ValueError("Recovered native bytes differ from completed checkpoint")
    with original.open("rb") as stream:
        original_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    with checkpoint.open("rb") as stream:
        current_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if original_sha != current_sha:
        raise ValueError("Recovered native bytes differ from completed checkpoint")
    return {"basis": "byte_identical_completed_recovery",
            "completed_steps": sorted(_COMPLETED), "original_vdf": str(original),
            "original_control_report": str(control),
            "original_control_sha256": hashlib.sha256(payload).hexdigest(),
            "native_vdf_sha256": current_sha}


def _make_job(report: dict, report_path: Path, payload: bytes, source: Path,
              output: Path, manifest: dict) -> dict:
    relative = _relative(report.get("relative_path")).as_posix()
    canonical = report_relative(source, report_path).as_posix()
    if relative.casefold() != canonical.casefold():
        raise ValueError("Canonical report location differs from raw relative identity")
    key = hashlib.sha256(relative.casefold().encode("utf-8")).hexdigest()[:16]
    if report.get("job_id") != key or report.get("input_kind") != "mcp":
        raise ValueError("Checkpoint job ID/input kind does not match raw MCP identity")
    raw_value = report.get("input_path")
    if not isinstance(raw_value, str) or not Path(raw_value).is_absolute():
        raise ValueError("Raw MCP input must be an absolute path")
    raw = Path(raw_value).resolve()
    _nonempty(raw)
    if tuple(part.casefold() for part in raw.parts[-len(PurePosixPath(relative).parts):]) != tuple(
            part.casefold() for part in PurePosixPath(relative).parts):
        raise ValueError("Raw MCP input differs from relative identity")
    source_layout = load_layout(source)
    if source_layout and raw != (Path(source_layout["data_root"]) / relative).resolve():
        raise ValueError("Raw MCP input differs from the configured dataset identity")
    if _inside(raw, output) or _inside(output, raw.parent):
        raise ValueError("Output overlaps raw input")
    first, last = _range(report)
    products = report.get("artifacts")
    if not isinstance(products, dict):
        raise ValueError("Missing native checkpoint artifacts")
    checkpoint = _artifact(products.get("vdf"), source)
    if checkpoint.suffix.casefold() != ".vdf" or _inside(checkpoint, output):
        raise ValueError("Unsafe native VDF checkpoint")
    checkpoint_roots = artifact_roots(source, relative)
    if not any(_inside(checkpoint, root) for root in checkpoint_roots):
        raise ValueError("VDF checkpoint belongs to a different take identity")
    completion = checkpoint_completion(report, source, checkpoint)
    # Reject escaped references even when only the VDF will be consumed. Missing
    # legacy FBX/VSK is not an obstacle to a fresh export from a valid VDF.
    for value in products.values():
        if (not isinstance(value, str) or not Path(value).is_absolute()
                or not in_run(source, Path(value))
                or not any(_inside(Path(value).resolve(), root.resolve()) for root in checkpoint_roots)):
            raise ValueError("Source artifact escapes declared root")
        if _inside(Path(value).resolve(), output):
            raise ValueError("Output overlaps source artifact")
    for entry in report.get("subject_exports", []):
        value = entry.get("path")
        if (not isinstance(value, str) or not Path(value).is_absolute()
                or not in_run(source, Path(value))
                or not any(_inside(Path(value).resolve(), root.resolve()) for root in checkpoint_roots)
                or _inside(Path(value).resolve(), output)):
            raise ValueError("Source subject export escapes declared root or overlaps output")
    expected, expected_count, resolution = _roster(report, relative, manifest)
    props = []
    for value in (report.get("expected_props"), report.get("effective_props"),
                  [row["name"] for row in report.get("props", [])],
                  [row["name"] for row in report.get("subjects", [])
                   if row.get("rig_class") == "PROP_OR_NONHUMAN"],
                  manifest.get("expected_props"), manifest.get("prop_takes", {}).get(relative.casefold())):
        for name in _names(value, "props", empty=True) or []:
            if name not in props:
                props.append(name)
    dataset = report.get("dataset") or {}
    if report.get("props_enabled", dataset.get("props_enabled", True)):
        for name in (resolution.get("capture") or {}).get("capture_subjects", []):
            if name in _CAMERAS and name not in props:
                props.append(name)
    policy = deepcopy(report.get("policy") or {})
    policy.update(DEFAULT_REVIEW_POLICY)
    paths = take_paths(output, relative, data_root=source_layout["data_root"] if source_layout else None)
    log_dir = paths["log_dir"]
    if source_layout or load_layout(output):
        source_take = take_paths(source, relative)["take_root"].resolve()
        if _inside(paths["take_root"].resolve(), source_take) or _inside(source_take, paths["take_root"].resolve()):
            raise ValueError("Output sequence version overlaps source results")
    checkpoint_stat = checkpoint.stat()
    job = {
        "job_id": key, "input_path": raw_value, "input_kind": "mcp", "relative_path": relative,
        "expected_subjects": expected, "expected_props": props or report.get("expected_props"),
        "dataset_mode": bool(report.get("dataset_mode", bool(dataset))),
        "sequence_kind": report.get("sequence_kind", dataset.get("sequence_kind")),
        "split_subjects": bool(report.get("split_subjects", bool(dataset))),
        "props_enabled": bool(report.get("props_enabled", dataset.get("props_enabled", True))),
        "expected_human_count": expected_count, "roster_resolution": resolution,
        "confirmed_subject_roster": resolution["confirmed"], "identity_validated": False,
        "destination": str(paths["destination"]),
        "review_destination": str(paths["review_destination"]),
        "report_path": str(paths["report_path"]),
        "log_path": str(log_dir / "processing.log"), "native_log_path": str(log_dir / "shogun.log"),
        "preflight_errors": [], "checkpoint_vdf": str(checkpoint), "checkpoint_audit": report,
        "review_repair": True, "source_report": str(report_path),
        "source_report_sha256": hashlib.sha256(payload).hexdigest(),
        "source_output": str(source), "checkpoint_completion": completion,
        "checkpoint_fingerprint": {"size": checkpoint_stat.st_size, "mtime_ns": checkpoint_stat.st_mtime_ns},
        "review_policy": policy, "priority": _priority(report, first, last, resolution, props),
    }
    if source_layout:
        job["layout_data_root"] = source_layout["data_root"]
    if "timecode" in products:
        job["source_timecode"] = str(_artifact(products["timecode"], source))
    for field in ("destination", "review_destination", "report_path", "log_path", "native_log_path"):
        root = paths["take_root"] if source_layout or load_layout(output) else output
        if not _inside(Path(job[field]).resolve(), root.resolve()):
            raise ValueError(f"Output path escapes output root: {field}")
    return job


def _resume(job: dict, output: Path) -> bool:
    report_path = Path(job["report_path"])
    if report_path.exists() or report_path.is_symlink():
        _nonempty(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        if (report.get("status") not in {"SUCCESS", "NEEDS_REVIEW"}
                or report.get("job_id") != job["job_id"]
                or _identity(Path(report.get("input_path", ""))) != _identity(Path(job["input_path"]))
                or report.get("source_report_sha256") != job["source_report_sha256"]):
            raise ValueError("Existing output report is incomplete or has different source provenance")
        artifacts = report.get("artifacts")
        if not isinstance(artifacts, dict) or not {"vdf", "fbx", "vsk"} <= artifacts.keys():
            raise ValueError("Existing output lacks complete native artifacts")
        roots = [Path(job[field]).resolve() for field in ("destination", "review_destination")]
        for value in artifacts.values():
            artifact = _artifact(value, output)
            if not any(_inside(artifact, root) for root in roots):
                raise ValueError("Existing artifact belongs to a different take")
        exports = report.get("subject_exports", [])
        if not isinstance(exports, list):
            raise ValueError("Invalid existing subject exports")
        for row in exports:
            artifact = _artifact(row.get("path"), output)
            if not any(_inside(artifact, root) for root in roots):
                raise ValueError("Existing subject export belongs to a different take")
        return True
    for field in ("destination", "review_destination"):
        directory = Path(job[field])
        if (directory.exists() or directory.is_symlink()) and (
                not directory.is_dir() or any(directory.rglob("*"))):
            raise ValueError("Unreported output artifacts would collide with repair")
    for field in ("log_path", "native_log_path"):
        if Path(job[field]).exists() or Path(job[field]).is_symlink():
            raise ValueError("Unreported output logs would be overwritten")
    return False


def plan_review_repairs(source_output: Path, output: Path, *, manifest: dict | None = None,
                        limit: int | None = None) -> dict:
    """Plan without writes; invalid/colliding candidates are explicitly skipped.

    ``jobs`` is the sparse-first pending queue after ``limit``. ``preserved``
    contains matching completed repair outputs. ``roster_pending`` covers the
    entire eligible queue, not just the limited launch. ``selected_total`` counts
    pending eligible jobs before the limit; ``eligible`` also includes preserved.
    """
    source, output = Path(source_output).resolve(), Path(output).resolve()
    if not source.is_dir() or (load_layout(source) is None and not (source / "takes").is_dir()):
        raise ValueError("Source output must contain canonical takes reports or a sequence layout")
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a nonnegative integer or None")
    if _inside(source, output):
        raise ValueError("Output must not equal or contain source output")
    if _inside(output, source):
        nested = output.relative_to(source).parts
        if len(nested) != 2 or nested[0] != "_repair_runs":
            raise ValueError("Nested output must be source/_repair_runs/<runid>")
    for bucket in ("takes", "_needs_review", ".work", "_failed", "logs"):
        if not _inside((output / bucket).resolve(), output):
            raise ValueError(f"Output bucket escapes output root: {bucket}")
    manifest = manifest or {}
    skipped, candidates = [], []
    scanned = candidate_count = 0
    for report_path in sorted(report_paths(source)):
        scanned += 1
        try:
            report_relative(source, report_path)
            payload = report_path.read_bytes()
            report = json.loads(payload.decode("utf-8-sig"))
            if report.get("status") != "NEEDS_REVIEW":
                continue
            if report.get("quality", {}).get("basis") != "post_processing":
                raise ValueError("Review quality is not post-processing evidence")
            jitter = report.get("policy", {}).get("prop_jitter_max_correction", 0)
            if isinstance(jitter, bool) or not isinstance(jitter, (float, int)) or not math.isfinite(jitter) or jitter < 0:
                raise ValueError("Invalid source jitter policy")
            if jitter != 0:
                continue
            candidate_count += 1
            candidates.append(_make_job(report, report_path, payload, source, output, manifest))
        except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
            skipped.append({"source_report": str(report_path), "reason": str(error)})
    ids = Counter(job["job_id"] for job in candidates)
    destinations = Counter(_identity(Path(job["destination"])) for job in candidates)
    jobs, preserved, roster_pending = [], [], []
    for job in candidates:
        try:
            if ids[job["job_id"]] != 1 or destinations[_identity(Path(job["destination"]))] != 1:
                raise ValueError("Duplicate job identity or output destination")
            completed = _resume(job, output)
            (preserved if completed else jobs).append(job)
            if job["roster_resolution"]["pending"]:
                roster_pending.append({"job_id": job["job_id"], "relative_path": job["relative_path"],
                                       "source_report": job["source_report"], **job["roster_resolution"]})
        except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
            skipped.append({"source_report": job["source_report"], "job_id": job["job_id"], "reason": str(error)})
    jobs.sort(key=lambda job: (job["priority"]["tier"], job["priority"]["bad_fraction"],
                               job["priority"]["longest_bad_run"], job["relative_path"].casefold()))
    total = len(jobs)
    jobs = jobs if limit is None else jobs[:limit]
    counts = {"reports_scanned": scanned, "candidates": candidate_count,
              "eligible": total + len(preserved), "selected_total": total, "selected": len(jobs),
              "preserved": len(preserved), "skipped": len(skipped), "roster_pending": len(roster_pending),
              "deferred_by_limit": total - len(jobs)}
    return {"jobs": jobs, "skipped": skipped, "roster_pending": roster_pending,
            "preserved": preserved, "counts": counts}
