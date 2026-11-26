"""Per-process Shogun take processing; no SDK connection or file discovery at import.

Final usability is evaluated after repair, occlusion fill and solving. Pre-fill
coverage is retained as audit evidence, not a final rejection. Native capture and
reload requests persist as a native_diagnostics JSONL artifact before each call;
FAILED requests retain their arguments, scene source and range in the report.
SDK existence does not establish measured provenance, label accuracy or identity.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from math import isfinite
from pathlib import Path
from time import perf_counter
import shutil
import hashlib
import json
from datetime import datetime, timezone

from pipeline_session import load_native_file


# Kept identical to CombinedProcessing.hsl / the installed default operations.
_RECONSTRUCT = """
resetScene -keepClipsAndCal;
grayscalefitOptions -grayscaleCircleFittingEnabled true -numThreads 0;
reconstructOptions
-environmentalDriftTolerance 1.5
-minCamsToStartTrajectory 3
-minCamsToContinueTraj 2
-reconMinSeparation 0.0
-minCentroidRadius 0.0
-maxCentroidRadius 50.0
-minReconstructionRadius 0.0
-maxReconstructionRadius 1000.0
-computeRadius true
-fitMethod "3DPredictions"
-predMatchFactor 2.0
-minTrajLength 1
-startupError 150.0
-predictionError 150.0
-minVolume <<-100000, -100000, -100000>>
-maxVolume <<100000, 100000, 100000>>
-numThreads 2;
reconstruct;
"""
_LABEL_OPTIONS = """
quickPostLabelOptions
-useLabelingClusters {clusters}
-labelCompletenessEntranceTheshold 0.85
-useRobustBooting {robust}
-bootingQuality 0.0
-bootingVersusTracking 0.0
-trackingQuality 0.0
-smoothingFactor 25.0
-singlePass false
-clearExistingLabels false;
"""
_OCCLUSION = """
setOcclusionFixMethod "kinematic";
quickPostOcclusionFixOptions -applyFixedMarkers true -backup true -markerSmoothing 0.3 -dataFidelity 0.0 -transitionTime 0.1;
quickPost "fixocclusiononly";
"""
# DefaultOps/FilterData.hsl: MarkersOnly, AllTime, AllTranslations.
# The installed cutoff is 0.100 (not an asserted Hz value); threshold is 35.000.
_FILTER_MARKERS = (
    "selectByType Marker; selectRange -all; selectProperty Translation; filter 0.100 35.000;"
)
_FILTER_PARAMETERS = {
    "operation": "filter", "cutoff": 0.100, "threshold": 35.000,
    "objects": "MarkersOnly", "ranges": "AllTime", "channels": "AllTranslations",
    "parameters_source": "Scripts/DefaultOps/FilterData.hsl",
}
_SOLVE_OPTIONS = (
    "setUseAxiomSolving true; "
    "quickPostSolveOptions -priorImportance 0.03 -meanPoseRatio 0.0 -numThreads 0;"
)


def marker_coverage(
    marker_frames: dict[str, int], frame_count: int,
    core_markers: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Summarize per-marker existence counts in an inclusive processing range.

    Configured-but-absent core markers remain in the denominator. An omitted
    core set means every marker configured on the subject, not only keyed ones.
    """
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if any(not isinstance(n, int) or n < 0 or n > frame_count for n in marker_frames.values()):
        raise ValueError("marker counts must be integers within the processing range")
    core = list(marker_frames) if core_markers is None else list(core_markers)
    if len(core) != len(set(core)) or any(not isinstance(name, str) or not name for name in core):
        raise ValueError("core marker names must be unique nonempty strings")
    total = sum(marker_frames.values())
    core_total = sum(marker_frames.get(name, 0) for name in core)
    return {
        "expected_markers": len(marker_frames),
        "keyed_markers": sum(n > 0 for n in marker_frames.values()),
        "keyed_marker_frames": total,
        "frame_count": frame_count,
        "coverage": total / (len(marker_frames) * frame_count) if marker_frames else 0.0,
        "core_markers": core,
        "core_coverage": core_total / (len(core) * frame_count) if core else 0.0,
        "missing_core_markers": [name for name in core if marker_frames.get(name, 0) == 0],
        "unconfigured_core_markers": [name for name in core if name not in marker_frames],
        "marker_frames": dict(marker_frames),
    }


def classify_subject(
    marker_count: int, labeling_bone_count: int, solving_bone_count: int,
    coverage: dict, min_coverage: float = 1.0,
) -> str:
    """Return MISSING_SETUP, ZERO_DATA, PARTIAL or PRESENT (not identity approval)."""
    if not isfinite(min_coverage) or not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be finite and between zero and one")
    if marker_count <= 0 or labeling_bone_count <= 0 or solving_bone_count <= 0:
        return "MISSING_SETUP"
    if coverage["keyed_marker_frames"] == 0:
        return "ZERO_DATA"
    if (not coverage["core_markers"] or coverage["missing_core_markers"]
            or coverage["core_coverage"] < min_coverage):
        return "PARTIAL"
    return "PRESENT"


def evaluate_subject_quality(setup: dict, coverage: dict, solve: dict,
                             min_coverage: float = 1.0, geometry: dict | None = None) -> dict:
    """Judge final availability AND physical marker geometry, not pre-fill history."""
    classification = classify_subject(
        setup["markers"], setup["labeling_bones"], setup["solving_bones"],
        coverage, min_coverage)
    reasons = [] if classification == "PRESENT" else [classification.lower()]
    for skeleton in ("labeling", "solving"):
        metrics = solve.get(skeleton, {})
        if metrics.get("keyed_bones", 0) == 0 or metrics.get("animated_frames", 0) == 0:
            reasons.append(f"{skeleton}_skeleton_has_no_animation")
        elif metrics["animation_coverage"] < min_coverage:
            reasons.append(f"{skeleton}_skeleton_animation_partial")
    blocking_reasons = list(reasons)
    if geometry is None:
        reasons.append("human_geometry_evidence_unavailable:not_inspected")
        blocking_reasons.append("human_geometry_evidence_unavailable:not_inspected")
    else:
        reasons.extend(geometry.get("review_reasons", []))
        if not geometry.get("usable", True):
            blocking_reasons.extend(geometry.get("review_reasons", []) or ["human_geometry_unusable"])
    return {"classification": classification, "review_reasons": reasons,
            "usable": not blocking_reasons}


def roster_review_reasons(human_names: list[str], expected: list[str] | None, repair_subject: str | None,
                          tolerate_missing: bool = False) -> list[str]:
    """A named repair is target-only, not an assertion of the complete cast."""
    if repair_subject:
        return []
    if expected is None:
        return [] if tolerate_missing else ["expected_roster_unknown"]
    return [f"unexpected_human:{name}" for name in human_names if name not in expected]


def _quote(text: str) -> str:
    if any(ord(char) < 32 for char in text):
        raise ValueError("HSL strings must not contain control characters")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _sdk_context():
    # These shipped wrappers use the SAME global client as session.api; they
    # neither construct ViconShogunPost nor reconnect to a default GUI instance.
    from ViconShogunPostSDK.Scene import Scene
    from ViconShogunPostSDK.Timeline import Timeline
    return Scene(), Timeline()


def _direct_native_call(api, operation, *args):
    return operation(*args)


class _NativeDiagnostics:
    """Persist the exact in-flight native request before it can block or fail."""

    def __init__(self, report, path, session):
        self.path = Path(path)
        self.identity = {"native_pid": getattr(session, "pid", None),
                         "control_port": getattr(session, "port", None)}
        self.summary = {"journal_file": self.path.name, "completed_calls": 0,
                        "failed_calls": []}
        report["native_diagnostics"] = self.summary
        report.setdefault("artifacts", {})["native_diagnostics"] = self.path.name

    def _record(self, entry):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(entry, **self.identity), ensure_ascii=False,
                                    allow_nan=False) + "\n")
            stream.flush()

    def __call__(self, api, operation, *args):
        if api == "LoadFile":
            self.summary["scene_source"] = args[0]
            self.summary["range"] = {}
        entry = {"api": api, "arguments": [value if isinstance(value, (str, int, float, bool)) or value is None
                                          else repr(value)[:200] for value in args],
                 "status": "RUNNING",
                 "scene_source": self.summary.get("scene_source"),
                 "range": dict(self.summary.get("range", {})),
                 "started_at": datetime.now(timezone.utc).isoformat()}
        self.summary["last_call"] = entry
        self._record(entry)
        started = perf_counter()
        try:
            result = operation(*args)
        except Exception as error:
            entry.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                         elapsed_seconds=round(perf_counter() - started, 6))
            self.summary["failed_calls"].append(dict(entry))
            self._record(entry)
            error.add_note(f"Native request: {api}{args!r}; journal: {self.path}")
            raise
        else:
            range_keys = {"Timeline.GetAnimationStart": "animation_start",
                          "Timeline.GetAnimationEnd": "animation_end",
                          "Timeline.GetPlayStart": "play_start",
                          "Timeline.GetPlayEnd": "play_end"}
            if api in range_keys:
                self.summary.setdefault("range", {})[range_keys[api]] = int(result)
            entry.update(status="SUCCESS", elapsed_seconds=round(perf_counter() - started, 6))
            self.summary["completed_calls"] += 1
            self._record(entry)
            return result


_STAGE_OBSERVER = ContextVar("pipeline_stage_observer", default=None)


@contextmanager
def stage_observer(observer):
    """Observe stages only in this execution context, never in serialized policy.

    ``before_stage(report, entry)`` may gate execution; ``after_stage`` observes
    the completed result. Observers must not change native processing policy.
    Nested contexts restore the previous observer even when processing exits.
    """
    token = _STAGE_OBSERVER.set(observer)
    try:
        yield
    finally:
        _STAGE_OBSERVER.reset(token)


@contextmanager
def _stage(report: dict, name: str, subject: str | None = None):
    entry = {"name": name, "status": "RUNNING"}
    if subject is not None:
        entry["subject"] = subject
    report["stages"].append(entry)
    observer = _STAGE_OBSERVER.get()
    if observer is not None:
        observer.before_stage(report, entry)
    started = perf_counter()
    try:
        yield entry
    except Exception as exc:
        entry.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        entry["status"] = "SUCCESS"
    finally:
        entry["elapsed_seconds"] = round(perf_counter() - started, 6)
        if observer is not None:
            observer.after_stage(report, entry)


def _skip(report: dict, name: str, reason: str, subject: str | None = None):
    entry = {"name": name, "status": "SKIPPED", "reason": reason, "elapsed_seconds": 0.0}
    if subject is not None:
        entry["subject"] = subject
    report["stages"].append(entry)
    observer = _STAGE_OBSERVER.get()
    if observer is not None:
        observer.after_stage(report, entry)


def _ranges(timeline, native_call=None) -> dict:
    # SDK GetTrajectory/GetBone*Animation explicitly return animation-range
    # arrays. The start is NOT assumed to be zero, one, or the play-range start.
    call = native_call or _direct_native_call
    a = int(call("Timeline.GetAnimationStart", timeline.GetAnimationStart))
    b = int(call("Timeline.GetAnimationEnd", timeline.GetAnimationEnd))
    p = int(call("Timeline.GetPlayStart", timeline.GetPlayStart))
    q = int(call("Timeline.GetPlayEnd", timeline.GetPlayEnd))
    if b < a or q < p or p < a or q > b:
        raise RuntimeError(f"Invalid animation/play ranges: animation={a}..{b}, play={p}..{q}")
    # Loading a VDF does not evaluate its active scene in Post 1.21. Without an
    # explicit time evaluation GetTrajectory can report all existence masks
    # false despite stored keys. Frame evaluation changes no marker animation.
    call("Timeline.SetTimeFrames", timeline.SetTimeFrames, p)
    return {"animation_start": a, "animation_end": b, "play_start": p, "play_end": q,
            "animation_frame_count": b - a + 1, "frame_count": q - p + 1,
            "slice_start": p - a, "slice_stop": q - a + 1}


def _mask(animation, ranges: dict) -> list[bool]:
    if len(animation) != 4 or any(len(values) != ranges["animation_frame_count"] for values in animation):
        raise RuntimeError("SDK animation length does not match its reported inclusive animation range")
    x, y, z, exists = animation
    start, stop = ranges["slice_start"], ranges["slice_stop"]
    result = []
    for index in range(start, stop):
        present = bool(exists[index])
        if present and not all(isfinite(axis[index]) for axis in (x, y, z)):
            raise RuntimeError("Existing SDK animation sample contains nonfinite coordinates")
        result.append(present)
    return result


def _inspect(api, name: str, markers: list[str], ranges: dict, core: list[str] | None) -> dict:
    from pipeline_review_trim import unavailable_intervals
    counts = {}
    animation_total = 0
    core_names = markers if core is None else core
    core_present = [True] * ranges["frame_count"]
    for marker in markers:
        trajectory = api.GetTrajectory(name, marker)
        present = _mask(trajectory, ranges)
        counts[marker] = sum(present)
        if marker in core_names:
            core_present = [left and right for left, right in zip(core_present, present)]
        animation_total += sum(bool(value) for value in trajectory[3])
    if not core_names or set(core_names) - set(markers):
        core_present = [False] * ranges["frame_count"]
    metrics = marker_coverage(counts, ranges["frame_count"], core)
    metrics["animation_keyed_marker_frames"] = animation_total
    metrics["measurement"] = "SDK trajectory existence in the requested play range; not measured provenance"
    metrics["core_unavailable_intervals"] = unavailable_intervals(core_present, ranges["play_start"])
    return metrics


def _select(scene, character):
    # Selection by object ID, not wildcard/name matching. Refuse to execute a
    # selected operation unless the sole selected object is the exact target.
    scene.DeselectAllObjects()
    scene.SelectObject(character)
    selected = scene.GetSelectedObjects()
    if len(selected) != 1 or selected[0] != character or selected[0].Type != "Character":
        raise RuntimeError(f"Unable to select the exact sole Character {character.Name!r}")


def _descendants(session, scene, character, kind: str, *, native_call=None) -> list:
    call = native_call or _direct_native_call
    path = call("Character.Path", lambda: character.Path)
    call("HSL.selectByType", session.hsl, f"selectByType {kind} -childOf {_quote(path)};")
    return call("Scene.GetSelectedObjects", scene.GetSelectedObjects)


def _bone_metrics(session, api, name: str, skeleton: str, bones: list, ranges: dict) -> dict:
    from pipeline_review_trim import unavailable_intervals
    frames = ranges["frame_count"]
    present = [False] * frames
    rows = []
    for bone in bones:
        bone_name = bone.Name
        translations = _mask(api.GetBoneTranslationAnimation(name, bone_name, skeleton, "Local"), ranges)
        rotations = _mask(api.GetBoneRotationAnimation(name, bone_name, skeleton, "Local"), ranges)
        # Actual channel keys, in addition to the animation mask, distinguish a
        # skeleton definition/static offset from a solve with animated channels.
        translation_keys = int(str(session.hsl(f"getNumKeys Translation -onMod {_quote(bone.Path)};")).strip())
        rotation_keys = int(str(session.hsl(f"getNumKeys Rotation -onMod {_quote(bone.Path)};")).strip())
        if translation_keys < 0 or rotation_keys < 0:
            raise RuntimeError(f"Negative bone key count for {name}/{bone_name}")
        for index in range(frames):
            present[index] |= ((translations[index] and translation_keys > 0)
                               or (rotations[index] and rotation_keys > 0))
        rows.append({"name": bone_name, "translation_frames": sum(translations),
                     "rotation_frames": sum(rotations), "translation_keys": translation_keys,
                     "rotation_keys": rotation_keys})
    keyed = sum(row["translation_keys"] > 0 or row["rotation_keys"] > 0 for row in rows)
    return {"bones": len(bones), "keyed_bones": keyed,
            "animated_frames": sum(present), "frame_count": frames,
            "animation_coverage": sum(present) / frames, "bone_metrics": rows,
            "unavailable_intervals": unavailable_intervals(present, ranges["play_start"]),
            "validation": "at least one keyed bone channel per available frame; not every bone must be keyed"}


def inspect_final_quality(session, report: dict, detected: dict, ranges: dict,
                          minimum: float, core_sets: dict, rigid_options,
                          check_props: bool,
                          background_props: set[str] | frozenset[str] | None = None,
                          references: dict | None = None, native_call=None) -> list[str]:
    """Read final SDK state without filling, solving, or changing animation."""
    from pipeline_props import validate_prop_poses
    with _stage(report, "inspect_after_processing"):
        for name, (_, markers, _, _, row) in detected.items():
            if not row["processing_target"]:
                continue
            row["after"] = _inspect(session.api, name, markers, ranges, core_sets.get(name))
            from pipeline_human import inspect_human_geometry
            row["geometry"] = inspect_human_geometry(
                session.api, name, markers, ranges, detected[name][2], detected[name][3],
                (references or {}).get(name))
            row["postprocess_gain_marker_frames"] = (
                row["after"]["keyed_marker_frames"] - row["before_fill"]["keyed_marker_frames"]
                if row["after"]["frame_count"] == row["before_fill"]["frame_count"] else None)
    if check_props:
        with _stage(report, "validate_prop_poses"):
            review_mode = report.get("review_repair", False)
            blocking = validate_prop_poses(
                session, report["props"], ranges, rigid_options,
                set() if review_mode else background_props,
                include_background_reasons=review_mode, native_call=native_call)
            if review_mode:
                from pipeline_review_trim import required_prop_names
                required = set(required_prop_names(
                    report["props"], report["policy"]["camera_exemptions"]))
                waived = [p for p in report["props"]
                          if p["name"] not in required and p.get("review_reasons")]
                report["camera_exemptions"] = [
                    {"name": p["name"], "review_reasons": list(p["review_reasons"]),
                     "unresolved_intervals": deepcopy(p.get("unresolved_intervals", [])),
                     "repair_attempted": True} for p in waived]
                blocking = [reason for reason in blocking
                            if any(reason.startswith(f"prop:{name}:") for name in required)]
                report["quality"]["scope"] = {
                    "required_props": sorted(required),
                    "waived_failed_cameras": [p["name"] for p in waived],
                    "waiver": "only explicitly permitted cameras after repair attempts",
                    "all_non_camera_background_props_required": True}
            report["review_reasons"].extend(blocking)
            bg_reasons = [f"prop:{p['name']}:{r}" for p in report["props"]
                          if p.get("is_background") for r in p.get("review_reasons", [])]
            if bg_reasons:
                report["background_review_reasons"] = bg_reasons
    with _stage(report, "validate_skeleton_animation"):
        for name, (_, _, labeling, solving, row) in detected.items():
            if not row["processing_target"] or not labeling or not solving:
                continue
            row["solve"] = {}
            for skeleton, bones in (("Labeling", labeling), ("Solving", solving)):
                row["solve"][skeleton.lower()] = _bone_metrics(
                    session, session.api, name, skeleton, bones, ranges)
    context_reasons = list(report["review_reasons"])
    for row in report["subjects"]:
        if row.get("processing_target") and row.get("after") is not None:
            verdict = evaluate_subject_quality(
                row["setup"], row["after"], row.get("solve", {}), minimum, row.get("geometry"))
            row["classification"] = verdict["classification"]
            row["review_reasons"].extend(verdict["review_reasons"])
            row["quality"] = {"basis": "post_processing",
                              "usable": bool(verdict["usable"])}
            if not verdict["usable"]:
                report["review_reasons"].extend(f"{row['name']}:{reason}" for reason in row["review_reasons"])
        elif row.get("review_reasons"):
            report["review_reasons"].extend(f"{row['name']}:{reason}" for reason in row["review_reasons"])
    return context_reasons


def update_subject_export_quality(report: dict, context_reasons: list[str]) -> None:
    """Apply each person's final result plus the scene/prop context it exports."""
    humans = {row["name"]: row for row in report["subjects"]}
    for entry in report["subject_exports"]:
        row = humans[entry["subject"]]
        reasons = list(row["review_reasons"])
        if not row.get("processing_target"):
            reasons.append("outside_expected_subject_roster")
        entry["marker_classification"] = row["classification"]
        entry["human_review_reasons"] = reasons
        export_reasons = context_reasons + [f"{entry['subject']}:{reason}" for reason in reasons]
        entry["quality"] = {"basis": "post_processing",
                            "usable": row["quality"]["usable"] and not export_reasons,
                            "review_reasons": export_reasons}
        if "scope" in report.get("quality", {}):
            entry["quality"]["scope"] = deepcopy(report["quality"]["scope"])


def _dataset_scene(session, scene, job, report, *, native_call=None):
    """Derive names from this take's rigs; MP drops props before any labeling."""
    call = native_call or _direct_native_call
    humans, props = [], []
    for character in call("Scene.Objects.FilterByType", scene.Objects.FilterByType, "Character"):
        labeling = _descendants(session, scene, character, "LabelingBone", native_call=call)
        solving = _descendants(session, scene, character, "SolvingBone", native_call=call)
        # Retain damaged multi-bone human setups so inspection can report them.
        human = bool(labeling and solving) or len(labeling) > 1 or len(solving) > 1
        if human:
            humans.append(call("Character.Name", lambda: character.Name))
        else:
            props.append(call("Character.Name", lambda: character.Name))
            if not job.get("props_enabled", True):
                call("Scene.RemoveObject", lambda: scene.RemoveObject(character))
    call("Scene.DeselectAllObjects", scene.DeselectAllObjects)
    expected_count = job.get("expected_human_count", 2)
    report["dataset"] = {
        "sequence_kind": job.get("sequence_kind"), "expected_human_count": expected_count,
        "detected_humans": humans, "roster_source": "loaded_take_rig_definitions",
        "props_enabled": job.get("props_enabled", True),
        "excluded_props": props if not job.get("props_enabled", True) else [],
        "split_context": "all_props" if job.get("props_enabled", True) else "human_only",
    }
    if expected_count is not None and len(humans) != expected_count and not job.get("tolerate_missing_subjects"):
        report["review_reasons"].append(
            f"human_count_mismatch:expected={expected_count}:detected={len(humans)}")
    return humans, props


def process_take(session, job: dict, stage_dir: Path | str, policy: dict) -> dict:
    """Capture source evidence, repair/solve, then save/reload/recheck and export.

    Returns JSON-compatible report data; publishing and report-file writing belong
    to the scheduler. Hard SDK failures return FAILED with an inspectable native
    checkpoint where saving remains possible. No source file is ever saved over.
    A validated checkpoint job may set repair_checkpoint_props=False to retain
    prop bindings during human repair; final full-scene quality checks still run.
    Review-repair jobs otherwise retain their existing prop-repair default.
    """
    review_mode = bool(job.get("review_repair"))
    if review_mode:
        from pipeline_review_trim import DEFAULT_REVIEW_POLICY
        source_policy = job.get("review_policy", {})
        # New policy controls repair/camera/trim behavior; original per-take
        # quality thresholds remain authoritative and are never relaxed.
        thresholds = {key: source_policy[key] for key in
                      ("min_coverage", "core_markers", "prop_tolerance",
                       "prop_max_step", "ignore_labeling_clusters") if key in source_policy}
        policy = {**DEFAULT_REVIEW_POLICY, **policy, **thresholds, "repair_subject": None}
    report = {
        "status": "FAILED", "subjects": [], "props": [], "subject_exports": [], "stages": [], "artifacts": {},
        "review_reasons": [],
        "native_loads": [],
        "quality": {"schema_version": 2, "human_geometry_version": 1,
                    "basis": "post_processing", "usable": False},
        "limits": {
            "identity_validated": False, "label_accuracy_validated": False,
            "existence_is_measured_provenance": False,
            "vdf_source_provenance_unknown": job.get("input_kind") == "vdf",
            "roster_completeness": ("target_only" if policy.get("repair_subject") else
                                    "known" if job.get("expected_subjects") is not None else "unknown"),
            "prop_roster_completeness": ("target_only_skipped" if policy.get("repair_subject") else
                                        "known" if job.get("expected_props") is not None else "unknown"),
            "coverage_basis": "post-processing SDK marker existence in the requested play range",
        },
    }
    if review_mode:
        report["review_repair"] = True
        report["roster_resolution"] = deepcopy(job.get("roster_resolution", {}))
        report["source_report"] = job.get("source_report")
        report["source_report_sha256"] = job.get("source_report_sha256")
    loaded = False
    native_saved = False
    output_paths = {}
    try:
        source = Path(job.get("checkpoint_vdf") or job["input_path"])
        directory = Path(stage_dir).resolve()
        native_call = _NativeDiagnostics(report, directory / "native_diagnostics.jsonl", session)
        # Stage 0001: read-only roster parse of the original input (subjects /
        # rigid bodies / structure). A degraded scan is recorded, never fatal.
        with _stage(report, "parse") as parse_stage:
            try:
                import sequence_inspect
                parse_stage["result"] = sequence_inspect.parse_step(job["input_path"])
            except Exception as exc:
                parse_stage["result"] = {"warning": f"{type(exc).__name__}: {exc}"}
        kind = "vdf" if job.get("checkpoint_vdf") else job["input_kind"]
        checkpoint_audit = job.get("checkpoint_audit") if job.get("checkpoint_vdf") else None
        if review_mode and checkpoint_audit is None:
            raise ValueError("Review repair requires a validated completed checkpoint")
        if job.get("checkpoint_vdf"):
            if not isinstance(checkpoint_audit, dict) or checkpoint_audit.get("job_id") != job["job_id"]:
                raise ValueError("Completed checkpoint audit does not match this job")
            if (checkpoint_audit.get("input_path") != job["input_path"]
                    or Path(checkpoint_audit["artifacts"]["vdf"]).resolve() != source.resolve()):
                raise ValueError("Completed checkpoint source does not match this job")
            completed_steps = {entry["name"] for entry in checkpoint_audit["stages"]
                               if entry["status"] == "SUCCESS"}
            if not {"fix_occlusion", "solve_labeling", "solve_solving"} <= completed_steps:
                if not review_mode:
                    raise ValueError("Checkpoint has not completed fill and both solve stages")
                from pipeline_review_plan import checkpoint_completion
                completion = checkpoint_completion(
                    checkpoint_audit, Path(job["source_output"]), source)
                if completion != job.get("checkpoint_completion"):
                    raise ValueError("Recovered checkpoint completion evidence changed")
            if review_mode:
                fingerprint = job.get("checkpoint_fingerprint")
                stat = source.stat()
                if fingerprint != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
                    raise ValueError("Review checkpoint fingerprint changed")
                report_source = Path(job["source_report"])
                if hashlib.sha256(report_source.read_bytes()).hexdigest() != job.get("source_report_sha256"):
                    raise ValueError("Review source report hash changed")
            report["processing_source"] = {"path": str(source), "kind": "vdf",
                                           "checkpoint_resume_without_fill": True,
                                           "completed_steps_reused": ["fix_occlusion", "solve_labeling", "solve_solving"]}
            filter_evidence = [deepcopy(entry) for entry in checkpoint_audit["stages"]
                               if entry["name"] == "filter_markers"]
            report["processing_source"]["filter_markers_evidence_reused"] = filter_evidence
            if "filter_markers" in completed_steps:
                report["processing_source"]["completed_steps_reused"].append("filter_markers")
            report["limits"]["vdf_source_provenance_unknown"] = True
            if review_mode:
                report["processing_source"].update(
                    review_mutations_enabled=True,
                    checkpoint_fingerprint=deepcopy(job["checkpoint_fingerprint"]))
                report["processing_source"]["completion_evidence"] = deepcopy(
                    job.get("checkpoint_completion"))
        minimum = float(policy.get("min_coverage", 1.0))
        if not isfinite(minimum) or not 0.0 <= minimum <= 1.0:
            raise ValueError("min_coverage must be finite and between zero and one")
        from pipeline_jitter import JitterOptions
        from pipeline_rigid import RigidOptions
        jitter_options = JitterOptions(
            # An already-running scheduler retains its pre-feature policy; its
            # later workers must not silently acquire a new mutation stage.
            max_correction_mm=policy.get("prop_jitter_max_correction", 0.0),
            max_gap_frames=policy.get("prop_jitter_max_gap", 5),
            window_frames=policy.get("prop_jitter_window", 3))
        rigid_options = RigidOptions(
            tolerance_mm=float(policy.get("prop_tolerance", 2.0)),
            max_step_mm=float(policy.get("prop_max_step", 200.0)))
        if kind not in {"mcp", "vdf"} or source.suffix.lower() != "." + kind:
            raise ValueError("input_kind must match an MCP or VDF source")
        if not source.is_absolute() or not source.is_file() or not directory.is_dir():
            raise ValueError("Source must be an existing absolute file and stage_dir must already exist")
        output_paths = {role: directory / (source.stem + "." + role)
                        for role in (["vdf", "fbx", "vsk"] if job["input_kind"] == "mcp" else ["vdf", "fbx"])}
        if any(path.resolve() == source.resolve() or path.exists() for path in output_paths.values()):
            raise ValueError("Refusing to overwrite a source or preexisting staged artifact")
        expected = job.get("expected_subjects")
        if expected is not None and (not isinstance(expected, list) or not expected
                or any(not isinstance(name, str) or not name for name in expected)
                or len(expected) != len(set(expected))):
            raise ValueError("expected_subjects must be a nonempty unique string list or None")
        core_sets = policy.get("core_markers") or {}
        if not isinstance(core_sets, dict):
            raise ValueError("core_markers must map subject names to marker-name lists")
        for names in core_sets.values():
            if not isinstance(names, list) or not names:
                raise ValueError("Each configured core marker set must be a nonempty list")
            marker_coverage({}, 1, names)
        restricted = policy.get("repair_subject")
        report["policy"] = {"repair_subject": restricted, "min_coverage": minimum,
                            "ignore_labeling_clusters": bool(policy.get("ignore_labeling_clusters", False)),
                            "core_markers": core_sets,
                            "prop_tolerance": rigid_options.tolerance_mm,
                            "prop_max_step": rigid_options.max_step_mm,
                            "prop_jitter_max_correction": jitter_options.max_correction_mm,
                            "prop_jitter_max_gap": jitter_options.max_gap_frames,
                            "prop_jitter_window": jitter_options.window_frames,
                            "filter_markers": bool(policy.get("filter_markers", False)),
                            "filter_markers_executed": False,
                            "marker_filter_parameters": dict(_FILTER_PARAMETERS)}
        if review_mode:
            report["policy"].update(
                camera_exemptions=list(policy["camera_exemptions"]), auto_trim=bool(policy["auto_trim"]),
                trim_max_loss_fraction=policy["trim_max_loss_fraction"],
                trim_min_frames=policy["trim_min_frames"])
        scene, timeline = _sdk_context()
        with _stage(report, "load"):
            native_call("Scene.New", scene.New)
            load_native_file(session, source, native_call=native_call, audit=report["native_loads"])
            loaded = True
        references = {}
        if kind == "mcp":
            from pipeline_human_reference import capture_source_labels
            with _stage(report, "capture_original_live_labels") as capture_stage:
                references = capture_source_labels(
                    session, native_call=native_call, audit=capture_stage)
        elif job.get("input_kind") == "mcp":
            from pipeline_human_reference import capture_source_labels
            with _stage(report, "capture_original_live_labels") as capture_stage:
                native_call("Scene.New", scene.New)
                load_native_file(session, job["input_path"], native_call=native_call, audit=report["native_loads"])
                references = capture_source_labels(
                    session, native_call=native_call, audit=capture_stage)
                native_call("Scene.New", scene.New)
                load_native_file(session, source, native_call=native_call, audit=report["native_loads"])
        prop_expected = job.get("expected_props")
        props_enabled = job.get("props_enabled", True)
        # Human-only families (MP/BODY) always run this stage: prop subjects are
        # removed from the scene so they never reach the final FBX, regardless of
        # whether the optional dataset-mode roster audit is enabled.
        if job.get("dataset_mode") or not props_enabled:
            with _stage(report, "configure_dataset_scene"):
                model_humans, model_props = _dataset_scene(
                    session, scene, job, report, native_call=native_call)
                if expected is None:
                    expected = model_humans
                    report["limits"]["roster_completeness"] = "template_derived_count_checked"
                if not props_enabled:
                    prop_expected = []
                    report["limits"]["prop_roster_completeness"] = "disabled_human_only_MP_or_BODY"
                elif prop_expected is None:
                    prop_expected = model_props
                    report["limits"]["prop_roster_completeness"] = "template_derived"
                report["effective_subjects"] = expected
                report["effective_props"] = prop_expected
        if review_mode:
            resolution = job.get("roster_resolution", {})
            if resolution.get("pending") or (
                    job.get("expected_human_count") == 1
                    and not job.get("confirmed_subject_roster", False)):
                report["review_reasons"].append("confirmed_subject_roster_required")
        if kind == "mcp":
            with _stage(report, "reconstruct"):
                session.hsl(_RECONSTRUCT)
            with _stage(report, "initial_label"):
                # The initial pass is the unchanged CombinedProcessing pass;
                # ignore-clusters is an explicit REPAIR policy, not a new default.
                session.hsl(_LABEL_OPTIONS.format(clusters="true", robust="false") + "axiomLabel;")
        else:
            _skip(report, "reconstruct", "VDF source retained; no reset or reconstruction")
            _skip(report, "initial_label", "VDF existing labels retained")
        ranges = _ranges(timeline, native_call=native_call)
        if checkpoint_audit is not None and any(
                ranges[key] != checkpoint_audit["range"][key]
                for key in ("animation_start", "animation_end", "play_start", "play_end")):
            raise ValueError("Loaded checkpoint range differs from its completion evidence")
        report["range"] = {key: value for key, value in ranges.items() if not key.startswith("slice_")}
        characters = scene.Objects.FilterByType("Character")
        by_name = {}
        for character in characters:
            name = character.Name
            if name in by_name:
                raise RuntimeError(f"Character name is ambiguous: {name!r}")
            by_name[name] = character
        detected = {}
        with _stage(report, "inspect_before_fill"):
            for name, character in by_name.items():
                markers = list(session.api.GetMarkerNames(name))
                if len(markers) != len(set(markers)):
                    raise RuntimeError(f"Marker names are ambiguous for {name!r}")
                labeling = _descendants(session, scene, character, "LabelingBone", native_call=native_call)
                solving = _descendants(session, scene, character, "SolvingBone", native_call=native_call)
                clusters = _descendants(session, scene, character, "LabelingCluster", native_call=native_call)
                retargeting = _descendants(session, scene, character, "RetargetingBone", native_call=native_call)
                human = bool(labeling and solving)
                requested = name in expected if expected is not None else (name == restricted if restricted else human)
                incomplete = requested and not human
                before = _inspect(session.api, name, markers, ranges, core_sets.get(name))
                classification = (classify_subject(len(markers), len(labeling), len(solving), before, minimum)
                                  if requested or human else "NOT_APPLICABLE")
                row = {"name": name, "expected": name in expected if expected is not None else None,
                       "processing_target": requested,
                       "rig_class": "HUMAN_RIG" if human else ("INCOMPLETE_HUMAN_RIG" if incomplete else "PROP_OR_NONHUMAN"),
                       "setup": {"markers": len(markers), "labeling_bones": len(labeling),
                                 "solving_bones": len(solving), "labeling_clusters": len(clusters),
                                 "retargeting_bones": len(retargeting)},
                       "before": before, "before_fill": before, "after": None,
                       "classification": classification, "classification_before_fill": classification,
                       "repair_executed": False, "repair_reason": "not_evaluated", "repair_warnings": [],
                       "quality": {"basis": "post_processing", "usable": False},
                       "identity_validated": False, "review_reasons": []}
                report["subjects"].append(row)
                detected[name] = (character, markers, labeling, solving, row)
                if incomplete and not human:
                    report["review_reasons"].append(f"incomplete_human_setup:{name}")
            requested_names = list(expected) if expected is not None else ([restricted] if restricted else [
                name for name, (_, _, _, _, row) in detected.items() if row["processing_target"]])
            for name in requested_names:
                if name not in detected:
                    if job.get("tolerate_missing_subjects"):
                        report["limits"].setdefault("tolerated_absent_subjects", []).append(name)
                        continue
                    report["subjects"].append({"name": name, "expected": True if expected is not None else None,
                        "processing_target": True, "classification": "MISSING_SETUP", "before": None,
                        "before_fill": None, "after": None, "repair_executed": False, "repair_reason": "subject_absent",
                        "quality": {"basis": "post_processing", "usable": False},
                        "identity_validated": False, "review_reasons": ["subject_absent"]})
                    report["review_reasons"].append(f"missing_setup:{name}")
        if job.get("tolerate_missing_subjects"):
            # Single-person body takes: an actor with no captured data — or a
            # negligible amount of it (<1% coverage, e.g. a stray 3-frame walk-
            # in) — is excluded from the whole run: removed from the scene
            # (never exported), not repaired/solved, no review reasons. The
            # take's quality is judged on the surviving actors.
            def _negligible(row):
                if row.get("classification") in ("ZERO_DATA", "MISSING_SETUP"):
                    return True
                if row.get("classification") == "PARTIAL":
                    for phase in ("before_fill", "before"):
                        cov = (row.get(phase) or {}).get("coverage")
                        if isinstance(cov, (int, float)):
                            return cov < 0.01
                return False
            for row in report["subjects"]:
                if (row.get("rig_class") == "HUMAN_RIG" and row.get("processing_target")
                        and _negligible(row)):
                    entry = detected.get(row["name"])
                    if entry is not None:
                        native_call("Scene.RemoveObject", lambda node=entry[0]: scene.RemoveObject(node))
                        del detected[row["name"]]
                    row["processing_target"] = False
                    row["tolerated_no_data"] = True
                    row["quality"] = {"basis": "tolerated_no_data", "usable": True}
                    report["limits"].setdefault("tolerated_no_data_subjects", []).append(row["name"])
            humans = [row for row in report["subjects"] if row.get("rig_class") == "HUMAN_RIG"]
            if humans and all(row.get("tolerated_no_data") for row in humans):
                report["review_reasons"].append("no_data_subjects:all_humans_excluded")
        report["review_reasons"].extend(roster_review_reasons(
            [row["name"] for row in report["subjects"] if row.get("rig_class") == "HUMAN_RIG"],
            expected, restricted, bool(job.get("tolerate_missing_subjects"))))
        if review_mode and not job.get("dataset_mode"):
            expected_count = job.get("expected_human_count")
            actual_count = sum(row.get("rig_class") == "HUMAN_RIG" for row in report["subjects"])
            if expected_count is None:
                report["review_reasons"].append("expected_human_count_unknown")
            elif actual_count != expected_count:
                report["review_reasons"].append(
                    f"human_count_mismatch:expected={expected_count}:detected={actual_count}")
        if not requested_names:
            report["review_reasons"].append("no_human_processing_targets")
        from pipeline_props import process_props, repair_prop_jitter
        prop_solve = []
        preserve_checkpoint_props = checkpoint_audit is not None and (
            not review_mode or job.get("repair_checkpoint_props", True) is False)
        if checkpoint_audit is not None:
            report["processing_source"].update(
                prop_bindings_retained=preserve_checkpoint_props,
                repair_scope="humans_only" if preserve_checkpoint_props else "humans_and_props",
                final_validation_scope="humans_and_props")
        if preserve_checkpoint_props:
            report["props"] = deepcopy(checkpoint_audit.get("props", []))
            _skip(report, "inspect_repair_props", "completed_checkpoint_bindings_retained")
        elif restricted or not props_enabled:
            _skip(report, "inspect_repair_props", "legacy_target_only_or_MP_human_only")
        else:
            if review_mode:
                # Attempt every actual single-rigid setup, even if an old
                # manifest omitted furniture or other background props.
                discovered_props = [name for name, (_, _, labeling, solving, _) in detected.items()
                                    if len(labeling) == 1 and not solving]
                if prop_expected is None:
                    report["review_reasons"].append("expected_prop_roster_unknown")
                else:
                    report["review_reasons"].extend(
                        f"unexpected_prop:{name}" for name in discovered_props if name not in prop_expected)
                prop_expected = list(dict.fromkeys([*(prop_expected or []), *discovered_props]))
            with _stage(report, "inspect_repair_props"):
                report["props"], prop_reasons, prop_solve = process_props(
                    session, scene, by_name, ranges, prop_expected,
                    rigid_options,
                    {name: (labeling, solving) for name, (_, _, labeling, solving, _) in detected.items()})
                report["prop_review_reasons_before_fill"] = prop_reasons
                report["review_reasons"].extend(reason for reason in prop_reasons if not reason.startswith("prop:"))
        for name, (character, markers, labeling, solving, row) in detected.items():
            if not row["processing_target"]:
                row["repair_reason"] = "not_an_expected_human_target"
                _skip(report, "repair", row["repair_reason"], name)
                continue
            before = row["before"]
            if checkpoint_audit is not None:
                row["repair_reason"] = "completed_checkpoint_bindings_retained"
            elif row["classification"] == "MISSING_SETUP":
                row["repair_reason"] = "missing_marker_template_or_labeling_or_solving_skeleton"
            elif before["animation_keyed_marker_frames"] > 0:
                row["repair_reason"] = "existing_marker_data_skip_whole_subject_relabel"
            elif restricted and name != restricted:
                row["repair_reason"] = "outside_explicit_legacy_repair_subject"
            else:
                row["repair_reason"] = "all_configured_marker_data_absent_in_animation_range"
                with _stage(report, "repair", name):
                    _select(scene, character)
                    row["repair_executed"] = True
                    session.hsl(_LABEL_OPTIONS.format(
                        clusters="false" if policy.get("ignore_labeling_clusters", False) else "true",
                        robust="true") + "axiomLabel -selected;")
                    row["before_fill"] = _inspect(session.api, name, markers, ranges, core_sets.get(name))
                    row["repair_gain_marker_frames"] = row["before_fill"]["keyed_marker_frames"] - before["keyed_marker_frames"]
                if row["repair_gain_marker_frames"] <= 0:
                    row["repair_warnings"].append("repair_no_data_gain")
            if not row["repair_executed"]:
                _skip(report, "repair", row["repair_reason"], name)
            row["classification_before_fill"] = classify_subject(
                len(markers), len(labeling), len(solving), row["before_fill"], minimum)
        if checkpoint_audit is not None:
            _skip(report, "fix_occlusion", "validated_completed_checkpoint_reused")
        elif kind == "mcp":
            with _stage(report, "fix_occlusion"):
                session.hsl(_OCCLUSION)
        else:
            _skip(report, "fix_occlusion", "VDF provenance unknown; never blindly fill existing or previously filled data")
        if checkpoint_audit is not None:
            _skip(report, "filter_markers", "validated_completed_checkpoint_reused; never re-filter")
        elif kind != "mcp":
            _skip(report, "filter_markers", "VDF provenance unknown; never re-filter")
        elif not report["policy"]["filter_markers"]:
            _skip(report, "filter_markers", "disabled_by_policy")
        else:
            with _stage(report, "filter_markers") as filter_stage:
                filter_stage.update(_FILTER_PARAMETERS, hsl=_FILTER_MARKERS)
                native_call("HSL.filter_markers", session.hsl, _FILTER_MARKERS)
                report["policy"]["filter_markers_executed"] = True
        if preserve_checkpoint_props:
            _skip(report, "repair_prop_jitter", "validated_completed_checkpoint_reused")
        elif restricted or not props_enabled:
            _skip(report, "repair_prop_jitter", "legacy_target_only_or_MP_human_only")
        elif jitter_options.max_correction_mm == 0:
            _skip(report, "repair_prop_jitter", "disabled_zero_max_correction")
        else:
            with _stage(report, "repair_prop_jitter"):
                corrected_props = repair_prop_jitter(
                    session, scene, by_name, report["props"], ranges, rigid_options, jitter_options)
                prop_solve = list(dict.fromkeys([*prop_solve, *corrected_props]))
        reference_corrected = set()
        reference_expected = {}
        if references:
            from pipeline_human_repair import apply_reference_relabel
            for name, (character, markers, labeling, solving, row) in detected.items():
                if not row["processing_target"] or name not in references or not labeling or not solving:
                    continue
                with _stage(report, "repair_original_live_label_disagreement", name):
                    audit, expected_state = apply_reference_relabel(
                        session, scene, by_name, name, references[name], ranges)
                    row["reference_repair"] = audit
                    reference_expected.update(expected_state)
                    if audit["corrections"]:
                        reference_corrected.add(name)
        if checkpoint_audit is not None and reference_corrected:
            report["processing_source"]["reference_repaired_humans"] = sorted(reference_corrected)
            report["processing_source"]["completed_steps_reused"] = [
                name for name in ("fix_occlusion", "filter_markers") if name in completed_steps]
            report["processing_source"]["targeted_short_model_fills"] = [
                fill for *_, row in detected.values()
                for fill in row.get("reference_repair", {}).get("short_model_fills", [])]
            report["processing_source"]["checkpoint_resume_without_fill"] = not bool(
                report["processing_source"]["targeted_short_model_fills"])
            for name in sorted(reference_corrected):
                character, _, _, _, row = detected[name]
                with _stage(report, "solve_reference_repaired_human", name):
                    _select(scene, character)
                    session.hsl('setOcclusionFixMethod "kinematic"; quickPostOcclusionFixOptions -applyFixedMarkers true -backup true -markerSmoothing 0.3 -dataFidelity 0.0 -transitionTime 0.1; quickPost "fixocclusiononly" -selected;')
                    _select(scene, character)
                    session.hsl(_SOLVE_OPTIONS + 'quickPost "solvelabelingonly" -selected;')
                    _select(scene, character)
                    session.hsl(_SOLVE_OPTIONS + "solve -selectedCharacters;")
                    if row["setup"]["retargeting_bones"]:
                        _select(scene, character)
                        session.hsl("retarget -selectedCharacters;")
        if kind == "mcp":
            # Official solve hygiene: cap joint speed/acceleration so
            # sub-millimetre marker noise cannot become non-physical motion.
            try:
                session.hsl("SetSolveOptions -MaxJointSpeed 15 -MaxJointAccel 60;")
                report["solve_limits"] = {"max_joint_speed_rad_s": 15.0, "max_joint_accel_rad_s2": 60.0}
            except Exception as exc:
                report["solve_limits"] = {"applied": False, "note": f"{type(exc).__name__}: {exc}"[:200]}
        if checkpoint_audit is not None:
            for completed_stage in ("solve_labeling", "solve_solving", "retarget"):
                _skip(report, completed_stage, "validated_completed_checkpoint_reused")
        elif kind == "mcp":
            with _stage(report, "solve_labeling"):
                session.hsl(_SOLVE_OPTIONS + 'quickPost "solvelabelingonly";')
            with _stage(report, "solve_solving"):
                session.hsl(_SOLVE_OPTIONS + "solve;")
            if any(row["setup"]["retargeting_bones"] for *_, row in detected.values()):
                with _stage(report, "retarget"):
                    session.hsl("retarget;")
            else:
                _skip(report, "retarget", "no_retargeting_setup")
        else:
            for name, (character, markers, labeling, solving, row) in detected.items():
                if not row["processing_target"] or not labeling or not solving or not row["before_fill"]["keyed_marker_frames"]:
                    _skip(report, "solve", "not_target_or_missing_setup_or_no_marker_data", name)
                    continue
                with _stage(report, "solve_labeling", name):
                    _select(scene, character)
                    session.hsl(_SOLVE_OPTIONS + 'quickPost "solvelabelingonly" -selected;')
                with _stage(report, "solve_solving", name):
                    _select(scene, character)
                    session.hsl(_SOLVE_OPTIONS + "solve -selectedCharacters;")
                if row["setup"]["retargeting_bones"]:
                    with _stage(report, "retarget", name):
                        _select(scene, character)
                        session.hsl("retarget -selectedCharacters;")
                else:
                    _skip(report, "retarget", "no_retargeting_setup", name)
            if prop_solve:
                with _stage(report, "solve_rigid_props"):
                    scene.DeselectAllObjects()
                    for name in prop_solve:
                        scene.SelectObject(by_name[name])
                    selected = {node._ID for node in scene.GetSelectedObjects()}
                    if selected != {by_name[name]._ID for name in prop_solve}:
                        raise RuntimeError("Unable to select the exact rigid prop set for solving")
                    session.hsl(_SOLVE_OPTIONS + 'quickPost "solvelabelingonly" -selected;')
        if review_mode and prop_solve:
            with _stage(report, "solve_rigid_props"):
                scene.DeselectAllObjects()
                for name in prop_solve:
                    scene.SelectObject(by_name[name])
                selected = {node._ID for node in scene.GetSelectedObjects()}
                if selected != {by_name[name]._ID for name in prop_solve}:
                    raise RuntimeError("Unable to select the exact rigid prop set for solving")
                session.hsl(_SOLVE_OPTIONS + 'quickPost "solvelabelingonly" -selected;')
        if policy.get("denoise_static"):
            with _stage(report, "denoise_static_segments") as entry:
                from pipeline_denoise import denoise_static_humans
                entry["result"] = denoise_static_humans(native_call, session.api, detected, ranges)
        else:
            _skip(report, "denoise_static_segments", "option_disabled")
        structural_reasons = list(report["review_reasons"])
        structural_subject_reasons = {row["name"]: list(row["review_reasons"])
                                     for row in report["subjects"]}
        bg_set = policy.get("background_props") if policy else None
        context_review_reasons = inspect_final_quality(
            session, report, detected, ranges, minimum, core_sets, rigid_options,
            not restricted and props_enabled, bg_set, references, native_call=native_call)
        if review_mode:
            report["camera_repair_audit"] = deepcopy(report.get("camera_exemptions", []))
        # Deselect before full-scene exports; never use saveFile's selected flag.
        scene.DeselectAllObjects()
        with _stage(report, "save_native"):
            if (checkpoint_audit is not None and not review_mode
                    and not reference_corrected and not report.get("dataset", {}).get("excluded_props")):
                shutil.copy2(source, output_paths["vdf"])
                report["processing_source"]["native_bytes_reused"] = True
            else:
                session.api.SaveFile(output_paths["vdf"].as_posix())
            native_saved = True
            report["artifacts"]["vdf"] = output_paths["vdf"].name
        with _stage(report, "reload_validate_native"):
            report["quality_before_native_reload"] = {
                "review_reasons": list(report["review_reasons"]),
                "subjects": {row["name"]: {"after": deepcopy(row.get("after")),
                                           "geometry": deepcopy(row.get("geometry"))}
                             for row in report["subjects"]}}
            native_call("Scene.New", scene.New)
            load_native_file(session, output_paths["vdf"], native_call=native_call, audit=report["native_loads"])
            scene, timeline = _sdk_context()
            reloaded_ranges = _ranges(timeline, native_call=native_call)
            if reloaded_ranges != ranges:
                raise RuntimeError("Saved native range changed after reload")
            characters = {}
            for node in native_call("Scene.Objects.FilterByType", scene.Objects.FilterByType, "Character"):
                name = native_call("Character.Name", lambda node=node: node.Name)
                characters[name] = node
            report["review_reasons"] = list(structural_reasons)
            for name, (_, markers, _, _, row) in list(detected.items()):
                row["review_reasons"] = list(structural_subject_reasons.get(name, []))
                if name not in characters:
                    row["after"] = None
                    row["classification"] = "MISSING_SETUP"
                    if job.get("tolerate_missing_subjects"):
                        row["processing_target"] = False
                        row["quality"] = {"basis": "post_processing", "usable": True}
                        report["limits"].setdefault("tolerated_absent_subjects", []).append(name)
                    else:
                        row["quality"] = {"basis": "post_processing", "usable": False}
                        row["review_reasons"].append("subject_absent_after_native_reload")
                        report["review_reasons"].append(f"{name}:subject_absent_after_native_reload")
                    del detected[name]
                    continue
                character = characters[name]
                reloaded_markers = list(native_call("GetMarkerNames", session.api.GetMarkerNames, name))
                if set(reloaded_markers) != set(markers):
                    raise RuntimeError(f"Saved native marker setup changed: {name}")
                detected[name] = (
                    character, reloaded_markers,
                    _descendants(session, scene, character, "LabelingBone", native_call=native_call),
                    _descendants(session, scene, character, "SolvingBone", native_call=native_call), row)
            context_review_reasons = inspect_final_quality(
                session, report, detected, ranges, minimum, core_sets, rigid_options,
                not restricted and props_enabled, bg_set, references, native_call=native_call)
            if reference_expected:
                from pipeline_human_repair import verify_reference_preservation
                violations = verify_reference_preservation(scene, reference_expected, ranges)
                report["reference_repair_preservation"] = {"passed": not violations, "violations": violations}
                report["review_reasons"].extend(violations)
                context_review_reasons.extend(violations)
        with _stage(report, "export_fbx"):
            session.hsl("fbxExportOptions -exportActiveClipOnly true -collapseSubjects false "
                        "-exportPlayRangeOnly false -exportLabelingSetups true;")
            session.hsl(f"saveFile {_quote(output_paths['fbx'].as_posix())};")
            report["artifacts"]["fbx"] = output_paths["fbx"].name
        # Human-only families: the clean directly-usable FBX is produced by the
        # lightweight display trim (configured per run); archiving routes that
        # clean file to 03_exports while 02_processed keeps the full working set.
        if review_mode:
            from pipeline_review_trim import apply_review_trim, fbx_key_bounds
            context_review_reasons = apply_review_trim(
                session, scene, timeline, report, job, output_paths, detected,
                minimum, core_sets, rigid_options, props_enabled,
                structural_reasons, structural_subject_reasons, context_review_reasons, references)
            report["fbx_key_boundaries"] = fbx_key_bounds(output_paths["fbx"])
        if "vsk" in output_paths:
            with _stage(report, "export_vsk"):
                session.hsl(f"saveFile {_quote(output_paths['vsk'].as_posix())};")
                report["artifacts"]["vsk"] = output_paths["vsk"].name
        if job.get("split_subjects"):
            from pipeline_split import split_subject_fbx
            # Tolerantly excluded actors were removed from the scene, so they
            # have no FBX root; only processing targets are split out.
            human_names = [row["name"] for row in report["subjects"]
                           if row.get("rig_class") == "HUMAN_RIG" and row.get("processing_target")]
            if not human_names:
                report["review_reasons"].append("no_exportable_human_subjects")
            else:
                with _stage(report, "export_subject_sequences"):
                    human_models = set(report.get("dataset", {}).get("detected_humans", human_names))
                    prop_names = ([name for name in by_name if name not in human_models]
                                  if props_enabled else [])
                    entries = split_subject_fbx(output_paths["fbx"], human_names, prop_names,
                                                directory / "subjects", props_enabled)
                    for index, entry in enumerate(entries):
                        role = f"subject_{index}_fbx"
                        report["artifacts"][role] = entry["relative_path"]
                        entry["artifact_role"] = role
                        if report.get("review_trim", {}).get("applied"):
                            from pipeline_review_trim import validate_fbx_clip
                            entry["timing_validation"] = validate_fbx_clip(
                                directory / entry["relative_path"],
                                report["review_trim"]["full_fbx_key_boundaries"],
                                report["full_range_quality"]["range"],
                                report["review_trim"]["selected"], report["timing_map"]["fps"])
                    report["subject_exports"] = entries
                    update_subject_export_quality(report, context_review_reasons)
        report["review_reasons"] = list(dict.fromkeys(report["review_reasons"]))
        report["status"] = "NEEDS_REVIEW" if report["review_reasons"] else "SUCCESS"
        report["quality"]["usable"] = report["status"] == "SUCCESS"
    except Exception as exc:
        report["status"] = "FAILED"
        report["quality"]["usable"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        if loaded and not native_saved and "vdf" in output_paths and not output_paths["vdf"].exists():
            try:
                with _stage(report, "save_failure_checkpoint"):
                    session.api.SaveFile(output_paths["vdf"].as_posix())
                    report["artifacts"]["vdf"] = output_paths["vdf"].name
            except Exception as checkpoint_error:
                report["checkpoint_error"] = f"{type(checkpoint_error).__name__}: {checkpoint_error}"
    return report
