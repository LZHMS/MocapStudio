"""Evidence-based supplemental moves for standalone repairs and mesh exports.

This module plans only. The organizer's journal owns all actual moves/rewrites.
No filename similarity or actor name is used to assign an output to a raw take.
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline_layout import is_generated, take_paths

_SOURCE_FIELDS = {"source", "reference", "reference_mcp", "raw_mcp", "input_path",
                  "input_vdf", "processed_vdf", "checkpoint_vdf", "processing_source.path",
                  "raw_mcp.source", "processed_vdf.source"}


def _fields(path: Path) -> dict:
    import ijson
    result = {}
    try:
        with path.open("rb") as stream:
            for prefix, event, value in ijson.parse(stream):
                if prefix in _SOURCE_FIELDS and event == "string":
                    result[prefix] = value
    except ijson.JSONError:
        # Older diagnostic jobs serialize missing observations as NaN. Parse
        # those bounded documents with the same permissive stdlib decoder,
        # but still require a complete document and an exact raw source path.
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError(f"Nonstandard diagnostic JSON exceeds bounded identity reader: {path}") from None
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        result = {}
        for field in _SOURCE_FIELDS:
            value = payload
            for part in field.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if isinstance(value, str):
                result[field] = value
    return result


def raw_identity(value: str | Path, data_root: Path, source_roots: list[Path]) -> Path | None:
    """Resolve a real raw path or an explicitly scoped canonical/failed artifact."""
    candidate = Path(value).absolute()
    data_root = data_root.resolve()
    if candidate.is_relative_to(data_root) and not is_generated(candidate, data_root):
        if candidate.is_file() and candidate.suffix.casefold() in (".mcp", ".vdf"):
            return candidate.relative_to(data_root)
        return None
    for source in source_roots:
        source = source.resolve()
        if not candidate.is_relative_to(source):
            continue
        relative = candidate.relative_to(source)
        if relative.parts and relative.parts[0] in ("takes", "_needs_review", "logs"):
            pieces = relative.parts[1:]
            for index, piece in enumerate(pieces):
                if Path(piece).suffix.casefold() != ".mcp":
                    continue
                raw = Path(*pieces[:index + 1])
                canonical = source / "takes" / raw / "report.json"
                if (data_root / raw).is_file() and canonical.is_file():
                    artifact_roots = [source / bucket / raw / "artifacts"
                                      for bucket in ("takes", "_needs_review")]
                    if any(candidate.is_relative_to(root) for root in artifact_roots):
                        return raw
        if len(relative.parts) >= 2 and relative.parts[0] in ("_failed", ".work"):
            control = source / relative.parts[0] / relative.parts[1] / ".control" / "result.json"
            if control.is_file():
                identity = _fields(control).get("input_path")
                if identity:
                    path = Path(identity).absolute()
                    if (path.is_relative_to(data_root) and path.is_file()
                            and not is_generated(path, data_root)):
                        return path.relative_to(data_root)
    return None


def supplemental_plan(data_root: Path, source_roots: list[Path], control_root: Path) -> dict:
    """Build exact child moves; ambiguous outputs stay in their batch control."""
    data_root, control_root = data_root.resolve(), control_root.resolve()
    source_roots = list(dict.fromkeys(Path(path).resolve() for path in source_roots))
    lookup_roots = list(source_roots)
    for source in source_roots:
        repairs = source / "_repair_runs"
        if repairs.is_dir():
            lookup_roots.extend(path for path in repairs.iterdir() if path.is_dir())
    result = {"moves": [], "controls": [], "warnings": [], "mesh_reports": []}
    seen_sources = set()

    def move(source, target):
        source, target = Path(source).resolve(), Path(target).absolute()
        if source in seen_sources:
            return
        seen_sources.add(source)
        result["moves"].append({"source": str(source), "target": str(target)})

    def control(source):
        target = control_root / source.name
        if not any(Path(item["source"]) == source for item in result["controls"]):
            result["controls"].append({"source": str(source), "target": str(target),
                                       "version_key": source.name})
        return target

    for source in source_roots:
        mesh_manifest = source / "mesh_batch_report.json"
        if mesh_manifest.is_file():
            batch = json.loads(mesh_manifest.read_text(encoding="utf-8-sig"))
            records = batch.get("files", [])
            if not isinstance(records, list):
                raise ValueError(f"Invalid mesh file list: {mesh_manifest}")
            new_control = control(source)
            move(source, new_control)
            grouped = {}
            for record in records:
                if not isinstance(record, dict) or not isinstance(record.get("input"), str):
                    continue
                identity = raw_identity(record["input"], data_root, lookup_roots)
                if not isinstance(record.get("output"), str):
                    result["warnings"].append(f"Mesh attempt has no output; audit retained in control: {record['input']}")
                    continue
                generated = Path(record["output"])
                if identity is None or not generated.is_absolute() or not generated.is_relative_to(source):
                    result["warnings"].append(f"Unassigned mesh output retained in control: {record.get('output')}")
                    continue
                paths = take_paths(new_control, identity, data_root=data_root)
                # Only explicit source artifacts establish the relative subject branch.
                input_file = Path(record["input"])
                artifact_index = next((index for index, part in enumerate(input_file.parts)
                                       if part == "artifacts"), None)
                tail = (Path(*input_file.parts[artifact_index + 1:]) if artifact_index is not None
                        else Path(input_file.name))
                destination = paths["destination"] / tail.parent / generated.name
                if generated.is_file():
                    move(generated, destination)
                report_value = record.get("report")
                report = Path(report_value) if isinstance(report_value, str) else None
                if report is None or not report.is_file():
                    derived = source / "_reports" / generated.relative_to(source)
                    # Existing reports use the input basename, not the generated suffix.
                    guessed = derived.with_name(input_file.name + ".json")
                    if guessed.is_file():
                        payload = json.loads(guessed.read_text(encoding="utf-8-sig"))
                        if payload.get("input") == record["input"]:
                            report = guessed
                if report is not None and report.is_file() and report.is_relative_to(source):
                    report_destination = paths["take_root"] / "mesh_reports" / tail.parent / report.name
                    move(report, report_destination)
                    try:
                        detail = json.loads(report.read_text(encoding="utf-8-sig"))
                    except (OSError, ValueError):
                        detail = {}
                    verification = detail.get("verification_path")
                    if isinstance(verification, str):
                        verification_file = Path(verification)
                        if verification_file.is_file() and verification_file.is_relative_to(source):
                            move(verification_file, report_destination.parent / verification_file.name)
                if generated.is_file():
                    group = grouped.setdefault(identity.as_posix(), {"report_path": str(paths["report_path"]),
                        "relative_path": identity.as_posix(), "input_path": str(data_root / identity),
                        "purpose": "mesh", "tracking_quality_changed": False,
                        "artifacts": {}, "files": []})
                    role = "scene_fbx" if tail.parent == Path(".") else f"subject_{len(group['artifacts'])}_fbx"
                    if role in group["artifacts"]:
                        role = f"variant_{len(group['artifacts'])}_fbx"
                    group["artifacts"][role] = str(destination)
                    group["files"].append({"input": record["input"], "output": str(destination),
                                            "status": record.get("status", "UNKNOWN")})
            result["mesh_reports"].extend(grouped.values())
            continue

        reports = list((source / "takes").rglob("report.json"))
        # Ordinary MCP runs are handled by the canonical organizer, not twice.
        if reports and any((data_root / path.parent.relative_to(source / "takes")).is_file()
                           for path in reports):
            continue
        evidence = [source / name for name in ("repair_audit.json", "diagnostic_job.json",
                    "fresh_diagnosis.json", "report.json", "result.json") if (source / name).is_file()]
        evidence.extend(reports)
        identities = set()
        for path in evidence:
            for value in _fields(path).values():
                identity = raw_identity(value, data_root, lookup_roots)
                if identity is not None:
                    identities.add(identity)
        if len(identities) != 1:
            result["warnings"].append(f"Standalone output not uniquely attributable; retained: {source} ({len(identities)} identities)")
            continue
        identity = identities.pop()
        new_control = control(source)
        paths = take_paths(new_control, identity, data_root=data_root)
        move(source, paths["take_root"] / "diagnostics")
    return result
