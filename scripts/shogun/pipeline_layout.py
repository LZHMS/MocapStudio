"""Validated per-sequence storage with a separate batch control directory.

A layout manifest opts a dataset run into sequence-local storage. Generic runs
without a manifest retain their independent, centralized output layout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import uuid

MANIFEST = ".sequence-layout.json"
_BUCKETS = {".work", "_failed", ".quality_recheck_attempts", "_resume_history"}


def _relative(value: str | Path) -> Path:
    text = str(value).replace("\\", "/")
    path = Path(text)
    if (not text or path.is_absolute() or ":" in text
            or any(part in ("", ".", "..") for part in text.split("/"))):
        raise ValueError(f"Unsafe take identity: {value}")
    return path


def _version(value: str) -> str:
    if (not isinstance(value, str) or not value or value in (".", "..")
            or any(char in value for char in '/\\:<>"|?*')
            or value.rstrip(". ") != value):
        raise ValueError(f"Unsafe layout version key: {value!r}")
    return value


def _plain(path: Path) -> bool:
    """A path through a junction/symlink must not authorize another location."""
    return os.path.normcase(os.path.abspath(path)) == os.path.normcase(str(path.resolve()))


def _spec(output: Path, data_root: Path, version_key: str | None = None) -> dict:
    output, data_root = output.absolute(), data_root.absolute()
    if not _plain(output) or not _plain(data_root):
        raise ValueError("Layout roots must not traverse a junction or symbolic link")
    output, data_root = output.resolve(), data_root.resolve()
    if output == data_root or data_root.is_relative_to(output):
        raise ValueError("Control root must not equal or contain the raw dataset")
    if output.is_relative_to(data_root):
        relative = output.relative_to(data_root)
        if len(relative.parts) < 2 or relative.parts[0].casefold() != "_processing":
            raise ValueError("Dataset-local control roots must be below _processing/<run>")
    return {"schema_version": 1, "data_root": str(data_root),
            "version_key": _version(version_key if version_key is not None else output.name)}


def load_layout(output: str | Path) -> dict | None:
    output = Path(output).absolute()
    manifest = output / MANIFEST
    if not manifest.exists():
        return None
    if not _plain(manifest):
        raise ValueError("Layout manifest must not traverse a junction or symbolic link")
    value = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"Unsupported sequence layout manifest: {manifest}")
    if not isinstance(value.get("data_root"), str) or not Path(value["data_root"]).is_absolute():
        raise ValueError("Layout data_root must be absolute")
    if not isinstance(value.get("version_key"), str):
        raise ValueError("Layout version_key is required")
    spec = _spec(output, Path(value["data_root"]), value["version_key"])
    if set(value) != set(spec):
        raise ValueError(f"Unknown sequence layout fields: {sorted(set(value) - set(spec))}")
    return spec


def initialize_layout(output: str | Path, data_root: str | Path,
                      version_key: str | None = None) -> dict:
    output = Path(output).absolute()
    existing = load_layout(output)
    selected_version = version_key if version_key is not None else (
        existing["version_key"] if existing else None)
    spec = _spec(output, Path(data_root), selected_version)
    if not Path(spec["data_root"]).is_dir():
        raise ValueError("Layout raw dataset root does not exist")
    if existing is not None:
        if existing != spec:
            raise ValueError("Existing sequence layout belongs to a different dataset/version")
        return existing
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / (MANIFEST + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(spec, ensure_ascii=False, indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        # Windows rename refuses an existing destination. On POSIX, linking
        # the fully written temporary file gives the same atomic no-clobber
        # guarantee; the temporary name is removed immediately below.
        if os.name == "nt":
            os.rename(temporary, output / MANIFEST)
        else:
            os.link(temporary, output / MANIFEST)
            descriptor = os.open(output, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return spec


def _local_take(spec: dict, relative: Path) -> Path:
    parts = relative.parts
    if (relative.suffix.casefold() not in (".mcp", ".vdf")
            or any(part.casefold() in ("processed", "_processing") for part in parts)):
        raise ValueError(f"Sequence layout requires a raw MCP/VDF identity: {relative}")
    root = Path(spec["data_root"])
    take = root / relative.parent / "processed" / spec["version_key"] / relative.name
    if not _plain(take):
        raise ValueError(f"Sequence output traverses a junction or symbolic link: {take}")
    return take


def take_paths(output: str | Path, relative: str | Path, *,
               data_root: str | Path | None = None,
               version_key: str | None = None) -> dict[str, Path]:
    output, relative = Path(output).absolute(), _relative(relative)
    spec = load_layout(output)
    if data_root is not None or version_key is not None:
        proposed = _spec(output,
                         Path(data_root) if data_root is not None else Path(spec["data_root"]),
                         version_key if version_key is not None
                         else (spec["version_key"] if spec else None))
        if spec is not None and proposed != spec:
            raise ValueError("Planned dataset differs from configured sequence layout")
        spec = proposed
    take = _local_take(spec, relative) if spec else output / "takes" / relative
    paths = {"take_root": take, "report_path": take / "report.json",
             "destination": take / "artifacts",
             "review_destination": (take / "_needs_review" / "artifacts" if spec else
                                    output / "_needs_review" / relative / "artifacts"),
             "log_dir": take / "logs" if spec else output / "logs" / relative}
    for bucket, key in ((".work", "work_root"), ("_failed", "failed_root"),
                        (".quality_recheck_attempts", "recheck_root"),
                        ("_resume_history", "history_root")):
        paths[key] = (take if spec else output) / bucket
    return paths


def _local_takes(spec: dict):
    root = Path(spec["data_root"])

    def scan_error(exc):
        raise exc

    # Visit raw directories only: processed data must never become new inputs.
    for directory, children, _ in os.walk(root, followlinks=False, onerror=scan_error):
        children[:] = sorted(name for name in children
                             if name.casefold() not in ("processed", "_processing"))
        sequence = Path(directory)
        for name in children:
            if not _plain(sequence / name):
                raise ValueError(f"Linked raw sequence directory: {sequence / name}")
        version = sequence / "processed" / spec["version_key"]
        try:
            version_info = version.stat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(version_info.st_mode):
            continue
        if not _plain(version):
            raise ValueError(f"Linked sequence version directory: {version}")
        for take in sorted(version.iterdir()):
            if take.suffix.casefold() not in (".mcp", ".vdf"):
                continue
            if stat.S_ISDIR(take.stat().st_mode):
                relative = sequence.relative_to(root) / take.name
                if take.absolute() != _local_take(spec, relative):
                    raise ValueError(f"Noncanonical sequence take folder: {take}")
                yield take


def report_paths(output: str | Path):
    output = Path(output).resolve()
    spec = load_layout(output)
    if spec is None:
        root = output / "takes"
        try:
            root_info = root.stat()
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(root_info.st_mode):
            raise NotADirectoryError(f"Canonical report root is not a directory: {root}")
        if not _plain(root):
            raise ValueError(f"Linked canonical report root: {root}")

        def scan_error(exc):
            raise exc

        candidates = []
        for directory, children, files in os.walk(root, followlinks=False, onerror=scan_error):
            for name in children:
                child = Path(directory) / name
                attributes = getattr(child.lstat(), "st_file_attributes", 0)
                if (child.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                        or not _plain(child)):
                    raise ValueError(f"Linked canonical report directory: {child}")
            candidates.extend(Path(directory) / name for name in files if Path(name).match("report.json"))
        yield from sorted(candidates)
    else:
        for take in _local_takes(spec):
            report = take / "report.json"
            try:
                report_info = report.stat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(report_info.st_mode):
                report_relative(output, report)
                yield report


def report_relative(output: str | Path, report_path: str | Path) -> Path:
    output, report = Path(output).resolve(), Path(report_path).absolute()
    if not _plain(report):
        raise ValueError("Canonical report traverses a junction or symbolic link")
    spec = load_layout(output)
    if spec is None:
        relative = report.parent.relative_to(output / "takes")
    else:
        if report.name != "report.json" or len(report.parents) < 4:
            raise ValueError("Not a canonical sequence report")
        relative = report.parents[3].relative_to(Path(spec["data_root"])) / report.parent.name
    relative = _relative(relative)
    if report != take_paths(output, relative)["report_path"]:
        raise ValueError("Report belongs to a different sequence or version")
    return relative


def artifact_roots(output: str | Path, relative: str | Path) -> tuple[Path, Path]:
    paths = take_paths(output, relative)
    return paths["destination"], paths["review_destination"]


def in_run(output: str | Path, path: str | Path) -> bool:
    output, path = Path(output).absolute(), Path(path).absolute()
    if not _plain(output) or not _plain(path):
        return False
    output, path = output.resolve(), path.resolve()
    if path.is_relative_to(output):
        return True
    spec = load_layout(output)
    if spec is None:
        return False
    root = Path(spec["data_root"])
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    for index, part in enumerate(parts):
        if part.casefold() != "processed" or index + 2 >= len(parts):
            continue
        relative = Path(*parts[:index]) / parts[index + 2]
        try:
            take = _local_take(spec, _relative(relative))
        except ValueError:
            return False
        return path.is_relative_to(take)
    return False


def bucket_entries(output: str | Path, bucket: str):
    if bucket not in _BUCKETS:
        raise ValueError(f"Unknown per-take bucket: {bucket}")
    output = Path(output).resolve()
    roots = [output / bucket]  # Explicitly unassigned historical diagnostics stay batch-wide.
    spec = load_layout(output)
    if spec is not None:
        roots.extend(take / bucket for take in _local_takes(spec))
    for root in roots:
        if root.is_dir():
            for entry in sorted(root.iterdir()):
                if not in_run(output, entry):
                    raise ValueError(f"Run bucket entry escapes authorized storage: {entry}")
                yield entry


def is_generated(path: str | Path, data_root: str | Path) -> bool:
    try:
        relative = Path(path).absolute().relative_to(Path(data_root).absolute())
    except ValueError:
        return False
    return any(part.casefold() in ("processed", "_processing") for part in relative.parts)


def source_runs(source: str | Path) -> list[Path]:
    source = Path(source).resolve()
    repairs = source / "_repair_runs"
    return [source, *sorted(path for path in repairs.iterdir() if path.is_dir())] if repairs.is_dir() else [source]
