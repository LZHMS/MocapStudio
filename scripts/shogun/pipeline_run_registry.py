"""Best-effort output discovery for the independent MocapWeb dashboard.

Importing this module does not inspect outputs, create files or load the SDK.
Catalog records advertise roots, never process liveness or data usability.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import stat
import tempfile
import time
import uuid

from pipeline_layout import load_layout


DEFAULT_REGISTRY = "H:/MocapWeb/data/runs"
KINDS = frozenset({"processing", "review_repair", "human_repair", "quality_recheck", "human_target"})


def run_key(output_dir) -> str:
    return hashlib.sha256(str(Path(output_dir).resolve()).casefold().encode()).hexdigest()[:16]


def _absolute(value):
    return str(Path(value).expanduser().resolve()) if value is not None else None


def _data_path(output, value):
    """Keep catalog identity at the control root and raw input metadata separate."""
    if value is None:
        layout = load_layout(output)
        value = layout["data_root"] if layout else None
    return _absolute(value)


def _same_root(value, output):
    return isinstance(value, str) and _absolute(value).casefold() == str(output).casefold()


def _utc(value=None):
    moment = datetime.now(timezone.utc) if value is None else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("started_at must include a UTC offset")
    return moment.astimezone(timezone.utc).isoformat()


def open_shared_read(path):
    """Open a read-only binary stream whose handle permits atomic replacement.

    The caller owns closing the returned stream and any path/size authorization.
    An already-open stream continues to read its original file version after a
    writer replaces the directory entry; it cannot write through this handle.
    """
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        import msvcrt

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                           wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        create.restype = wintypes.HANDLE
        close = kernel.CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        close.restype = wintypes.BOOL
        # FILE_SHARE_DELETE is required for another process's atomic os.replace.
        handle = create(str(Path(path).absolute()), 0x80000000, 0x1 | 0x2 | 0x4,
                        None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except BaseException:
            close(handle)
            raise
        try:
            stream = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
    else:
        stream = Path(path).open("rb")
    return stream


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number: {value}")
    return number


def read_run_record(path, *, max_bytes=1048576, validate_opened=None):
    """Read one opened finite JSON object, allowing atomic replacement.

    OSError and ValueError propagate to callers. Optional validate_opened(stream)
    runs before bytes are read and may authorize this same opened handle.
    Catalog filename/id/schema authorization remains the consumer's job.
    """
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    with open_shared_read(path) as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("catalog record is not a regular file")
        if opened.st_size > max_bytes:
            raise ValueError("catalog record exceeds size limit")
        if validate_opened is not None:
            validate_opened(stream)
        content = stream.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise ValueError("catalog record exceeds size limit")
        value = json.loads(content.decode("utf-8-sig"), parse_constant=_reject_constant, parse_float=_parse_float)
    if not isinstance(value, dict):
        raise ValueError("catalog record must contain a JSON object")
    return value


def _read_object(path):
    try:
        value = read_run_record(path)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


@contextmanager
def _record_lock(path):
    # Separate roots never contend. A crashed writer leaves a harmless discovery
    # lock: do not delete another process's lock or delay native processing.
    deadline = time.monotonic() + 0.5
    while True:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"catalog record is busy: {path}") from None
            time.sleep(0.01)
    try:
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def _atomic_write(path, record):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.stem + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        deadline = time.monotonic() + 1.5
        while True:
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.005)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _source_launch(output):
    """Read only explicit launch metadata, never discover roots by scanning."""
    launch = _read_object(output / "production_launch.json")
    if not launch or not _same_root(launch.get("output_dir"), output):
        return None
    lock_path = output / ".pipeline.lock"
    lock = _read_object(lock_path)
    if lock_path.exists() and (not lock or lock.get("run_id") != launch.get("run_id")
                               or lock.get("pid") != launch.get("scheduler_pid")):
        return None
    kind = launch.get("kind")
    if kind is None:
        # Older explicit launches already record their real command. Do not
        # infer batch identity from directory names or a standalone audit.
        command = launch.get("command", [])
        scripts = {Path(item.replace("\\", "/")).name.casefold()
                   for item in command if isinstance(item, str)} if isinstance(command, list) else set()
        kind = next((value for script, value in (("step2_repair_humans.py", "human_repair"),
                    ("step3_repair_reviews.py", "review_repair"), ("step1_process_takes.py", "processing"))
                     if script in scripts), None)
    if kind not in KINDS - {"quality_recheck"}:
        return None
    if not isinstance(launch.get("run_id"), str) or not launch["run_id"]:
        return None
    pid = launch.get("scheduler_pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    try:
        started = _utc(launch["started_at"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"schema_version": 1, "id": run_key(output), "run_id": launch["run_id"],
            "kind": kind, "output_dir": str(output), "source_output": _absolute(launch.get("source_output")),
            "data_path": _data_path(output, launch.get("data_path")), "started_at": started,
            "scheduler_pid": pid, "label": f"{kind.replace('_', ' ')}: {output.name}",
            "planned_total": launch.get("planned_total")}


def register_run(output_dir, *, kind, run_id=None, source_output=None, data_path=None,
                 started_at=None, scheduler_pid=None, label=None, planned_total=None,
                 subject=None, input_path=None, registry_dir=None):
    """Atomically advertise a real launch; return its record, or warn and return None.

    Call only after preflight/dry-run exits and scheduler lock acquisition (where
    applicable). Explicit calls also allow operators to seed trusted historical
    launch metadata without importing a diagnostic or touching production data.
    Rechecks of an existing native root retain its identity and add an audit role.
    Sequence-local manifests supply omitted raw data_path, never output identity.
    """
    try:
        if kind not in KINDS:
            raise ValueError(f"unknown run kind: {kind}")
        if planned_total is not None and (not isinstance(planned_total, int)
                or isinstance(planned_total, bool) or planned_total < 0):
            raise ValueError("planned_total must be a nonnegative integer or None")
        pid = os.getpid() if scheduler_pid is None else scheduler_pid
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise ValueError("scheduler_pid must be a positive integer")
        output = Path(output_dir).expanduser().resolve()
        started = _utc(started_at)
        identity = run_key(output)
        record = {"schema_version": 1, "id": identity,
                  "run_id": run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + kind + "-" + uuid.uuid4().hex[:8],
                  "kind": kind, "output_dir": str(output), "source_output": _absolute(source_output),
                  "data_path": _data_path(output, data_path), "started_at": started, "scheduler_pid": pid,
                  "label": label or f"{kind.replace('_', ' ')}: {output.name}", "planned_total": planned_total}
        if subject is not None:
            record["subject"] = subject
        if input_path is not None:
            record["input_path"] = _absolute(input_path)
        registry = Path(registry_dir if registry_dir is not None else
                        os.environ.get("MOCAP_RUN_REGISTRY", DEFAULT_REGISTRY)).expanduser().resolve()
        registry.mkdir(parents=True, exist_ok=True)
        path = registry / (identity + ".json")
        with _record_lock(registry / (identity + ".lock")):
            if kind == "quality_recheck":
                previous = _read_object(path)
                if not (previous and previous.get("schema_version") == 1 and previous.get("id") == identity
                        and previous.get("kind") in KINDS - {"quality_recheck"}
                        and _same_root(previous.get("output_dir"), output)):
                    previous = _source_launch(output)
                if previous:
                    role = {key: record[key] for key in ("run_id", "started_at", "scheduler_pid")}
                    role["role"] = "independent_recheck"
                    record = dict(previous, quality_recheck=role)
            _atomic_write(path, record)
        return record
    except Exception as error:
        try:
            print(f"[WARNING] MocapWeb run catalog unavailable; processing continues: {type(error).__name__}: {error}",
                  file=sys.stderr, flush=True)
        except Exception:
            pass
        return None
