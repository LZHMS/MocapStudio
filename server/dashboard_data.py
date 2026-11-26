"""Read-only dashboard snapshots; no SDK imports and no production writes.

Stored reports, source coverage and verified process observations are deliberately
separate. A report is not a scheduler queue entry, and CPU activity is not a
quality guarantee. HTTP consumers share one background sampler.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import ntpath
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import threading
import time
from urllib.parse import quote

# The modules (check_health, step1_process_takes) are read-only references that
# live in the 4DHOI scripts directory; MOCAP_REPO_SCRIPTS overrides the location.
# This bootstrap must run before those imports because they resolve at import time.
DEFAULT_REPO_SCRIPTS = r"F:\projects\4DHOI\SceneDesign\4dhoi\scripts\shogun"


def ensure_repo_scripts_path():
    """Insert the Shogun pipeline scripts directory into sys.path at most once."""
    scripts = Path(os.environ.get("MOCAP_REPO_SCRIPTS") or DEFAULT_REPO_SCRIPTS)
    if not scripts.is_dir():
        return
    resolved = str(scripts.resolve())
    normalized = os.path.normcase(resolved)
    if not any(os.path.normcase(str(entry)) == normalized for entry in sys.path):
        sys.path.insert(0, resolved)


ensure_repo_scripts_path()

import check_health as health  # noqa: E402  (needs the sys.path bootstrap above)
import step1_process_takes  # noqa: E402  (needs the sys.path bootstrap above)
from pipeline_run_registry import open_shared_read
import pipeline_layout as layout
import mocap_plan


STATUSES = (*health.STATUSES, "RUNNING", "UNREPORTED", "UNKNOWN")
COVERAGE_LABEL = "输入清单结果覆盖（含失败），不是实时队列进度"
COMPONENT = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
TAIL_BYTES = 65536


def iso_epoch(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if health.numeric(value) else None


def contained(path, root):
    return path == root or root in path.parents


def scoped_path(root, value, *, relative_to=None, run_owned=False):
    """Resolve links before accepting a directory-local or exact run-owned path."""
    if not isinstance(value, (str, Path)) or not str(value) or "\x00" in str(value):
        raise ValueError("invalid path")
    text = str(value)
    _, remainder = ntpath.splitdrive(text)
    if ":" in remainder or text.replace("\\", "/").startswith(("//?/", "//./")):
        raise ValueError("device paths and alternate data streams are forbidden")
    if ".." in text.replace("\\", "/").split("/"):
        raise ValueError("parent traversal is forbidden")
    if os.name != "nt" and ntpath.isabs(text) and not Path(text).is_absolute():
        raise ValueError("foreign absolute path cannot be verified")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = (relative_to or root) / candidate
    if run_owned and not layout.in_run(root, candidate):
        raise ValueError("path escapes permitted run or traverses a link")
    candidate = candidate.resolve()
    if not (layout.in_run(root, candidate) if run_owned else contained(candidate, root)):
        raise ValueError("path escapes permitted directory")
    return candidate


def validate_opened_file(root, path, stream, *, nonempty=True, forbidden=None, check_current=True, run_owned=False):
    """Authorize this handle; catalog readers may retain an atomically replaced version."""
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or (nonempty and info.st_size <= 0):
        raise ValueError("not a nonempty regular file" if nonempty else "not a regular file")
    actual = path.resolve(strict=True)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        import msvcrt
        function = ctypes.WinDLL("kernel32", use_last_error=True).GetFinalPathNameByHandleW
        function.argtypes = (wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD)
        function.restype = wintypes.DWORD
        buffer = ctypes.create_unicode_buffer(32768)
        count = function(msvcrt.get_osfhandle(stream.fileno()), buffer, len(buffer), 0)
        if not count or count >= len(buffer):
            raise OSError(ctypes.get_last_error(), "Cannot verify opened file target")
        text = buffer.value
        if text.startswith("\\\\?\\UNC\\"):
            text = "\\\\" + text[8:]
        elif text.startswith("\\\\?\\"):
            text = text[4:]
        actual = Path(text).resolve(strict=True)
    authorized = layout.in_run(root, actual) if run_owned else contained(actual, root)
    if not authorized or (forbidden and forbidden(actual)):
        raise ValueError("opened file escapes permitted directory")
    if check_current or os.name != "nt":
        current = actual.stat()
        if (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("file changed during open")
    return info, actual


def open_scoped(root, value, *, relative_to=None, nonempty=True, forbidden=None, run_owned=False):
    """Open without blocking atomic publication, retaining all scoped handle guards."""
    path = scoped_path(root, value, relative_to=relative_to, run_owned=run_owned)
    if forbidden and forbidden(path):
        raise ValueError("source inputs are not downloadable")
    stream = open_shared_read(path)
    try:
        info, actual = validate_opened_file(root, path, stream, nonempty=nonempty,
                                          forbidden=forbidden, run_owned=run_owned)
        return stream, info, actual
    except BaseException:
        stream.close()
        raise


def read_json_file(path, root, *, run_owned=False):
    """Health-compatible evidence read that cannot lock out atomic publication."""
    try:
        before = path.stat()
        stream, _, _ = open_scoped(root, path, run_owned=run_owned)
        with stream:
            value = json.load(stream)
        after = path.stat()
        identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if identity(before) != identity(after):
            return None, None, "file changed while being read"
        if not isinstance(value, dict):
            return None, None, "expected a JSON object"
        return value, after.st_mtime, None
    except (OSError, ValueError, RuntimeError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def optional_json_file(path, root, *, run_owned=False):
    try:
        path.stat()
    except FileNotFoundError:
        return {"path": str(path), "present": False, "data": None, "error": None}
    except OSError as exc:
        return {"path": str(path), "present": None, "data": None, "error": str(exc)}
    data, modified, error = read_json_file(path, root, run_owned=run_owned)
    return {"path": str(path), "present": True, "data": data, "error": error,
            "modified_at": iso_epoch(modified), "modified_epoch": modified}


def shared_sessions(output):
    """Same health session evidence schema, with publication-safe scoped reads."""
    result = {"entries": [], "errors": []}
    try:
        children = list(layout.bucket_entries(output, ".work"))
    except FileNotFoundError:
        return result
    except (OSError, ValueError, RuntimeError) as exc:
        result["errors"].append(str(exc))
        return result
    for child in children:
        try:
            child = scoped_path(output, child, run_owned=True)
            if not child.is_dir():
                continue
            record = optional_json_file(child / ".control" / "session-ready.json", output, run_owned=True)
            if record["present"] is False:
                continue
            data = record["data"]
            if not record["error"] and (not health.positive_int(data.get("pid"))
                    or not health.positive_int(data.get("port")) or data["port"] > 65535
                    or health.parse_time(data.get("started_at")) is None):
                record["error"] = "invalid native PID, port or timezone-aware started_at"
            record.update(workspace=str(child), job_id=child.name.split("-", 1)[0])
            result["entries"].append(record)
            if record["error"]:
                result["errors"].append(f"{record['path']}: {record['error']}")
        except (OSError, ValueError, RuntimeError) as exc:
            result["errors"].append(str(exc))
    return result


def tail_log(output, path):
    result = {"path": str(path) if path else None, "text": "", "truncated": False, "error": None}
    if path is None:
        return result
    try:
        with open_scoped(output, path, nonempty=False, run_owned=True)[0] as stream:
            size = os.fstat(stream.fileno()).st_size
            start = max(0, size - TAIL_BYTES)
            stream.seek(start)
            chunk = stream.read(TAIL_BYTES)
        result.update(text=chunk.decode("utf-8-sig", errors="replace"), truncated=bool(start))
    except (OSError, ValueError, RuntimeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def safe_relative(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    text = value.replace("\\", "/")
    parts = text.split("/")
    if text.startswith("/") or ":" in text or any(part in ("", ".", "..") for part in parts):
        return None
    return PurePosixPath(text).as_posix()


def safe_component(value):
    return isinstance(value, str) and COMPONENT.fullmatch(value) is not None


def dashboard_report_error(report):
    error = health.validate_report(report)
    if error:
        return error
    if not safe_component(report["job_id"]):
        return "job_id is not a safe API identifier"
    for field in ("dataset", "policy", "session", "range", "quality"):
        if field in report and not isinstance(report[field], dict):
            return f"invalid {field}"
    for field in ("input_path", "relative_path", "sequence_kind", "error"):
        if field in report and not isinstance(report[field], str):
            return f"invalid {field}"
    frames = report.get("range", {}).get("frame_count")
    if frames is not None and (not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0):
        return "invalid range.frame_count"
    stages = report.get("stages", [])
    if not isinstance(stages, list) or any(not isinstance(stage, dict) for stage in stages):
        return "invalid stages"
    reasons = report.get("background_review_reasons", [])
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        return "invalid background_review_reasons"
    if "limits" in report and report["limits"] is not None and not isinstance(report["limits"], (dict, list)):
        return "invalid limits"

    def finite(value):
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite(child) for child in value.values())
        if isinstance(value, list):
            return all(finite(child) for child in value)
        return True

    return None if finite(report) else "non-finite JSON number"


def final_usable(report):
    quality = report.get("quality")
    if isinstance(quality, dict) and quality.get("basis") == "post_processing" and isinstance(quality.get("usable"), bool):
        return quality["usable"]
    return None


def activity_phase(native, worker, sample_seconds):
    if sample_seconds is None or sample_seconds <= 0:
        return "UNKNOWN"
    if native["observed_activity"] and worker["observed_activity"]:
        return "BOTH_COMPUTING"
    if worker["observed_activity"]:
        return "PYTHON_COMPUTING"
    if native["observed_activity"]:
        return "POST_COMPUTING"
    # A missing side/counter is not a measured zero.
    keys = ("cpu_seconds_delta", "read_bytes_delta", "write_bytes_delta")
    if all(health.numeric(side[key]) for side in (native, worker) for key in keys):
        return "NO_OBSERVED_ACTIVITY"
    return "UNKNOWN"


class DashboardData:
    def __init__(self, output_dir, data_path=None, refresh_seconds=15, *,
                 snapshot_provider=None, sleeper=time.sleep, sample_seconds=3, clock=time.monotonic, plan_dir=None, run=None):
        self.output = Path(output_dir).expanduser().resolve()
        spec = layout.load_layout(self.output)
        raw = data_path if data_path is not None else spec["data_root"] if spec else None
        self.data_path = Path(raw).expanduser().resolve() if raw is not None else None
        self.plan_root = (Path(plan_dir).expanduser().resolve() if plan_dir else
                          Path(__file__).resolve().parents[1] / "data" / "p2_hoi")
        self.plan = mocap_plan.load(self.plan_root)
        self.refresh_seconds = refresh_seconds
        self.snapshot_provider = snapshot_provider or health.process_snapshot
        self.sleeper = sleeper
        self.sample_seconds = sample_seconds
        self.clock = clock
        self._source_expires = -math.inf
        self._source_jobs = None
        self._source_error = None
        self._report_cache = {}
        self.run = dict(run or {})
        self._coverage_label = COVERAGE_LABEL
        self.inventory = {}

    def read_local_json(self, value, *, run_owned=True):
        stream, info, _ = open_scoped(self.output, value, run_owned=run_owned)
        with stream:
            if info.st_size > 32 * 1024 * 1024:
                raise ValueError("JSON evidence exceeds 32 MiB")
            return json.load(stream)

    def repair_inventory(self):
        """The selected plan, never its all-source inventory, defines coverage."""
        path = self.output / "repair_plan.json"
        required = self.run.get("kind") in ("human_repair", "review_repair")
        if not path.exists() and not required:
            return None
        self._coverage_label = "选中修复计划结果覆盖（含失败），不是实时队列进度"
        try:
            # Plans are batch control evidence, never searched in sequence storage.
            plan = self.read_local_json(path, run_owned=False)
            if not isinstance(plan, dict) or not isinstance(plan.get("jobs"), list):
                raise ValueError("repair_plan.jobs must be a list")
            jobs = {}
            for item in plan["jobs"]:
                if not isinstance(item, dict) or not safe_component(item.get("job_id")):
                    raise ValueError("invalid selected job")
                relative = safe_relative(item.get("relative_path"))
                if relative is None or item["job_id"] in jobs:
                    raise ValueError("invalid selected relative path or duplicate job_id")
                report_path = scoped_path(self.output, layout.take_paths(self.output, relative)["report_path"],
                                          run_owned=True)
                if any(job["report_path"] == str(report_path) for job in jobs.values()):
                    raise ValueError("duplicate selected report path")
                jobs[item["job_id"]] = dict(item, relative_path=relative, report_path=str(report_path))
            counts = plan.get("counts", {})
            self.inventory = {key: value for key, value in counts.items()
                              if isinstance(key, str) and type(value) is int and value >= 0} if isinstance(counts, dict) else {}
            roster = plan.get("roster_pending")
            inventory = plan.get("inventory")
            if isinstance(inventory, list) and all(isinstance(item, dict) for item in inventory):
                identities = {item["job_id"] for item in inventory if safe_component(item.get("job_id"))}
                reusable = {item["job_id"] for item in inventory if safe_component(item.get("job_id"))
                            and item.get("disposition") in ("selected", "deferred", "duplicate")}
                self.inventory["inventoried_identities"] = len(identities)
                self.inventory["identities_without_reusable_checkpoint"] = len(identities - reusable)
            if isinstance(roster, list):
                self.inventory["roster_pending"] = len(roster)
            return jobs, None
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            self.inventory = {}
            return None, f"选中修复计划未知；不使用源清单或登记总量代替：{exc}"

    def recheck_attempts(self):
        latest, warnings = {}, []
        try:
            for entry in layout.bucket_entries(self.output, ".quality_recheck_attempts"):
                path = entry / "attempt.json"
                try:
                    path = scoped_path(self.output, path, run_owned=True)
                    if not path.is_file():
                        continue
                    attempt = self.read_local_json(path)
                    if not isinstance(attempt, dict):
                        raise ValueError("invalid attempt")
                    report = scoped_path(self.output, attempt.get("report"), run_owned=True)
                    relative = layout.report_relative(self.output, report)
                    if layout.load_layout(self.output) is not None:
                        expected = layout.take_paths(self.output, relative)["recheck_root"]
                        if path.parent.parent not in (expected, self.output / ".quality_recheck_attempts"):
                            raise ValueError("attempt belongs to a different take")
                    if health.normalize_path(attempt.get("output")) != health.normalize_path(self.output):
                        raise ValueError("attempt output root mismatch")
                    started = health.parse_time(attempt.get("started_at"))
                    if started is None:
                        raise ValueError("invalid attempt start")
                    previous = latest.get(str(report))
                    if previous is None or started > health.parse_time(previous["started_at"]):
                        latest[str(report)] = attempt
                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                    warnings.append(f"Invalid recheck attempt {path}: {exc}")
        except (OSError, ValueError, RuntimeError) as exc:
            warnings.append(f"Recheck attempts unavailable: {exc}")
        return latest, warnings

    def independent_recheck(self, key, record, attempts):
        report = record.get("report") or {}
        embedded = report.get("independent_recheck")
        state = dict(embedded) if isinstance(embedded, dict) else {"status": "PENDING"}
        path = record.get("path")
        if path:
            try:
                sidecar = self.read_local_json(Path(path).with_name("independent_recheck.json"))
                if not isinstance(sidecar, dict) or sidecar.get("job_id") != key:
                    raise ValueError("independent recheck job identity mismatch")
                if (sidecar.get("status") not in ("RUNNING", "PASSED", "FAILED")
                        or health.parse_time(sidecar.get("finished_at") or sidecar.get("started_at")) is None):
                    raise ValueError("invalid independent recheck state/time")
                if sidecar.get("status") == "PASSED":
                    if self.run.get("kind") == "human_repair" and not sidecar.get("artifact_sha256"):
                        sidecar = dict(sidecar, status="UNKNOWN", error="Legacy PASSED lacks artifact fingerprint evidence")
                    stream, _, _ = open_scoped(self.output, path, run_owned=True)
                    with stream:
                        signature = hashlib.sha256(stream.read()).hexdigest()
                    if sidecar.get("report_sha256") != signature:
                        sidecar = dict(sidecar, status="UNKNOWN", error="Recheck evidence does not match current report")
                times = lambda item: health.parse_time(item.get("finished_at") or item.get("started_at"))
                if times(state) is None or (times(sidecar) is not None and times(sidecar) >= times(state)):
                    state = dict(sidecar)
            except FileNotFoundError:
                if self.run.get("kind") == "human_repair" and state.get("status") == "PASSED":
                    state = {"status": "UNKNOWN", "error": "Missing durable independent recheck sidecar"}
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                state = {"status": "UNKNOWN", "error": str(exc)}
        attempt = attempts.get(path)
        if attempt:
            current = health.parse_time(state.get("finished_at") or state.get("started_at"))
            started = health.parse_time(attempt.get("started_at"))
            if current is None or started >= current:
                phase = attempt.get("phase")
                status = ("FAILED" if phase in ("failed", "timed_out") else
                          "PASSED" if phase == "completed" and attempt.get("finished_at") else "RUNNING")
                if status == "PASSED" and self.run.get("kind") == "human_repair":
                    status = "UNKNOWN"
                state = {"status": status, "phase": phase, "started_at": attempt.get("started_at"),
                         "finished_at": attempt.get("finished_at"), "error": attempt.get("error"),
                         "worker_pid": attempt.get("worker_pid"), "native_pid": attempt.get("native_pid")}
        if state.get("status") not in ("PENDING", "RUNNING", "PASSED", "FAILED", "UNKNOWN"):
            state["status"] = "UNKNOWN"
        state["operational_checked"] = state["status"] == "PASSED"
        state["meaning"] = "PASSED 仅表示独立检查完成，不表示数据可用。"
        return state

    def source_inventory(self):
        selected = self.repair_inventory()
        if selected is not None:
            return selected
        if self.data_path is None:
            return None, "未提供 DataPath；输入总数、未有报告数量和覆盖比例未知。"
        now = self.clock()
        if now < self._source_expires:
            return self._source_jobs, self._source_error
        self._source_expires = now + 60
        try:
            # rglob can omit inaccessible directories on some Python versions.
            # Refuse to call a partial enumeration an exact source inventory.
            errors = []
            if self.data_path.is_dir():
                for directory, directories, _ in os.walk(self.data_path, onerror=lambda exc: errors.append(str(exc))):
                    directories[:] = [name for name in directories
                                      if not layout.is_generated(Path(directory) / name, self.data_path)]
                    for name in directories:
                        child = Path(directory) / name
                        if child.is_symlink() or child.resolve() != child.absolute():
                            errors.append(f"Cannot certify recursive source inventory through link: {child}")
            if errors:
                raise OSError("; ".join(errors))
            local = layout.load_layout(self.output) is not None
            jobs = step1_process_takes.build_jobs(self.data_path, self.output, self.output / "logs", "mcp",
                                                 step1_process_takes.load_manifest(None), None, None, True,
                                                 dataset_layout=local)
            jobs = [job for job in jobs if not layout.is_generated(job["input_path"], self.data_path)]
            self._source_jobs = {job["job_id"]: job for job in jobs}
            if len(self._source_jobs) != len(jobs):
                raise ValueError("source job IDs are not unique")
            self._source_error = None
        except (OSError, ValueError, RuntimeError) as exc:
            self._source_jobs = None
            self._source_error = f"输入清单未知：{type(exc).__name__}: {exc}"
        return self._source_jobs, self._source_error

    def reports(self, sources):
        """Read only helper-certified canonical reports for this exact run."""
        records, warnings, seen = {}, [], set()
        self._report_scan_complete = True
        expected = {str(Path(job["report_path"]).resolve()): key for key, job in (sources or {}).items()}
        try:
            for candidate in layout.report_paths(self.output):
                relative = layout.report_relative(self.output, candidate).as_posix()
                path = candidate
                data, modified, error = None, None, None
                try:
                    path = scoped_path(self.output, candidate, run_owned=True)
                    info = path.stat()
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError("report is not a regular file")
                    signature = (info.st_mtime_ns, info.st_size, info.st_dev, info.st_ino)
                    seen.add(str(path))
                    cached = self._report_cache.get(str(path))
                    if cached and cached[0] == signature:
                        data, modified, error = cached[1:]
                    else:
                        data, modified, error = read_json_file(path, self.output, run_owned=True)
                        error = error or dashboard_report_error(data)
                        self._report_cache[str(path)] = (signature, data, modified, error)
                except (OSError, ValueError, RuntimeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                key = expected.get(str(candidate.resolve()))
                if key is None and isinstance(data, dict) and safe_component(data.get("job_id")):
                    key = data["job_id"]
                if key is None:
                    key = "unknown-" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
                if key in (sources or {}) and isinstance(data, dict) and data.get("job_id") != key:
                    error = "report job_id disagrees with the source inventory"
                record = {"report": data if not error else None, "path": str(path), "modified": modified,
                          "error": error, "relative_path": relative}
                if error:
                    warnings.append(f"{candidate}: {error}")
                if key in records:
                    warnings.append(f"Duplicate canonical job_id {key}; newest report selected; identity ambiguous")
                    if (modified or 0) <= (records[key]["modified"] or 0):
                        continue
                records[key] = record
        except (OSError, ValueError, RuntimeError) as exc:
            self._report_scan_complete = False
            warnings.append(f"报告清单不完整：{type(exc).__name__}: {exc}")
        self._report_cache = {key: value for key, value in self._report_cache.items() if key in seen}
        return records, warnings

    def artifact_rows(self, key, record):
        report = record.get("report") or {}
        rows, states = [], []
        for role, value in report.get("artifacts", {}).items():
            available, size, resolved = False, None, None
            try:
                if not safe_component(role):
                    raise ValueError("invalid artifact role")
                resolved = scoped_path(self.output, value, relative_to=Path(record["path"]).parent, run_owned=True)
                if self.artifact_forbidden(resolved, report, Path(record["path"])):
                    raise ValueError("artifact outside permitted take products")
                info = resolved.stat()
                available = stat.S_ISREG(info.st_mode) and info.st_size > 0
                if available:
                    size = info.st_size
                states.append("complete" if available else "missing")
            except FileNotFoundError:
                states.append("missing")
            except (OSError, ValueError, RuntimeError, TypeError):
                states.append("unknown")
            rows.append({"role": role, "name": resolved.name if resolved else str(role),
                         "path": str(resolved) if resolved else str(value), "size_bytes": size,
                         "available": available,
                         "url": (f"/api/artifacts/{quote(key, safe='')}/{quote(role, safe='')}" +
                                 ("?run=" + self.run["id"] if self.run.get("id") else "")) if available else None})
        if self.run.get("kind") == "human_target" and report:
            required = {"baseline"} | ({"candidate", "fbx"} if report.get("status") == "SUCCESS" else set())
            integrity = ("missing" if required - report.get("artifacts", {}).keys() or "missing" in states else
                         "unknown" if "unknown" in states else "complete")
            return rows, integrity
        if report.get("status") == "FAILED":
            integrity = "not_applicable"
        elif not report:
            integrity = "unknown"
        else:
            # Existing checker is the canonical definition of required products.
            issues = health.check_artifacts(report, Path(record["path"]), self.output)
            if "missing" in states or any(issue["state"] == "missing" for issue in issues):
                integrity = "missing"
            elif "unknown" in states or issues:
                integrity = "unknown"
            else:
                integrity = "complete"
        return rows, integrity

    def source_file(self, path, report=None):
        if self.data_path is not None and contained(path, self.data_path):
            # The raw tree contains sequence-local outputs; only exact run-owned
            # generated paths may bypass the blanket raw-input prohibition.
            if not layout.is_generated(path, self.data_path) or not layout.in_run(self.output, path):
                return True
        value = (report or {}).get("input_path")
        if isinstance(value, str) and value:
            try:
                if path == Path(value).expanduser().resolve():
                    return True
            except (OSError, ValueError, RuntimeError):
                return True
        if path.suffix.casefold() in (".mcp", ".x2d"):
            return True
        for value in (self.run.get("input_path"), self.run.get("source_output")):
            if isinstance(value, str) and value:
                source = Path(value).resolve()
                if path == source or (source.is_dir() and not contained(self.output, source) and contained(path, source)):
                    return True
        return False

    def artifact_forbidden(self, path, report, report_path):
        if self.source_file(path, report):
            return True
        if self.run.get("kind") == "human_target" or layout.load_layout(self.output) is None:
            return False
        relative = layout.report_relative(self.output, report_path)
        return not any(contained(path, root.resolve())
                       for root in layout.artifact_roots(self.output, relative))

    def process_evidence(self):
        attempts, _ = self.recheck_attempts()
        lock_before = optional_json_file(self.output / ".pipeline.lock", self.output)
        sessions_before = shared_sessions(self.output)
        pids = [entry["data"].get("pid") for entry in sessions_before["entries"] if entry.get("data")]
        for attempt in attempts.values():
            pids.extend(attempt.get(field) for field in ("parent_pid", "worker_pid", "native_pid"))
        if lock_before.get("data"):
            pids.append(lock_before["data"].get("pid"))
        first = self.snapshot_provider([pid for pid in pids if health.positive_int(pid)])
        self.sleeper(self.sample_seconds)
        sessions_after = shared_sessions(self.output)
        pids.extend(entry["data"].get("pid") for entry in sessions_after["entries"] if entry.get("data"))
        attempts_after, attempt_errors = self.recheck_attempts()
        attempts.update(attempts_after)
        for attempt in attempts.values():
            pids.extend(attempt.get(field) for field in ("parent_pid", "worker_pid", "native_pid"))
        second = self.snapshot_provider([pid for pid in pids if health.positive_int(pid)])
        lock_after = optional_json_file(self.output / ".pipeline.lock", self.output)
        before = health.identify_lock(lock_before, first, self.output)
        after = health.identify_lock(lock_after, second, self.output)
        stable = (lock_before["present"] == lock_after["present"] and not lock_before["error"]
                  and not lock_after["error"] and lock_before["data"] == lock_after["data"])
        old_sessions = health.identify_sessions(sessions_before, first, before)
        observations = health.identify_sessions(sessions_after, second, after)
        old = {(row.get("workspace"), row.get("native_pid")): row for row in old_sessions}
        maps = (health.process_map(first), health.process_map(second))
        start, end = health.parse_time(first.get("sampled_at")), health.parse_time(second.get("sampled_at"))
        sample = max(0, (end - start).total_seconds()) if start and end else None
        stable_owner = (stable and before["identity"] == after["identity"] == "OWNED"
                        and health.same_process(before.get("process"), after.get("process")))
        active, warnings = {}, sessions_before["errors"] + sessions_after["errors"]
        for snapshot in (first, second):
            if snapshot.get("status") != "ok":
                warnings.append(snapshot.get("error") or "进程查询不可用；活动数量未知。")
            if snapshot.get("memory_error"):
                warnings.append("内存查询不可用：" + snapshot["memory_error"])
        for row in observations:
            if row["ownership"] != "OWNED" or not safe_component(row.get("job_id")):
                continue
            prior = old.get((row.get("workspace"), row.get("native_pid")), {})
            continuous = stable_owner and prior.get("ownership") == "OWNED"
            native = health.activity(prior.get("process") if continuous else None, row.get("process"))
            worker_before = maps[0].get(prior.get("worker_pid")) if continuous else None
            worker_after = maps[1].get(row.get("worker_pid"))
            worker = health.activity(worker_before, worker_after)
            phase = activity_phase(native, worker, sample)
            active[row["job_id"]] = {"native_pid": row.get("native_pid"), "worker_pid": row.get("worker_pid"),
                "age_seconds": row.get("age_seconds"), "sample_seconds": sample,
                "native_cpu_delta_seconds": native["cpu_seconds_delta"],
                "worker_cpu_delta_seconds": worker["cpu_seconds_delta"],
                "native_read_bytes_delta": native["read_bytes_delta"], "native_write_bytes_delta": native["write_bytes_delta"],
                "worker_read_bytes_delta": worker["read_bytes_delta"], "worker_write_bytes_delta": worker["write_bytes_delta"],
                "native_memory_bytes": row.get("process", {}).get("working_set_bytes"),
                "worker_memory_bytes": (worker_after or {}).get("working_set_bytes"), "phase": phase,
                "started_at": row.get("started_at")}
        # Independent recheck sessions live outside .work. Require the actual
        # parent/worker/native process chain, not the audit's RUNNING label.
        recheck_uncertain = bool(attempt_errors)
        warnings.extend(attempt_errors)
        for report_path, attempt in attempts.items():
            if attempt.get("finished_at"):
                continue
            worker_after = maps[1].get(attempt.get("worker_pid"))
            native_after = maps[1].get(attempt.get("native_pid"))
            parent = maps[1].get(attempt.get("parent_pid"))
            tokens = health.command_tokens((parent or {}).get("command_line"))
            target = health.option_value(tokens, "--output-dir") or health.option_value(tokens, "-OutputDir")
            started = health.parse_time(attempt.get("started_at"))
            worker_created = health.parse_time((worker_after or {}).get("created_at"))
            parent_created = health.parse_time((parent or {}).get("created_at"))
            if not (parent and worker_after and started and parent_created and worker_created
                    and parent_created <= started and (worker_created - started).total_seconds() >= -2
                    and worker_after.get("name", "").casefold() in health.PYTHON_NAMES
                    and worker_after.get("ppid") == parent.get("pid")
                    and target and ntpath.isabs(target) and health.normalize_path(target) == health.normalize_path(self.output)
                    and any(ntpath.basename(token).casefold() in ("step2_repair_humans.py", "step4_recheck_quality.py") for token in tokens)):
                recheck_uncertain = True
                continue
            if native_after and (native_after.get("ppid") != worker_after["pid"]
                                 or native_after.get("name", "").casefold() not in health.NATIVE_NAMES):
                native_after = None
            prior_worker = maps[0].get(worker_after["pid"])
            prior_native = maps[0].get((native_after or {}).get("pid"))
            worker_activity = health.activity(prior_worker, worker_after)
            native_activity = health.activity(prior_native, native_after)
            key = next((key for key, record in getattr(self, "_current_reports", {}).items() if record.get("path") == report_path), None)
            if key is None:
                continue
            active[key] = {"worker_pid": worker_after["pid"], "native_pid": (native_after or {}).get("pid"),
                           "phase": activity_phase(native_activity, worker_activity, sample),
                           "started_at": attempt.get("started_at"), "sample_seconds": sample,
                           "operation": "independent_recheck",
                           "native_cpu_delta_seconds": native_activity["cpu_seconds_delta"],
                           "worker_cpu_delta_seconds": worker_activity["cpu_seconds_delta"]}
        if not stable:
            warnings.append("采样间锁发生变化；属于状态切换，不证明失败。")
        uncertain_sessions = bool(recheck_uncertain or sessions_before["errors"] or sessions_after["errors"]
                                  or any(row["ownership"] == "UNKNOWN" for row in observations))
        known = first.get("status") == second.get("status") == "ok" and stable
        # Visible owned processes are a lower bound if any other identity is unknown.
        count_known = known and not uncertain_sessions and after["identity"] in ("OWNED", "ABSENT")
        if uncertain_sessions:
            warnings.append("部分会话归属未知；已显示的活动实例仅为确认下界，精确活动总数未知。")
        memory = second.get("memory") or {}
        phases = [row["phase"] for row in active.values()]
        progress = sum(phase in ("POST_COMPUTING", "PYTHON_COMPUTING", "BOTH_COMPUTING") for phase in phases)
        measurable = count_known and all(phase != "UNKNOWN" for phase in phases)
        result = {"overall": "UNKNOWN", "reason": "证据不足；短采样无活动不证明卡死。",
            "run_id": after.get("run_id"), "scheduler_pid": after.get("pid"), "verified_owner": after["identity"] == "OWNED",
            "active_count": len(active) if count_known else None, "active_progress_count": progress if measurable else None,
            "python_busy_count": sum(phase in ("PYTHON_COMPUTING", "BOTH_COMPUTING") for phase in phases) if measurable else None,
            "native_busy_count": sum(phase in ("POST_COMPUTING", "BOTH_COMPUTING") for phase in phases) if measurable else None,
            "memory_free_bytes": memory.get("free_bytes"), "memory_total_bytes": memory.get("total_bytes"),
            "observed_at": second.get("sampled_at")}
        if known and after["identity"] in ("DEAD", "MISMATCH"):
            result.update(overall="CONFIRMED_ERROR", reason=after["reason"])
        elif stable_owner and progress:
            result.update(overall="RUNNING", reason="已核实任务的 Python/Post 在采样间有 CPU 或 I/O 活动；具体阶段未知，活动不保证最终正确。")
        return result, active, warnings, (first, second, before, after, observations, known)

    def collect(self):
        if not self.output.is_dir():
            raise ValueError(f"Output directory does not exist or is inaccessible: {self.output}")
        if self.run.get("kind") == "human_target":
            return self.collect_target()
        sources, source_error = self.source_inventory()
        reports, warnings = self.reports(sources)
        if self._coverage_label != COVERAGE_LABEL and sources is not None:
            excluded = set(reports) - set(sources)
            if excluded:
                warnings.append(f"{len(excluded)} canonical reports are outside the selected repair plan and excluded from planned coverage.")
            reports = {key: record for key, record in reports.items() if key in sources}
        self._current_reports = reports
        report_warnings = list(warnings)
        runtime, active, process_warnings, evidence = self.process_evidence()
        if self._coverage_label != COVERAGE_LABEL and sources is not None:
            active = {key: value for key, value in active.items() if key in sources}
        attempts, attempt_warnings = self.recheck_attempts()
        warnings.extend(attempt_warnings)
        warnings.extend(process_warnings)
        if source_error:
            warnings.append(source_error)
        rows, details = {}, {}
        for key in sorted(set(sources or {}) | set(reports) | set(active)):
            job = (sources or {}).get(key, {})
            record = reports.get(key, {})
            report_unknown = not self._report_scan_complete and not record
            report = record.get("report") or {}
            relative = safe_relative(job.get("relative_path") or record.get("relative_path") or report.get("relative_path"))
            group_number, category = step1_process_takes.dataset_sequence(Path(relative)) if relative else (None, None)
            kind = job.get("sequence_kind") or report.get("sequence_kind") or (report.get("dataset") or {}).get("sequence_kind") or category or "UNKNOWN"
            activity = active.get(key)
            recheck = self.independent_recheck(key, record, attempts)
            artifacts, integrity = self.artifact_rows(key, record)
            reported = report.get("status")
            status = "RUNNING" if activity or recheck["status"] == "RUNNING" else reported or ("UNKNOWN" if record or report_unknown else "UNREPORTED")
            session = report.get("session") if isinstance(report.get("session"), dict) else {}
            started, finished = health.parse_time(session.get("started_at")), health.parse_time(session.get("finished_at"))
            duration = max(0, (finished - started).total_seconds()) if started and finished else None
            reasons = report.get("review_reasons", [])
            summary = ("具体阶段未知；CPU活动不保证最终正确。旧报告状态仅作历史解释。" if activity else
                       record.get("error") or ("报告目录枚举不完整；是否已有报告未知。" if report_unknown else
                       "未有报告；不能据此推断实时队列或待处理状态。" if not report else None))
            if not activity and integrity in ("missing", "unknown") and reported in ("SUCCESS", "NEEDS_REVIEW"):
                summary = "报告记录已完成，但产物缺失或无法核实；不能作为完整发布结果。"
            row = {"id": key, "name": PurePosixPath(relative).name if relative else key,
                "relative_path": relative or "", "input_path": job.get("input_path") or report.get("input_path"),
                "group": f"group{group_number}" if group_number is not None else "UNKNOWN", "kind": str(kind),
                "status": status, "reported_status": reported,
                "usable": False if recheck["status"] in ("FAILED", "RUNNING", "UNKNOWN")
                                  or (self.run.get("kind") == "human_repair" and recheck["status"] != "PASSED")
                                  else final_usable(report),
                "quality_scope": report.get("quality", {}).get("scope"),
                "independent_recheck": recheck, "artifact_integrity": integrity,
                "subjects": [item["name"] for item in report.get("subjects", []) if item.get("processing_target") is True and item.get("name")],
                "duration_seconds": duration, "frames": (report.get("range") or {}).get("frame_count"),
                "updated_at": iso_epoch(record.get("modified")),
                "started_at": activity.get("started_at") if activity else started.isoformat() if started else None,
                "review_count": len(reasons), "error": report.get("error") or record.get("error"),
                "summary_reason": summary, "active": {field: value for field, value in activity.items() if field != "started_at"} if activity else None}
            rows[key] = row
            plan_code = re.sub(r"\.[A-Za-z0-9]+\Z", "", re.sub(r"\AG\d+-", "", row["name"]))
            row.update(self.plan.lookup(row["group"], plan_code) if self.plan else
                       {"plan_title": None, "plan_actors": None, "plan_props": None, "plan_notes": None})
            details[key] = {"take": row, "quality": report.get("quality"), "dataset": report.get("dataset"), "policy": report.get("policy"),
                "session": session,
                "review_reasons": reasons, "background_review_reasons": report.get("background_review_reasons", []),
                "error": row["error"], "stages": report.get("stages", []) if isinstance(report.get("stages", []), list) else [],
                "subjects": [subject for subject in report.get("subjects", [])
                             if subject.get("processing_target") is True or subject.get("rig_class") == "HUMAN_RIG"],
                "props": report.get("props", []), "artifacts": artifacts,
                "report_path": record.get("path"), "limits": report.get("limits"),
                "processing_source": report.get("processing_source"),
                "reference_repair_preservation": report.get("reference_repair_preservation"),
                "camera_repair_audit": report.get("camera_repair_audit"),
                "independent_recheck": recheck}
            if (recheck["status"] in ("FAILED", "RUNNING", "UNKNOWN")
                    or (self.run.get("kind") == "human_repair" and recheck["status"] != "PASSED")) and isinstance(report.get("quality"), dict):
                details[key]["quality"] = dict(report["quality"], usable=False)
        first, second, before, after, observations, known = evidence
        valid_jobs = {key: entry for key, entry in reports.items() if entry.get("report")}
        batch_report = optional_json_file(self.output / "batch_report.json", self.output)
        summary = health.final_summary(batch_report, valid_jobs)
        batch_data = batch_report.get("data") if isinstance(batch_report, dict) else None
        if not runtime.get("run_id") and isinstance(batch_data, dict) and batch_data.get("run_id"):
            runtime["run_id"] = str(batch_data["run_id"])
        candidates = []
        for snapshot in (first, second):
            for process in snapshot.get("processes", []):
                tokens = health.command_tokens(process.get("command_line"))
                target = health.option_value(tokens, "-OutputDir") or health.option_value(tokens, "--output-dir")
                if any(ntpath.basename(token).casefold() in (*health.SCHEDULER_SCRIPTS, "step4_recheck_quality.py") for token in tokens) and (target is None or health.normalize_path(target) == health.normalize_path(self.output)):
                    candidates.append(process["pid"])
        recheck_summary = optional_json_file(self.output / "quality_recheck_report.json", self.output)
        recheck_unfinished = (recheck_summary.get("data") or {}).get("finished") is False
        completion_clear = (not report_warnings and not process_warnings and not attempt_warnings and not recheck_unfinished
                            and not any(row["independent_recheck"]["status"] in ("RUNNING", "FAILED", "UNKNOWN") for row in rows.values())
                            and (sources is None or all(key in valid_jobs for key in sources))
                            and all(row["artifact_integrity"] == "complete" for row in rows.values()
                                    if row["status"] in ("SUCCESS", "NEEDS_REVIEW")))
        if (runtime["overall"] == "UNKNOWN" and known and before["identity"] == after["identity"] == "ABSENT"
                and summary["complete"] and completion_clear and not candidates
                and not any(process.get("name", "").casefold() in health.NATIVE_NAMES for snapshot in (first, second) for process in snapshot.get("processes", []))
                and not any(row["ownership"] == "UNKNOWN" for row in observations)):
            runtime.update(overall="COMPLETED", reason="最终批报告与当前记录一致；两次采样均无锁、调度器和原生进程。完成不等于全部质量通过。")
        counts = {status: sum(row["status"] == status for row in rows.values()) for status in STATUSES}
        if sources is None:
            counts["UNREPORTED"] = None
        if not self._report_scan_complete:
            counts = dict.fromkeys(STATUSES)
            warnings.append("报告目录枚举不完整；状态数量未知，可见记录及已验证发布数仅为下界。")
        terminal = sum(key in valid_jobs and rows[key]["status"] != "RUNNING" for key in sources) if sources is not None and self._report_scan_complete else None
        published = sum(row["status"] in ("SUCCESS", "NEEDS_REVIEW") and row["artifact_integrity"] == "complete" for row in rows.values())
        total = len(sources) if sources is not None else None
        unreported = sum(key not in reports and key not in active for key in sources) if sources is not None and self._report_scan_complete else None
        if any(row["reported_status"] == "FAILED" for row in rows.values()):
            warnings.append("FAILED 是当前磁盘历史记录，不等于本轮新失败；运行中的同一任务以 RUNNING 优先显示。")
        if not self.plan:
            warnings.append("p2_hoi 台本数据未能加载；序列的物体、演员与拍摄备注不可用。")
        limits = ["具体阶段未知；CPU活动不保证最终正确。", "没有实时调度队列快照；不推算精确待处理数、吞吐或 ETA。",
                  "报告及进程观察非原子快照，状态切换和权限不足保留未知。", "产物仅验证非空普通文件，不加载 Shogun、不验证实测来源或绝对精度。",
                  "输入清单最多缓存60秒；日志详情按需只读尾部最多64KiB。"]
        log_path = self.output / "production.log"
        try:
            log_path = scoped_path(self.output, log_path)
            excerpt = health.log_excerpt(log_path if log_path.is_file() else None)
        except (OSError, ValueError, RuntimeError) as exc:
            excerpt = {"recent_events": [], "error": str(exc)}
        if excerpt.get("error"):
            warnings.append("调度日志不可读：" + excerpt["error"])
        overview = {"schema_version": 1, "output_dir": str(self.output), "data_path": str(self.data_path) if self.data_path else None,
            "generated_at": health.utc_now(), "refresh_interval_seconds": self.refresh_seconds, "refreshing": False, "last_error": None,
            "health": runtime, "counts": counts,
            "coverage": {"input_total": total, "terminal_count": terminal, "published_count": published,
                         "unreported_count": unreported, "ratio": terminal / total if total and terminal is not None else None, "label": self._coverage_label},
            "takes": list(rows.values()), "groups": sorted({row["group"] for row in rows.values()}),
            "plan": self.plan.stats(),
            "kinds": sorted({row["kind"] for row in rows.values()}), "recent_events": excerpt["recent_events"],
            "warnings": list(dict.fromkeys(warnings)), "limits": limits}
        overview["run"] = dict(self.run)
        overview["inventory"] = dict(self.inventory)
        overview["inventory_units"] = {"source_failed": "source_reports", "invalid_checkpoint": "source_reports",
                                       "identity_conflict": "source_reports", "duplicate": "source_reports",
                                       "skipped_informal_name": "source_reports",
                                       "identities_without_reusable_checkpoint": "unique_job_ids",
                                       "roster_pending": "roster_evidence_items"}
        required_rechecks = self.run.get("kind") == "human_repair"
        rechecks_applicable = required_rechecks or self.run.get("kind") == "quality_recheck" or bool(attempts) or any(
            row["independent_recheck"]["status"] != "PENDING" for row in rows.values())
        overview["rechecks"] = {name.lower(): sum(row["independent_recheck"]["status"] == name for row in rows.values())
                                if self._report_scan_complete and rechecks_applicable and (not required_rechecks or sources is not None)
                                and (name != "PENDING" or required_rechecks) else None
                                for name in ("PASSED", "FAILED", "RUNNING", "PENDING", "UNKNOWN")}
        return {"overview": overview, "rows": rows, "details": details, "reports": reports}

    def target_audit(self):
        audit = self.read_local_json(self.output / "repair_audit.json")
        if not isinstance(audit, dict) or audit.get("subject") != self.run.get("subject"):
            raise ValueError("target audit subject disagrees with catalog")
        if audit.get("status") not in ("SUCCESS", "NEEDS_REVIEW", "FAILED"):
            raise ValueError("invalid target audit status")
        scope = audit.get("quality_scope")
        if (not isinstance(scope, dict) or scope.get("subject") != self.run["subject"]
                or scope.get("human_geometry") != "target_only"
                or scope.get("other_humans_validated") is not False or scope.get("props_validated") is not False):
            raise ValueError("target audit must explicitly limit quality scope")
        source = audit.get("source")
        if not isinstance(source, str) or not Path(source).is_absolute():
            raise ValueError("invalid target audit source")
        if self.run.get("input_path") and Path(source).resolve() != Path(self.run["input_path"]).resolve():
            raise ValueError("target audit source disagrees with catalog")
        if not isinstance(audit.get("accepted"), bool) or not isinstance(audit.get("source_unchanged"), bool):
            raise ValueError("missing target preservation/acceptance audit")
        if audit["status"] == "SUCCESS" and (not audit["accepted"] or not audit["source_unchanged"]):
            raise ValueError("inconsistent target audit success")
        if not isinstance(audit.get("artifacts"), dict) or any(not safe_component(key) or not isinstance(value, str)
                                                             for key, value in audit["artifacts"].items()):
            raise ValueError("invalid target artifact declarations")
        if not isinstance(audit.get("review_reasons"), list) or any(not isinstance(item, str) for item in audit["review_reasons"]):
            raise ValueError("invalid target review reasons")
        if "range" in audit and not isinstance(audit["range"], dict):
            raise ValueError("invalid target range")
        return audit

    def collect_target(self):
        """One subject audit, not an invented canonical full-scene report."""
        key = "target-" + hashlib.sha256(self.run["subject"].encode()).hexdigest()[:16]
        warnings, audit, record = [], {}, {}
        error = None
        try:
            audit = self.target_audit()
            path = scoped_path(self.output, self.output / "repair_audit.json")
            report = dict(audit, job_id=key, input_path=audit["source"])
            record = {"path": str(path), "report": report, "modified": path.stat().st_mtime}
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            error = str(exc)
            warnings.append("Target audit unavailable: " + error)
        pid = self.run.get("scheduler_pid")
        observed = self.snapshot_provider([pid] if health.positive_int(pid) else [])
        processes = health.process_map(observed)
        owner = processes.get(pid)
        tokens = health.command_tokens((owner or {}).get("command_line"))
        target = health.option_value(tokens, "--output-dir")
        subject = health.option_value(tokens, "--subject")
        source = health.option_value(tokens, "--input-vdf")
        created, started = health.parse_time((owner or {}).get("created_at")), health.parse_time(self.run.get("started_at"))
        verified = bool(owner and created and started and created <= started
                        and owner.get("name", "").casefold() in health.PYTHON_NAMES
                        and target and ntpath.isabs(target) and health.normalize_path(target) == health.normalize_path(self.output)
                        and subject == self.run["subject"] and source and ntpath.isabs(source)
                        and (not self.run.get("input_path") or health.normalize_path(source) == health.normalize_path(self.run["input_path"]))
                        and any(ntpath.basename(token).casefold() == "step2_repair_single_human.py" for token in tokens))
        native = [process for process in processes.values() if verified and process.get("ppid") == pid
                  and process.get("name", "").casefold() in health.NATIVE_NAMES]
        known = observed.get("status") == "ok"
        status = "RUNNING" if verified else audit.get("status") or ("UNKNOWN" if error else "UNREPORTED")
        artifacts, integrity = self.artifact_rows(key, record)
        # SUCCESS here is explicitly target-only; full-scene usable remains null.
        scope = audit.get("quality_scope") or {"subject": self.run["subject"], "human_geometry": "target_only",
                                               "other_humans_validated": False, "props_validated": False}
        row = {"id": key, "name": self.run["subject"], "relative_path": "", "input_path": self.run.get("input_path") or audit.get("source"),
               "group": "UNKNOWN", "kind": "human_target", "status": status, "reported_status": audit.get("status"),
               "usable": None, "target_accepted": audit.get("accepted"), "quality_scope": scope,
               "artifact_integrity": integrity, "subjects": [self.run["subject"]],
               "review_count": len(audit.get("review_reasons", [])), "error": error or audit.get("error"),
               "frames": audit.get("range", {}).get("frame_count"), "duration_seconds": None,
               "updated_at": iso_epoch(record.get("modified")), "started_at": self.run.get("started_at"),
               "summary_reason": "单人修复审计；SUCCESS 仅限目标人体，不验证其他人体、道具或完整场景可用性。",
               "active": {"worker_pid": pid, "native_pid": native[0]["pid"] if native else None, "phase": "UNKNOWN"} if verified else None}
        complete = bool(audit and known and owner is None and integrity in ("complete", "not_applicable")
                        and not any(process.get("name", "").casefold() in health.NATIVE_NAMES for process in processes.values()))
        runtime = {"overall": "RUNNING" if verified else "COMPLETED" if complete else "UNKNOWN",
                   "reason": row["summary_reason"], "run_id": self.run.get("run_id"), "scheduler_pid": pid,
                   "verified_owner": verified, "active_count": 1 if verified else 0 if known and owner is None else None,
                   "active_progress_count": None, "native_busy_count": None, "python_busy_count": None,
                   "memory_free_bytes": (observed.get("memory") or {}).get("free_bytes"),
                   "memory_total_bytes": (observed.get("memory") or {}).get("total_bytes"), "observed_at": observed.get("sampled_at")}
        terminal = int(bool(audit) and not verified) if known else None
        detail = {"take": row, "quality": {"usable": None, "scope": scope, "target_accepted": audit.get("accepted")},
                  "quality_scope": scope, "repair_audit": audit, "subjects": [{"name": self.run["subject"],
                      "before": audit.get("before"), "after": audit.get("after")}],
                  "props": [], "stages": [], "artifacts": artifacts, "review_reasons": audit.get("review_reasons", []),
                  "background_review_reasons": [], "error": row["error"], "report_path": record.get("path"),
                  "limits": scope, "session": {}, "dataset": None, "policy": None}
        overview = {"schema_version": 1, "run": dict(self.run), "output_dir": str(self.output),
                    "data_path": str(self.data_path) if self.data_path else None,
                    "generated_at": health.utc_now(), "refresh_interval_seconds": self.refresh_seconds,
                    "refreshing": False, "last_error": None, "health": runtime,
                    "counts": {value: int(status == value) for value in STATUSES},
                    "coverage": {"input_total": 1, "terminal_count": terminal, "published_count": int(bool(audit) and integrity == "complete"),
                                 "unreported_count": int(not audit and not verified), "ratio": terminal,
                                 "label": "单目标人体修复审计覆盖；不是完整场景质量或实时队列进度"},
                    "takes": [row], "groups": ["UNKNOWN"], "kinds": ["human_target"], "plan": self.plan.stats(),
                    "inventory": {}, "rechecks": {name: None for name in ("passed", "failed", "running", "pending", "unknown")},
                    "warnings": warnings, "limits": [row["summary_reason"]], "recent_events": []}
        return {"overview": overview, "rows": {key: row}, "details": {key: detail},
                "reports": {key: record} if record else {}}

    def detail(self, snapshot, key):
        if key not in snapshot["details"]:
            raise KeyError(key)
        detail = dict(snapshot["details"][key])
        # 源文件最后修改时间（≈拍摄保存时间）：只读 stat，缺失/无权限时保持 null。
        input_path = detail["take"].get("input_path")
        detail["input_modified"] = None
        if isinstance(input_path, str) and input_path:
            try:
                detail["input_modified"] = datetime.fromtimestamp(Path(input_path).stat().st_mtime, timezone.utc).isoformat()
            except OSError:
                pass
        relative = safe_relative(detail["take"]["relative_path"])
        # Never trust report-provided logs or reconstruct an output nickname.
        # The raw identity selects the helper's exact per-take log directory.
        logs = {}
        for role, name in (("processing", "processing.log"), ("shogun", "shogun.log")):
            path = layout.take_paths(self.output, relative)["log_dir"] / name if relative else None
            logs[role] = tail_log(self.output, path)
        detail["logs"] = logs
        return detail

    def open_artifact(self, snapshot, key, role):
        record = snapshot["reports"].get(key)
        if not record or not record.get("report") or not safe_component(role):
            raise KeyError(role)
        # Recheck this one canonical report at the download boundary so stale
        # cache links cannot expose a role removed by a newer publication.
        report_path = scoped_path(self.output, record["path"], run_owned=True)
        if self.run.get("kind") == "human_target":
            audit = self.target_audit()
            expected = "target-" + hashlib.sha256(self.run["subject"].encode()).hexdigest()[:16]
            if key != expected or role not in audit["artifacts"]:
                raise KeyError(role)
            return open_scoped(self.output, audit["artifacts"][role], relative_to=report_path.parent,
                               forbidden=lambda path: self.source_file(path, {"input_path": audit["source"]}))
        layout.report_relative(self.output, report_path)
        report, _, error = read_json_file(report_path, self.output, run_owned=True)
        if error or health.validate_report(report) or report.get("job_id") != key or role not in report.get("artifacts", {}):
            raise KeyError(role)
        return open_scoped(self.output, report["artifacts"][role], relative_to=report_path.parent,
                           forbidden=lambda path: self.artifact_forbidden(path, report, report_path), run_owned=True)


class SnapshotCache:
    """One sampler, immutable published snapshots, and coalesced refresh requests."""
    def __init__(self, data):
        self.data = data
        self._condition = threading.Condition()
        self._snapshot = None
        self._refreshing = False
        self._last_error = None
        self._requested = False
        self._stopped = False
        self._thread = None

    def start(self):
        with self._condition:
            if self._thread is not None:
                return
            self._requested = self._refreshing = True
            self._thread = threading.Thread(target=self._run, name="shogun-dashboard-sampler", daemon=True)
            self._thread.start()

    def close(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=35)

    def request_refresh(self):
        with self._condition:
            if self._refreshing or self._stopped:
                return False
            self._requested = self._refreshing = True
            self._condition.notify()
            return True

    def _run(self):
        deadline = time.monotonic()
        while True:
            with self._condition:
                while not self._stopped and not self._requested and time.monotonic() < deadline:
                    self._condition.wait(deadline - time.monotonic())
                if self._stopped:
                    return
                self._requested = False
                self._refreshing = True
            started = time.monotonic()
            try:
                snapshot = self.data.collect()
            except Exception as exc:
                with self._condition:
                    self._last_error = f"{type(exc).__name__}: {exc}"
            else:
                with self._condition:
                    self._snapshot = snapshot
                    self._last_error = None
            with self._condition:
                self._refreshing = False
                self._condition.notify_all()
            deadline = max(started + self.data.refresh_seconds, time.monotonic())

    def snapshot(self):
        with self._condition:
            return self._snapshot

    def overview(self, refresh=False):
        if refresh:
            self.request_refresh()
        with self._condition:
            if self._snapshot is not None:
                value = dict(self._snapshot["overview"])
            else:
                value = {"schema_version": 1, "output_dir": str(self.data.output),
                    "data_path": str(self.data.data_path) if self.data.data_path else None, "generated_at": None,
                    "refresh_interval_seconds": self.data.refresh_seconds,
                    "health": {"overall": "LOADING", "reason": "正在收集首次只读快照。", "run_id": None,
                        "scheduler_pid": None, "verified_owner": False, "active_count": None, "active_progress_count": None,
                        "python_busy_count": None, "native_busy_count": None, "memory_free_bytes": None,
                        "memory_total_bytes": None, "observed_at": None}, "counts": dict.fromkeys(STATUSES),
                    "coverage": {"input_total": None, "terminal_count": None, "published_count": None,
                        "unreported_count": None, "ratio": None, "label": COVERAGE_LABEL},
                    "takes": [], "groups": [], "kinds": [], "recent_events": [], "warnings": [], "limits": []}
            value.update(refreshing=self._refreshing, last_error=self._last_error)
            return value
