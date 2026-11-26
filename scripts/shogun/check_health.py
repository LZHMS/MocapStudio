"""One-shot, read-only Shogun batch evidence collector (Python standard library only).

Relative artifact references are resolved relative to their canonical report.json.
Sequence-local manifests scope discovery to the exact run's processed roots;
generic centralized output remains supported. No SDK is imported or connected.
Process snapshots are deliberately injectable; importing this module starts nothing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import ntpath
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

from pipeline_layout import bucket_entries, in_run, is_generated, report_paths, report_relative, take_paths


STATUSES = ("SUCCESS", "NEEDS_REVIEW", "FAILED")
PYTHON_NAMES = {"python.exe", "pythonw.exe", "python3.exe", "python3.13.exe"}
NATIVE_NAMES = {"shogunpostcl.exe", "shogunpost.exe"}
SCHEDULER_SCRIPTS = {"step1_process_takes.py", "step3_repair_reviews.py", "step2_repair_humans.py"}
TAIL_BYTES = 65536


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_time(value):
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except ValueError:
        return None


def positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def read_json(path):
    """A changed/in-flight JSON file is unknown, never an empty successful result."""
    try:
        before = path.stat()
        with path.open("r", encoding="utf-8-sig") as stream:
            value = json.load(stream)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            return None, None, "file changed while being read"
        if not isinstance(value, dict):
            return None, None, "expected a JSON object"
        return value, after.st_mtime, None
    except (OSError, ValueError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def optional_json(path):
    try:
        path.stat()
    except FileNotFoundError:
        return {"path": str(path), "present": False, "data": None, "error": None}
    except OSError as exc:
        return {"path": str(path), "present": None, "data": None, "error": str(exc)}
    data, modified, error = read_json(path)
    return {"path": str(path), "present": True, "data": data,
            "modified_at": datetime.fromtimestamp(modified, timezone.utc).isoformat() if modified else None,
            "modified_epoch": modified,
            "error": error}


def scan_sessions(output):
    result = {"entries": [], "errors": []}
    try:
        children = list(bucket_entries(output, ".work"))
    except (OSError, ValueError) as exc:
        result["errors"].append(f"{output}: {exc}")
        return result
    for child in children:
        try:
            if not child.is_dir():
                continue
            if not in_run(output, child / ".control" / "session-ready.json"):
                result["errors"].append(f"{child}: native session escapes run scope")
                continue
            record = optional_json(child / ".control" / "session-ready.json")
            if record["present"] is False:
                continue  # An initializing workspace has no ready native session yet.
            data = record["data"]
            if not record["error"] and (not positive_int(data.get("pid"))
                    or not positive_int(data.get("port")) or data["port"] > 65535
                    or parse_time(data.get("started_at")) is None):
                record["error"] = "invalid native PID, port or timezone-aware started_at"
            record["workspace"] = str(child)
            record["job_id"] = child.name.split("-", 1)[0]
            result["entries"].append(record)
            if record["error"]:
                result["errors"].append(f"{record['path']}: {record['error']}")
        except OSError as exc:
            result["errors"].append(f"{child}: {exc}")
    return result


def process_snapshot(pids=()):
    """One scoped CIM request; no windows, SDK connections or process control."""
    if os.name != "nt":
        return {"status": "unknown", "sampled_at": utc_now(), "processes": [],
                "error": "Process identity/activity is only queried on Windows", "memory": None}
    names = sorted(PYTHON_NAMES | NATIVE_NAMES)
    predicates = [f"Name='{name}'" for name in names]
    predicates.extend(f"ProcessId={pid}" for pid in sorted(set(pids)) if positive_int(pid))
    query = " OR ".join(predicates)
    script = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$items = @(Get-CimInstance Win32_Process -Filter '__FILTER__' | ForEach-Object {
    $created = $null
    if ($null -ne $_.CreationDate) { $created = $_.CreationDate.ToUniversalTime().ToString('o') }
    $cpu = $null
    if ($null -ne $_.KernelModeTime -and $null -ne $_.UserModeTime) {
        $cpu = ([double]$_.KernelModeTime + [double]$_.UserModeTime) / 10000000.0
    }
    [pscustomobject]@{pid=[int]$_.ProcessId; ppid=[int]$_.ParentProcessId;
        name=$_.Name; created_at=$created; command_line=$_.CommandLine;
        cpu_seconds=$cpu; working_set_bytes=$_.WorkingSetSize;
        read_bytes=$_.ReadTransferCount; write_bytes=$_.WriteTransferCount}
})
$memory = $null
$memoryError = $null
try {
    $os = Get-CimInstance Win32_OperatingSystem
    $memory = @{total_bytes=([double]$os.TotalVisibleMemorySize * 1024);
                free_bytes=([double]$os.FreePhysicalMemory * 1024)}
} catch { $memoryError = $_.Exception.Message }
[pscustomobject]@{status='ok'; sampled_at=[DateTime]::UtcNow.ToString('o');
    processes=$items; memory=$memory; memory_error=$memoryError} | ConvertTo-Json -Depth 5 -Compress
""".replace("__FILTER__", query.replace("'", "''"))
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, encoding="utf-8-sig", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW, timeout=30, check=False)
        if completed.returncode:
            raise OSError(f"PowerShell exit {completed.returncode}: {completed.stderr.strip()[:1000]}")
        data = json.loads(completed.stdout)
        if not isinstance(data, dict) or data.get("status") != "ok" or not isinstance(data.get("processes"), list):
            raise ValueError("invalid process snapshot payload")
        for row in data["processes"]:
            if not isinstance(row, dict) or not positive_int(row.get("pid")) or not isinstance(row.get("name"), str):
                raise ValueError("invalid process identity in snapshot")
        return data
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"status": "unknown", "sampled_at": utc_now(), "processes": [],
                "error": f"{type(exc).__name__}: {exc}", "memory": None}


def run_start(run_id):
    try:
        return datetime.strptime(run_id.split("-", 1)[0], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


def normalize_path(value):
    return ntpath.normcase(ntpath.normpath(str(value).replace("/", "\\")))


def command_tokens(value):
    return [token.strip('"') for token in re.findall(r'"[^"\r\n]*"|[^\s"]+', value or "")]


def option_value(tokens, name):
    for index, token in enumerate(tokens):
        if token.casefold() == name.casefold() and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def process_map(snapshot):
    return {row["pid"]: row for row in snapshot.get("processes", [])}


def identify_lock(record, snapshot, output):
    result = {"path": record["path"], "present": record["present"], "identity": "UNKNOWN",
              "pid": None, "run_id": None, "reason": record.get("error")}
    if record["present"] is False:
        result.update(identity="ABSENT", reason="No pipeline lock observed")
        return result
    lock = record.get("data")
    if not lock or not positive_int(lock.get("pid")) or not run_start(lock.get("run_id")):
        result["reason"] = result["reason"] or "Invalid lock PID/run_id or in-flight lock write"
        return result
    result.update(pid=lock["pid"], run_id=lock["run_id"])
    if snapshot.get("status") != "ok":
        result["reason"] = "Process query unavailable; lock liveness is unknown"
        return result
    owner = process_map(snapshot).get(lock["pid"])
    if owner is None:
        result.update(identity="DEAD", reason="Lock PID is absent from a successful scoped process query")
        return result
    result["process"] = owner
    created = parse_time(owner.get("created_at"))
    start = run_start(lock["run_id"])
    tokens = command_tokens(owner.get("command_line"))
    explicit_output = option_value(tokens, "-OutputDir") or option_value(tokens, "--output-dir")
    if owner.get("name", "").casefold() not in PYTHON_NAMES:
        result.update(identity="MISMATCH", reason="Lock PID belongs to a non-Python process (possible PID reuse)")
    elif created and (created - start).total_seconds() > 2:
        result.update(identity="MISMATCH", reason="Process was created after the lock run began (possible PID reuse)")
    elif tokens and not any(ntpath.basename(token).casefold() in SCHEDULER_SCRIPTS for token in tokens):
        result.update(identity="MISMATCH", reason="Lock PID command does not run a supported Shogun scheduler")
    elif explicit_output and not ntpath.isabs(explicit_output):
        result["reason"] = "Scheduler uses a relative output path; its working directory is not available"
    elif explicit_output and normalize_path(explicit_output) != normalize_path(output):
        result.update(identity="MISMATCH", reason="Lock PID command targets another output directory")
    elif created is None or not tokens or explicit_output is None:
        result["reason"] = "Creation time or explicit scheduler/output command identity is unavailable"
    else:
        result.update(identity="OWNED", reason="Python command, output path and creation time match this lock")
    return result


def same_process(left, right):
    return bool(left and right and left.get("pid") == right.get("pid")
                and left.get("name", "").casefold() == right.get("name", "").casefold()
                and parse_time(left.get("created_at")) is not None
                and parse_time(left.get("created_at")) == parse_time(right.get("created_at")))


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def activity(first, second):
    result = {"cpu_seconds_delta": None, "read_bytes_delta": None, "write_bytes_delta": None,
              "observed_activity": False}
    if not same_process(first, second):
        return result
    for source, destination in (("cpu_seconds", "cpu_seconds_delta"), ("read_bytes", "read_bytes_delta"),
                                ("write_bytes", "write_bytes_delta")):
        before, after = first.get(source), second.get(source)
        if numeric(before) and numeric(after) and after >= before:
            result[destination] = after - before
    result["observed_activity"] = any(numeric(result[key]) and result[key] > 0 for key in
                                     ("cpu_seconds_delta", "read_bytes_delta", "write_bytes_delta"))
    return result


def identify_sessions(records, snapshot, lock):
    processes = process_map(snapshot)
    rows = []
    for record in records["entries"]:
        row = {"workspace": record["workspace"], "job_id": record["job_id"],
               "ownership": "UNKNOWN", "reason": record["error"]}
        data = record.get("data")
        if record["error"] or not data:
            rows.append(row)
            continue
        row.update(native_pid=data["pid"], port=data["port"], version=data.get("version"),
                   started_at=data["started_at"])
        now = parse_time(snapshot.get("sampled_at"))
        row["age_seconds"] = max(0, (now - parse_time(data["started_at"])).total_seconds()) if now else None
        native = processes.get(data["pid"])
        if snapshot.get("status") != "ok":
            row["reason"] = "Process query unavailable"
        elif native is None:
            row.update(ownership="STALE", reason="Historical ready PID is no longer present")
        else:
            created = parse_time(native.get("created_at"))
            worker = processes.get(native.get("ppid"))
            worker_created = parse_time(worker.get("created_at")) if worker else None
            owner_created = parse_time(lock.get("process", {}).get("created_at"))
            native_tokens = command_tokens(native.get("command_line"))
            port = option_value(native_tokens, "-controlStreamPort")
            if native.get("name", "").casefold() != "shogunpostcl.exe":
                row.update(ownership="STALE", reason="Ready PID now belongs to an unrelated process")
            elif created and (created - parse_time(data["started_at"])).total_seconds() > 2:
                row.update(ownership="STALE", reason="Ready PID was reused after this session started")
            elif port is not None and port != str(data["port"]):
                row.update(ownership="STALE", reason="Native command port differs from historical ready metadata")
            elif lock["identity"] != "OWNED":
                row["reason"] = "No verified scheduler identity for the native parent chain"
            elif not created or not worker_created or not owner_created or port is None:
                row["reason"] = "Native/worker creation time or command identity is unavailable"
            elif (worker.get("name", "").casefold() not in PYTHON_NAMES
                  or worker.get("ppid") != lock["pid"]
                  or not owner_created <= worker_created <= created):
                row.update(ownership="STALE", reason="Live PID is not in this scheduler's valid worker parent chain")
            else:
                row.update(ownership="OWNED", reason="Native name/time/port and worker/scheduler parent chain match",
                           worker_pid=worker["pid"], process=native)
        rows.append(row)
    return rows


def log_excerpt(path):
    result = {"path": str(path) if path else None, "tail_byte_limit": TAIL_BYTES,
              "truncated": False, "recent_events": [], "worker_hints": [], "error": None}
    if path is None:
        return result
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            start = max(0, size - TAIL_BYTES)
            stream.seek(start)
            chunk = stream.read(TAIL_BYTES)
        if start:
            chunk = chunk.split(b"\n", 1)[-1] if b"\n" in chunk else b""
        # A partial last line may still be being flushed by the scheduler.
        if chunk and not chunk.endswith(b"\n"):
            chunk = chunk.rsplit(b"\n", 1)[0] if b"\n" in chunk else b""
        result["truncated"] = bool(start)
        events = []
        for line in chunk.decode("utf-8-sig", errors="replace").splitlines():
            if re.search(r"\[(?:RUN|SUCCESS|NEEDS_REVIEW|FAILED|ERROR|SUMMARY|INTERRUPTED)\]", line):
                events.append(line[:500])
            match = re.search(r"\[RUN\] worker=(\d+) port=(\d+) (.*)", line)
            if match:
                result["worker_hints"].append({"worker_pid": int(match[1]), "port": int(match[2]),
                                               "input_path": match[3][:500]})
        result["recent_events"] = events[-20:]
        result["worker_hints"] = result["worker_hints"][-32:]
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def failure_class(error):
    lowered = str(error).casefold()
    if "x2d" in lowered and any(word in lowered for word in ("missing", "not found", "no x2d", "no matching", "cannot find")):
        return "missing_x2d"
    if "worker exited without" in lowered and ("report" in lowered or "result" in lowered):
        return "worker_exited_without_result"
    return "other"


def review_details(report):
    categories = set()
    reasons = report.get("review_reasons", [])
    human_quality = Counter()
    export_quality = Counter()
    humans = {row.get("name") for row in report.get("subjects", []) if row.get("processing_target") is True
              and row.get("rig_class") == "HUMAN_RIG"}

    def classify(reason):
        lower = reason.casefold()
        if any(word in lower for word in ("count_mismatch", "roster", "missing_expected_subject", "unexpected_subject", "unexpected_human")):
            categories.add("roster")
        elif lower.startswith("prop:"):
            categories.add("prop")
        elif reason.split(":", 1)[0] in humans:
            categories.add("human")
        else:
            categories.add("other")

    for reason in reasons:
        classify(reason)
    for row in report.get("subjects", []):
        if row.get("processing_target") is True and row.get("rig_class") == "HUMAN_RIG":
            quality = row.get("quality", {})
            usable = quality.get("usable") if quality.get("basis") == "post_processing" else None
            human_quality["usable" if usable is True else "review" if usable is False else "unknown"] += 1
            if usable is False or row.get("review_reasons"):
                categories.add("human")
    if any(row.get("review_reasons") for row in report.get("props", [])):
        categories.add("prop")
    for row in report.get("subject_exports", []):
        quality = row.get("quality", {})
        usable = quality.get("usable") if quality.get("basis") == "post_processing" else None
        export_quality["usable" if usable is True else "review" if usable is False else "unknown"] += 1
        for reason in quality.get("review_reasons", []):
            classify(reason)
    if report["status"] == "NEEDS_REVIEW" and not categories:
        categories.add("other")
    return categories, human_quality, export_quality


def validate_report(data):
    if not isinstance(data.get("job_id"), str) or not data["job_id"] or data.get("status") not in STATUSES:
        return "missing/invalid job_id or status"
    for key in ("subjects", "props", "subject_exports"):
        value = data.get(key, [])
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            return f"invalid {key}"
        for row in value:
            if "quality" in row and not isinstance(row["quality"], dict):
                return f"invalid {key}.quality"
            if "name" in row and not isinstance(row["name"], str):
                return f"invalid {key}.name"
            for reasons in (row.get("review_reasons", []), row.get("quality", {}).get("review_reasons", [])):
                if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
                    return f"invalid {key} review reasons"
    reasons = data.get("review_reasons", [])
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        return "invalid review_reasons"
    if not isinstance(data.get("artifacts", {}), dict):
        return "invalid artifacts mapping"
    if data["status"] != "FAILED":
        quality = data.get("quality", {})
        if not isinstance(quality, dict) or quality.get("basis") != "post_processing":
            return "final post_processing quality evidence is missing"
        if quality.get("usable") is not (data["status"] == "SUCCESS"):
            return "status and final quality.usable disagree"
        for row in data.get("subjects", []):
            if row.get("processing_target") is True:
                quality = row.get("quality", {})
                if (row.get("rig_class") != "HUMAN_RIG" or quality.get("basis") != "post_processing"
                        or not isinstance(quality.get("usable"), bool)):
                    return "human target final rig/quality evidence is missing"
                if data["status"] == "SUCCESS" and quality["usable"] is not True:
                    return "SUCCESS disagrees with final human target quality"
        for row in data.get("subject_exports", []):
            quality = row.get("quality", {})
            if quality.get("basis") != "post_processing" or not isinstance(quality.get("usable"), bool):
                return "subject export final quality evidence is missing"
            if data["status"] == "SUCCESS" and quality["usable"] is not True:
                return "SUCCESS disagrees with final subject export quality"
    return None


def check_artifacts(report, report_path, output=None):
    issues = []
    products = report.get("artifacts", {})
    roots = None
    if output is not None:
        paths = take_paths(output, report_relative(output, report_path))
        roots = (paths["destination"].resolve(), paths["review_destination"].resolve())
    required = {"vdf", "fbx"}
    source_path = report.get("input_path", "")
    kind = report.get("input_kind") or (Path(source_path).suffix.lower().lstrip(".") if isinstance(source_path, str) else "")
    if kind == "mcp":
        required.add("vsk")
    elif kind != "vdf":
        issues.append({"role": None, "path": None, "state": "unknown", "reason": "input kind is unknown"})
    for role in sorted(required - products.keys()):
        issues.append({"role": role, "path": None, "state": "missing", "reason": "required artifact role absent"})
    for role, value in products.items():
        if not isinstance(value, str) or not value:
            issues.append({"role": role, "path": value, "state": "unknown", "reason": "artifact reference is not a nonempty path string"})
            continue
        if ntpath.isabs(value) and not Path(value).is_absolute() and os.name != "nt":
            issues.append({"role": role, "path": value, "state": "unknown", "reason": "Windows absolute artifact path cannot be stat'ed on this platform"})
            continue
        path = Path(value) if Path(value).is_absolute() else report_path.parent / value
        if roots is not None and (not in_run(output, path)
                or not any(path.resolve().is_relative_to(root) for root in roots)):
            issues.append({"role": role, "path": str(path), "state": "unknown",
                           "reason": "artifact reference is outside this take's artifact roots"})
            continue
        try:
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                issues.append({"role": role, "path": str(path), "state": "missing", "reason": "not a nonempty regular file"})
        except FileNotFoundError:
            issues.append({"role": role, "path": str(path), "state": "missing", "reason": "referenced file does not exist"})
        except OSError as exc:
            issues.append({"role": role, "path": str(path), "state": "unknown", "reason": str(exc)})
    for export in report.get("subject_exports", []):
        role = export.get("artifact_role")
        if not isinstance(role, str) or role not in products:
            issues.append({"role": role, "path": export.get("path"), "state": "missing",
                           "reason": "subject export has no current artifacts role"})
    return issues


def report_inventory(output, since=None):
    result = {"scope": "current canonical report stock, NOT completions in this run",
              "canonical_files": 0, "unique_jobs": 0, "status_counts": {key: 0 for key in STATUSES},
              "verified_counts": {key: 0 for key in STATUSES}, "unknown_count": 0,
              "failure_classes": {key: 0 for key in ("missing_x2d", "worker_exited_without_result", "other")},
              "failures": [], "issues": [], "artifact_issues": [], "missing_artifact_count": 0,
              "review_categories": {key: 0 for key in ("human", "prop", "roster", "other")},
              "review_reasons": {}, "human_quality": {}, "subject_export_quality": {},
              "since": since.isoformat() if since else None,
              "newer_report_counts": {key: 0 for key in STATUSES}, "new_failure_events": None}
    jobs = {}
    try:
        paths = list(report_paths(output))
    except (OSError, ValueError) as exc:
        result["issues"].append(f"{output}: {exc}")
        result["unknown_count"] += 1
        return result, jobs
    for path in paths:
        result["canonical_files"] += 1
        data, modified, error = read_json(path)
        try:
            relative = report_relative(output, path)
            if data is not None and data.get("relative_path") is not None and Path(data["relative_path"]) != relative:
                error = error or "report relative_path disagrees with canonical take identity"
        except (OSError, TypeError, ValueError) as exc:
            error = error or str(exc)
        error = error or validate_report(data)
        if error:
            result["unknown_count"] += 1
            result["issues"].append(f"{path}: {error}")
            continue
        job_id = data["job_id"]
        if job_id in jobs:
            result["issues"].append(f"Duplicate job_id {job_id}: {jobs[job_id]['path']} and {path}; newest report selected, ownership ambiguous")
            result["unknown_count"] += 1
            if modified <= jobs[job_id]["modified"]:
                continue
        jobs[job_id] = {"report": data, "path": str(path), "modified": modified}

    result["unique_jobs"] = len(jobs)
    reason_counts, humans, exports = Counter(), Counter(), Counter()
    for job_id, entry in jobs.items():
        report, path = entry["report"], Path(entry["path"])
        status = report["status"]
        result["status_counts"][status] += 1
        if since and entry["modified"] >= since.timestamp():
            result["newer_report_counts"][status] += 1
        if status == "FAILED":
            category = failure_class(report.get("error", ""))
            result["failure_classes"][category] += 1
            result["failures"].append({"job_id": job_id, "report": str(path), "classification": category,
                                        "error": str(report.get("error", ""))[:1000],
                                        "event_time_known": False})
            result["verified_counts"][status] += 1
        else:
            issues = check_artifacts(report, path, output)
            for issue in issues:
                issue.update(job_id=job_id, report=str(path))
            result["artifact_issues"].extend(issues)
            if issues:
                result["unknown_count"] += 1
            else:
                result["verified_counts"][status] += 1
        if status != "FAILED":
            categories, human, exported = review_details(report)
            for category in categories:
                result["review_categories"][category] += 1
            reason_counts.update(report.get("review_reasons", []))
            humans.update(human)
            exports.update(exported)
    result["missing_artifact_count"] = sum(issue["state"] == "missing" for issue in result["artifact_issues"])
    result["review_reasons"] = dict(reason_counts)
    result["human_quality"] = dict(humans)
    result["subject_export_quality"] = dict(exports)
    return result, jobs


def final_summary(record, jobs):
    result = {"path": record["path"], "present": record["present"], "complete": False,
              "reason": record.get("error"), "run_id": None, "counts": None}
    data = record.get("data")
    if not data:
        result["reason"] = result["reason"] or "No readable final batch summary"
        return result
    result.update(run_id=data.get("run_id"), counts=data.get("counts"))
    rows, counts = data.get("jobs"), data.get("counts")
    if not run_start(data.get("run_id")) or not isinstance(rows, list) or not isinstance(counts, dict):
        result["reason"] = "Invalid final batch summary schema"
        return result
    if any(not isinstance(counts.get(key), int) or isinstance(counts[key], bool) or counts[key] < 0 for key in STATUSES):
        result["reason"] = "Invalid final counts"
        return result
    summary_time = record.get("modified_epoch")
    if not numeric(summary_time) or any(entry["modified"] > summary_time for entry in jobs.values()):
        result["reason"] = "Current canonical reports are newer than the final summary; it cannot establish current completion"
        return result
    seen, actual = set(), Counter()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("job_id"), str) or row["job_id"] in seen or row.get("status") not in STATUSES:
            result["reason"] = "Invalid or duplicate jobs in final summary"
            return result
        seen.add(row["job_id"])
        actual[row["status"]] += 1
        current = jobs.get(row["job_id"])
        if current is None or current["report"]["status"] != row["status"]:
            result["reason"] = "Final summary does not match current canonical reports"
            return result
    if not rows or any(actual[key] != counts[key] for key in STATUSES):
        result["reason"] = "Final summary jobs/counts are incomplete or inconsistent"
        return result
    result.update(complete=True, reason="Final jobs/counts agree with the referenced current canonical reports")
    return result


def collect_health(output, sample_seconds=3, log=None, data_path=None, since=None,
                   snapshot_provider=process_snapshot, sleeper=time.sleep):
    """Collect twice, with injected snapshots/sleep for deterministic process-free tests."""
    output = Path(output).expanduser().resolve()
    if not output.is_dir():
        raise ValueError(f"Output directory does not exist or is inaccessible: {output}")
    if not numeric(sample_seconds) or sample_seconds < 0:
        raise ValueError("sample-seconds must be finite and nonnegative")
    if since is not None and (not isinstance(since, datetime) or since.tzinfo is None):
        raise ValueError("since must be a timezone-aware ISO timestamp")
    if log is not None:
        log = Path(log).expanduser().resolve()
        if not log.is_file():
            raise ValueError(f"Explicit log does not exist or is inaccessible: {log}")
    else:
        candidate = output / "production.log"
        log = candidate if candidate.is_file() else None
    inventory = None
    if data_path is not None:
        source = Path(data_path).expanduser().resolve()
        if not source.exists():
            raise ValueError(f"Data path does not exist: {source}")
        errors = []
        count = 0
        if source.is_file():
            if source.suffix.casefold() != ".mcp":
                raise ValueError("data-path file must be an MCP (no VDFs are loaded or inspected)")
            count = 1
        elif source.is_dir():
            for directory, directories, files in os.walk(source, onerror=lambda exc: errors.append(str(exc))):
                directories[:] = [name for name in directories
                                  if not is_generated(Path(directory) / name, source)]
                count += sum(Path(name).suffix.casefold() == ".mcp" for name in files)
        else:
            raise ValueError(f"Data path is not a regular file/directory: {source}")
        inventory = {"path": str(source), "mcp_files": count, "complete": not errors, "errors": errors,
                     "scope": "extension-only source inventory; NOT pending jobs"}
    sampling_started = utc_now()
    lock_before = optional_json(output / ".pipeline.lock")
    sessions_before = scan_sessions(output)
    launch = optional_json(output / "production_launch.json")

    def target_pids(lock, sessions):
        values = [lock.get("data", {}).get("pid")] if lock.get("data") else []
        values.extend(entry["data"].get("pid") for entry in sessions["entries"] if entry.get("data"))
        return [pid for pid in values if positive_int(pid)]

    first = snapshot_provider(target_pids(lock_before, sessions_before))
    sleeper(sample_seconds)
    sessions_after = scan_sessions(output)
    second = snapshot_provider(target_pids(lock_before, sessions_after))
    lock_after = optional_json(output / ".pipeline.lock")
    owner_before = identify_lock(lock_before, first, output)
    owner_after = identify_lock(lock_after, second, output)
    stable_lock = (lock_before["present"] == lock_after["present"] and not lock_before["error"]
                   and not lock_after["error"] and lock_before["data"] == lock_after["data"])
    before = identify_sessions(sessions_before, first, owner_before)
    after = identify_sessions(sessions_after, second, owner_after)
    old = {(row.get("workspace"), row.get("native_pid")): row for row in before}
    active = []
    for row in after:
        if row["ownership"] == "OWNED":
            prior = old.get((row["workspace"], row["native_pid"]), {})
            row.update(activity(prior.get("process") if prior.get("ownership") == "OWNED" else None, row.get("process")))
            active.append(row)
    owner_activity = activity(owner_before.get("process"), owner_after.get("process")) if stable_lock else activity(None, None)
    reports, jobs = report_inventory(output, since)
    summary = final_summary(optional_json(output / "batch_report.json"), jobs)
    excerpt = log_excerpt(log)
    warnings = []
    for snapshot in (first, second):
        if snapshot.get("status") != "ok":
            warnings.append(snapshot.get("error", "Process query unavailable"))
        if snapshot.get("memory_error"):
            warnings.append("System memory query unavailable: " + snapshot["memory_error"])
    warnings.extend(sessions_before["errors"] + sessions_after["errors"])
    if excerpt["error"]:
        warnings.append("Log tail unavailable: " + excerpt["error"])
    if reports["status_counts"]["FAILED"]:
        warnings.append("FAILED reports are current stored records, not evidence of new failures or this run's crashes")
    if reports["unknown_count"]:
        warnings.append("Report inventory contains unknown/in-flight/invalid or missing-artifact evidence")
    if any(row["ownership"] == "UNKNOWN" for row in after):
        warnings.append("Some ready metadata cannot be attributed to a live owned native session")
    if not stable_lock:
        warnings.append("Lock changed during sampling; this is a transition, not proof of failure")
    launch_data = launch.get("data") or {}
    concurrency = None
    if (owner_after["identity"] == "OWNED" and launch_data.get("run_id") == owner_after["run_id"]
            and launch_data.get("scheduler_pid") == owner_after["pid"] and positive_int(launch_data.get("max_workers"))):
        concurrency = {"max_workers": launch_data["max_workers"], "source": launch["path"],
                       "scope": "matching launch metadata, not a measured queue length"}
    elif launch["present"]:
        warnings.append("Launch metadata is unreadable, stale or not corroborated by the current lock; concurrency unknown")
    native_snapshots = [[row for row in snapshot.get("processes", []) if row.get("name", "").casefold() in NATIVE_NAMES]
                        for snapshot in (first, second)]
    live_scheduler_candidates = []
    for snapshot in (first, second):
        candidates = []
        for row in snapshot.get("processes", []):
            tokens = command_tokens(row.get("command_line"))
            target = option_value(tokens, "-OutputDir") or option_value(tokens, "--output-dir")
            if (any(ntpath.basename(token).casefold() in SCHEDULER_SCRIPTS for token in tokens)
                    and (target is None or normalize_path(target) == normalize_path(output))):
                candidates.append(row["pid"])
        live_scheduler_candidates.append(candidates)
    known_queries = first.get("status") == second.get("status") == "ok"
    evidence_unknown = bool(reports["unknown_count"] or sessions_before["errors"] or sessions_after["errors"]
                            or (inventory and not inventory["complete"]))
    overall, code = "UNKNOWN", 2
    reason = "Insufficient evidence to establish active progress or completed batch; short idle samples do not prove a deadlock"
    if stable_lock and known_queries and owner_after["identity"] in ("DEAD", "MISMATCH"):
        overall, code, reason = "CONFIRMED_ERROR", 1, owner_after["reason"]
    elif (stable_lock and known_queries and not evidence_unknown and owner_before["identity"] == owner_after["identity"] == "OWNED"
          and same_process(owner_before.get("process"), owner_after.get("process"))
          and any(row["observed_activity"] for row in active)):
        overall, code = "RUNNING", 0
        reason = "Verified owned native session exhibited CPU-time or I/O activity during this sample (not a throughput guarantee)"
    elif (stable_lock and known_queries and not evidence_unknown
          and owner_before["identity"] == owner_after["identity"] == "ABSENT" and summary["complete"]
          and not any(native_snapshots) and not any(live_scheduler_candidates)
          and not any(row["ownership"] == "UNKNOWN" for row in after)):
        overall, code = "COMPLETED", 0
        reason = "Consistent final batch summary, absent lock and no native Shogun processes in both scoped samples"
    limits = [
        "Read-only observations are not an atomic scheduler snapshot; disappearing files and process transitions may require manual confirmation.",
        "Report counts include older runs; report mtime/--since indicates newer reports, never newly occurring failure events.",
        "No exact completed-this-run, pending count, ETA or percentage is inferred from report/work directory counts.",
        "Long duration, low GPU use, quiet logs and zero short-sample CPU are not deadlock evidence.",
        "Artifacts are only stat'ed for regular nonempty files, not opened, hashed, loaded or quality-verified; relative paths are report-relative.",
        "Only post_processing quality is accepted; human/prop/roster review categories may overlap and do not denote scheduler failures.",
        "Process ownership requires visible creation times, command identity and a valid scheduler/worker/native parent chain; access denial stays unknown.",
        "No SDK/debugger attachment, VDF access, process control, production retries or snapshot files are performed.",
        "Native process absence is a scoped process observation, not a probe of SDK connections; GPU telemetry is not used for health.",
    ]
    return {"schema_version": 1, "output_dir": str(output), "sampling": {
                "started_at": sampling_started, "first_sample_at": first.get("sampled_at"),
                "second_sample_at": second.get("sampled_at"), "finished_at": utc_now(),
                "requested_seconds": sample_seconds},
            "process_queries": [{key: snapshot.get(key) for key in ("status", "sampled_at", "error", "memory", "memory_error")}
                                for snapshot in (first, second)],
            "lock": {"stable_during_sample": stable_lock, "first": owner_before, "second": owner_after,
                     "activity": owner_activity},
            "sessions": {"owned_active_count": len(active) if second.get("status") == "ok" else None, "owned_active": active,
                         "stale_count": sum(row["ownership"] == "STALE" for row in after),
                         "unknown_count": sum(row["ownership"] == "UNKNOWN" for row in after),
                         "observations": after,
                         "native_process_counts": [len(rows) if snapshot.get("status") == "ok" else None
                                                   for rows, snapshot in zip(native_snapshots, (first, second))],
                         "live_scheduler_candidate_pids": live_scheduler_candidates},
            "concurrency": concurrency, "source_inventory": inventory, "reports": reports,
            "final_summary": summary, "log": excerpt, "warnings": list(dict.fromkeys(warnings)),
            "limits": limits, "overall": overall, "reason": reason, "exit_code": code}


def human_readable(result):
    if result.get("overall") == "ENTRY_ERROR":
        return "ENTRY_ERROR: " + result["reason"]
    lines = [f"{result['overall']}: {result['reason']}", f"Output: {result['output_dir']}"]
    sample = result["sampling"]
    lines.append(f"Samples: {sample['first_sample_at']} -> {sample['second_sample_at']} (requested {sample['requested_seconds']}s)")
    owner = result["lock"]["second"]
    lines.append(f"Lock: {owner['identity']} pid={owner['pid']} run_id={owner['run_id']}; {owner['reason']}")
    lines.append("Scheduler activity: " + json.dumps(result["lock"]["activity"], ensure_ascii=False))
    sessions = result["sessions"]
    lines.append(f"Sessions: owned active={sessions['owned_active_count']}, stale metadata={sessions['stale_count']}, unknown={sessions['unknown_count']}")
    for row in sessions["owned_active"]:
        lines.append(f"  native={row['native_pid']} worker={row['worker_pid']} port={row['port']} age={row['age_seconds']}s "
                     f"CPU delta={row['cpu_seconds_delta']}s read={row['read_bytes_delta']} write={row['write_bytes_delta']} bytes; {row['workspace']}")
    lines.append("Concurrency: " + json.dumps(result["concurrency"], ensure_ascii=False))
    lines.append("Source inventory: " + json.dumps(result["source_inventory"], ensure_ascii=False))
    reports = result["reports"]
    lines.append("Reports (current stock, NOT this run's progress): " + json.dumps(reports["status_counts"]))
    lines.append("Artifact/quality-evidenced counts: " + json.dumps(reports["verified_counts"]) + f"; unknown={reports['unknown_count']}")
    lines.append("Stored failure classes (NOT new failures): " + json.dumps(reports["failure_classes"]))
    lines.append("Review take categories (overlapping): " + json.dumps(reports["review_categories"]))
    lines.append("Human final quality: " + json.dumps(reports["human_quality"]) + "; subject exports: " + json.dumps(reports["subject_export_quality"]))
    if reports["since"]:
        lines.append(f"Reports modified since {reports['since']} (NOT new failure events): " + json.dumps(reports["newer_report_counts"]))
    lines.append(f"Missing referenced artifact/role count: {reports['missing_artifact_count']}")
    for issue in reports["artifact_issues"]:
        lines.append(f"  {issue['state']}: {issue['job_id']} {issue['role']} {issue['path']}: {issue['reason']}")
    for issue in reports["issues"]:
        lines.append("  Report unknown: " + issue)
    for failure in reports["failures"]:
        lines.append(f"  Stored FAILED {failure['job_id']} [{failure['classification']}]: {failure['error']}")
    lines.append("Final summary: " + str(result["final_summary"]["reason"]))
    lines.append(f"Recent log event excerpt: {result['log']['path']} (last <= {TAIL_BYTES} bytes; no event-time inference)")
    lines.extend("  " + line for line in result["log"]["recent_events"])
    lines.extend("Warning: " + warning for warning in result["warnings"])
    lines.extend("Limit: " + limit for limit in result["limits"])
    return "\n".join(lines)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--sample-seconds", type=float, default=3)
    result.add_argument("--log", help="Optional explicit scheduler log; otherwise only an existing output/production.log is used")
    result.add_argument("--data-path", help="Explicit read-only MCP inventory; never used to infer pending count")
    result.add_argument("--since", help="Timezone-aware ISO timestamp; labels newer reports, NOT new failure events")
    result.add_argument("--json", action="store_true", help="Write the observation as JSON to stdout")
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        options = parser().parse_args(arguments)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1
    try:
        since = parse_time(options.since) if options.since else None
        if options.since and since is None:
            raise ValueError("--since requires an ISO timestamp with a UTC offset or Z")
        result = collect_health(options.output_dir, options.sample_seconds, options.log, options.data_path, since)
    except (OSError, ValueError) as exc:
        result = {"schema_version": 1, "overall": "ENTRY_ERROR", "reason": str(exc), "exit_code": 1}
    except KeyboardInterrupt:
        result = {"schema_version": 1, "overall": "UNKNOWN", "reason": "Health sampling interrupted; production was not signalled", "exit_code": 2}
    if options.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif "sampling" not in result:
        print(f"{result['overall']}: {result['reason']}")
    else:
        print(human_readable(result))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
