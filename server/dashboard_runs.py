"""Explicit local run discovery and one bounded, demand-selected sampler."""
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import threading
import time

from dashboard_data import DashboardData, SnapshotCache, health, layout, optional_json_file, scoped_path, validate_opened_file
from pipeline_run_registry import read_run_record

KINDS = {"processing", "review_repair", "human_repair", "quality_recheck", "human_target"}
DEFAULT_REGISTRY = Path("H:/MocapStudio/data/runs")


def output_id(output):
    return hashlib.sha256(str(Path(output).resolve()).casefold().encode()).hexdigest()[:16]


def validate_record(value, filename):
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported catalog schema")
    required = {"schema_version", "id", "run_id", "kind", "output_dir", "source_output", "data_path",
                "started_at", "scheduler_pid", "label", "planned_total"}
    if required - value.keys():
        raise ValueError("missing catalog schema fields")
    identifier = value.get("id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{16}", identifier) or filename != identifier + ".json":
        raise ValueError("invalid catalog id/filename")
    output = value.get("output_dir")
    if not isinstance(output, str) or not output or not Path(output).is_absolute() or "\x00" in output:
        raise ValueError("output_dir must be absolute")
    root = Path(output).resolve()
    if output_id(root) != identifier:
        raise ValueError("catalog id/output root mismatch")
    if value.get("kind") not in KINDS:
        raise ValueError("invalid run kind")
    for field in ("run_id", "label"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError("invalid " + field)
    try:
        started = datetime.fromisoformat(str(value.get("started_at")).replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("started_at must be UTC ISO time") from None
    if started.tzinfo is None or started.utcoffset().total_seconds() != 0:
        raise ValueError("started_at must be UTC ISO time")
    if not health.positive_int(value.get("scheduler_pid")):
        raise ValueError("invalid scheduler_pid")
    total = value.get("planned_total")
    if total is not None and (type(total) is not int or total < 0):
        raise ValueError("invalid planned_total")
    result = dict(value, output_dir=str(root))
    for field in ("source_output", "data_path", "input_path"):
        path = value.get(field)
        if path is not None:
            if not isinstance(path, str) or not path or not Path(path).is_absolute() or "\x00" in path:
                raise ValueError("invalid absolute " + field)
            result[field] = str(Path(path).resolve())
    if value["kind"] == "human_target" and (not isinstance(value.get("subject"), str) or not value["subject"].strip()):
        raise ValueError("human_target requires subject")
    recheck = value.get("quality_recheck")
    if recheck is not None:
        if (not isinstance(recheck, dict) or not health.positive_int(recheck.get("scheduler_pid"))
                or health.parse_time(recheck.get("started_at")) is None
                or not isinstance(recheck.get("run_id"), str)):
            raise ValueError("invalid independent recheck metadata")
    return result


class RunCatalog:
    def __init__(self, primary, registry=None, *, snapshot_provider=None, clock=time.monotonic):
        self.primary = primary
        self.registry = Path(registry or os.environ.get("MOCAP_RUN_REGISTRY") or DEFAULT_REGISTRY).expanduser().resolve()
        self.snapshot_provider = snapshot_provider or health.process_snapshot
        self.clock = clock
        self._process_at = -float("inf")
        self._processes = {}
        self._lock = threading.RLock()
        self.records = {}

    def refresh(self):
        with self._lock:
            records, warnings = {}, []
            try:
                paths = sorted(self.registry.glob("*.json")) if self.registry.is_dir() else []
                for path in paths:
                    try:
                        candidate = scoped_path(self.registry, path)
                        value = read_run_record(candidate, validate_opened=lambda stream:
                            validate_opened_file(self.registry, candidate, stream, check_current=False))
                        record = validate_record(value, path.name)
                        if record["id"] in records or any(item["output_dir"].casefold() == record["output_dir"].casefold() for item in records.values()):
                            raise ValueError("duplicate catalog id/output root")
                        records[record["id"]] = record
                    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as exc:
                        warnings.append(f"Rejected catalog record {path.name}: {exc}")
            except OSError as exc:
                warnings.append(f"Run registry unavailable: {exc}")
            identifier = output_id(self.primary.output)
            try:
                # Only configured controls, never arbitrary raw/generated trees.
                controls = layout.source_runs(self.primary.output)
                for control in controls:
                    control = scoped_path(self.primary.output, control)
                    control_id = output_id(control)
                    if control_id not in records:
                        records[control_id] = {"schema_version": 1, "id": control_id, "run_id": None,
                            "kind": "processing", "output_dir": str(control),
                            "data_path": str(self.primary.data_path) if self.primary.data_path else None,
                            "source_output": str(self.primary.output) if control != self.primary.output else None,
                            "label": control.name, "started_at": None, "planned_total": None}
            except (OSError, ValueError, RuntimeError) as exc:
                warnings.append(f"Configured source controls unavailable: {exc}")
            # A malformed child must not hide the explicitly selected primary.
            if identifier not in records:
                records[identifier] = {"schema_version": 1, "id": identifier, "run_id": None,
                    "kind": "processing", "output_dir": str(self.primary.output),
                    "data_path": str(self.primary.data_path) if self.primary.data_path else None,
                    "source_output": None, "label": self.primary.output.name, "started_at": None, "planned_total": None}
            if self.clock() - self._process_at >= self.primary.refresh_seconds:
                pids = [item.get("scheduler_pid") for item in records.values() if health.positive_int(item.get("scheduler_pid"))]
                pids.extend(item["quality_recheck"]["scheduler_pid"] for item in records.values() if item.get("quality_recheck"))
                for item in records.values():
                    root = Path(item["output_dir"])
                    lock = optional_json_file(root / ".pipeline.lock", root)
                    pid = (lock.get("data") or {}).get("pid")
                    if health.positive_int(pid):
                        pids.append(pid)
                self._processes = self.snapshot_provider(pids)
                self._process_at = self.clock()
            runs = []
            for item in records.values():
                row = {key: item.get(key) for key in ("id", "label", "kind", "output_dir", "run_id", "started_at", "planned_total")}
                root = Path(item["output_dir"])
                row["available"] = root.is_dir()
                lock = optional_json_file(root / ".pipeline.lock", root)
                identity = health.identify_lock(lock, self._processes, root)
                if identity["identity"] == "OWNED":
                    row["active"] = True
                else:
                    # Recheck/one-target tools do not acquire the pipeline lock.
                    recheck = item.get("quality_recheck") or {}
                    scheduler_pid = recheck.get("scheduler_pid") or item.get("scheduler_pid")
                    process = health.process_map(self._processes).get(scheduler_pid)
                    tokens = health.command_tokens((process or {}).get("command_line"))
                    target = health.option_value(tokens, "--output-dir") or health.option_value(tokens, "-OutputDir")
                    script = "step2_repair_single_human.py" if item["kind"] == "human_target" else "step4_recheck_quality.py" if item["kind"] == "quality_recheck" or recheck else None
                    created = health.parse_time((process or {}).get("created_at"))
                    started = health.parse_time(recheck.get("started_at") or item.get("started_at"))
                    if (script and process and process.get("name", "").casefold() in health.PYTHON_NAMES and created and started
                            and created <= started and target and Path(target).is_absolute()
                            and health.normalize_path(target) == health.normalize_path(root)
                            and any(os.path.basename(token.replace("\\", "/")).casefold() == script for token in tokens)):
                        row["active"] = True
                runs.append(row)
            runs.sort(key=lambda row: (health.parse_time(row.get("started_at")) or datetime.min.replace(tzinfo=timezone.utc),
                                      row["id"]), reverse=True)
            self.records = records
            default = next((row["id"] for row in runs if row.get("active") and row["available"]), identifier)
            return {"schema_version": 1, "default_run": default, "runs": runs, "warnings": warnings}


class MultiRunCache:
    """Only the requested run is polled; at most four snapshots are retained."""
    def __init__(self, primary, registry=None, *, catalog=None, max_runs=4):
        self.catalog = catalog or RunCatalog(primary, registry)
        self.primary_id = output_id(primary.output)
        self.caches = OrderedDict([(self.primary_id, SnapshotCache(primary))])
        self.selected = self.primary_id
        self.max_runs = max(2, max_runs)
        self._condition = threading.Condition(threading.RLock())
        self._thread = None
        self._stopped = False
        self._requested = True
        self.catalog.refresh()
        primary.run = dict(self.catalog.records[self.primary_id])

    def runs(self):
        return self.catalog.refresh()

    def for_run(self, identifier=None):
        if identifier is None:
            catalog = self.catalog.refresh()
            identifier = catalog.get("default_run") or self.primary_id
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{16}", identifier):
            raise KeyError(identifier)
        self.catalog.refresh()
        with self._condition:
            record = self.catalog.records.get(identifier)
            if record is None:
                raise KeyError(identifier)
            cache = self.caches.get(identifier)
            if cache is None:
                primary = self.caches[self.primary_id].data
                data = DashboardData(record["output_dir"], record.get("data_path"), primary.refresh_seconds,
                    plan_dir=primary.plan_root, run=record)
                cache = SnapshotCache(data)
                self.caches[identifier] = cache
            previous = cache.data.run
            identity_changed = any(previous.get(field) != record.get(field) for field in ("run_id", "kind", "started_at"))
            if identity_changed:
                with cache._condition:
                    cache._snapshot = None
                self._requested = True
            cache.data.run = dict(record)
            self.caches.move_to_end(identifier)
            if self.selected != identifier or (cache.snapshot() is None and not cache._refreshing):
                self._requested = True
            self.selected = identifier
            while len(self.caches) > self.max_runs:
                victim = next(key for key in self.caches if key not in (self.primary_id, identifier))
                del self.caches[victim]
            self._condition.notify_all()
            return cache

    def start(self):
        with self._condition:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="dashboard-run-sampler", daemon=True)
                self._thread.start()

    def close(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=35)

    def _run(self):
        deadline = 0
        while True:
            with self._condition:
                while not self._stopped and not self._requested and time.monotonic() < deadline:
                    self._condition.wait(deadline - time.monotonic())
                if self._stopped:
                    return
                cache = self.caches[self.selected]
                self._requested = False
            with cache._condition:
                cache._refreshing = True
                cache._requested = False
            started = time.monotonic()
            run_identity = tuple(cache.data.run.get(field) for field in ("run_id", "kind", "started_at"))
            try:
                snapshot = cache.data.collect()
            except Exception as exc:
                with cache._condition:
                    cache._last_error = f"{type(exc).__name__}: {exc}"
            else:
                with self._condition:
                    current_identity = tuple(cache.data.run.get(field) for field in ("run_id", "kind", "started_at"))
                    if run_identity == current_identity:
                        with cache._condition:
                            cache._snapshot, cache._last_error = snapshot, None
                    else:
                        self._requested = True
            with cache._condition:
                cache._refreshing = False
            deadline = max(started + cache.data.refresh_seconds, time.monotonic())

    def overview(self, identifier=None, refresh=False):
        cache = self.for_run(identifier)
        if refresh:
            with self._condition:
                self._requested = True
                self._condition.notify_all()
        value = cache.overview()
        value["run"] = dict(cache.data.run)
        if not value["run"].get("run_id") and value.get("health", {}).get("run_id"):
            value["run"]["run_id"] = value["health"]["run_id"]
        return value
