"""Loopback-only durable queue for isolated web_worker.py processes.

Reading configuration, datasets or results never imports the native SDK. The
manager owns metadata.json; only workers write state.json. Lost live sessions
are interrupted, not resumed. Run with --help for local path configuration.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, quote, unquote, urlsplit
import uuid

from pipeline_run_registry import open_shared_read

TERMINAL = frozenset({"SUCCESS", "NEEDS_REVIEW", "FAILED", "CANCELLED", "INTERRUPTED"})
CAPTURE_DUMP_RE = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.(mcp|vdf|fbx)", re.I)


def is_capture_dump(name):
    """Shogun Capture raw dumps named like `longyufeng_2026-11-04_13-58-29.mcp`
    are not take sequences: never discover, search, or enqueue them."""
    return bool(CAPTURE_DUMP_RE.fullmatch(str(name)))
ACTIVE = frozenset({"STARTING", "WAITING", "RUNNING", "PAUSE_REQUESTED"})
EXCLUDED = frozenset({".git", "node_modules", "processed", "_processing", ".work", "__pycache__"})
# Explicit canonical outputs are valid fresh inputs; discovery still skips them.
INPUT_EXCLUDED = EXCLUDED - {"processed", "_processing"}
DEFAULT_OPTIONS = {
    "filter_markers": True, "expected_subjects": None, "expected_props": None,
    "dataset_mode": True, "attach_meshes": False, "lightweight": True,
    "drop_unused_labeling": True, "prop_jitter_max_correction": 0.0, "prop_tolerance": 2.0,
    "denoise_static": False,
}
STAGES = [
    {"id": "parse", "label": "Parse", "description": "Read-only roster parse: subjects/rigid bodies/structure from the original file (no SDK)"},
    {"id": "preflight", "label": "Preflight", "description": "Inspect actual input and processing requirements"},
    {"id": "load", "label": "Load", "description": "Load an isolated native session"},
    {"id": "processing", "label": "Native processing", "description": "Observe actual native stages, repeated per subject"},
    {"id": "attach_meshes", "label": "Attach meshes", "description": "Attach configured real assets with Blender"},
    {"id": "lightweight", "label": "Display trim", "description": "Lossless display-oriented FBX trimming"},
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def natural_key(text):
    """Natural order: group2 before group10 (digit runs compare as integers)."""
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", str(text))]


def inside(path, root):
    return path == root or root in path.parents


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path, default=None):
    try:
        with open_shared_read(path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return default


class APIError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class StateLock:
    """OS-held lock; stale lock files never require deleting or PID guessing."""
    def __init__(self, path):
        self.stream = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0, os.SEEK_END)
                if not self.stream.tell():
                    self.stream.write(b"\0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("Another manager owns this state directory") from None

    def close(self):
        self.stream.close()


def discover_executable(explicit, name, patterns):
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            raise ValueError(f"Executable does not exist: {candidate}")
        return str(candidate)
    candidates = []
    command = shutil.which(name)
    if command:
        candidates.append(Path(command))
    for root, pattern in patterns:
        if root.is_dir():
            candidates.extend(root.glob(pattern))
    candidates = [path.resolve() for path in candidates if path.is_file()]
    def version(path):
        return tuple(int(n) for n in re.findall(r"\d+", str(path.parent)))
    return str(max(candidates, key=version)) if candidates else None


class JobManager:
    def __init__(self, state_dir, data_roots, *, executable=None, blender=None,
                 worker_script=None, max_workers=1, cancel_grace=3.0, jobs_dir=None):
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state_lock = StateLock(self.state_dir / "manager.lock")
        self.jobs_dir = Path(jobs_dir).expanduser().resolve() if jobs_dir else self.state_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.data_roots = list(dict.fromkeys(Path(root).expanduser().resolve() for root in data_roots))
        if not self.data_roots:
            self._state_lock.close()
            raise ValueError("At least one data root is required")
        self.executable = executable
        self.blender = blender
        self.worker_script = Path(worker_script or Path(__file__).with_name("web_worker.py")).resolve()
        self.cancel_grace = cancel_grace
        self.csrf_token = secrets.token_urlsafe(32)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self._owned = {}
        self._jobs = {}
        self._inspect_scans = {}
        settings = read_json(self.state_dir / "config.json", {})
        saved_limit = settings.get("max_workers", max_workers)
        self.max_workers = saved_limit if type(saved_limit) is int and 1 <= saved_limit <= 8 else max_workers
        for directory in sorted(self.jobs_dir.iterdir()):
            if not directory.is_dir() or directory.is_symlink() or not re.fullmatch(r"[a-f0-9]{32}", directory.name):
                continue
            metadata = read_json(directory / "metadata.json")
            if not isinstance(metadata, dict) or metadata.get("id") != directory.name:
                continue
            self._jobs[directory.name] = metadata
            if metadata.get("state") not in TERMINAL | {"QUEUED"}:
                self._set_status(metadata, "INTERRUPTED", "Manager restarted; the previous live session cannot be resumed. Retry creates a new run.")

    def _directory(self, metadata):
        return self.jobs_dir / metadata["id"]

    def _save(self, metadata):
        atomic_json(self._directory(metadata) / "metadata.json", metadata)

    def _set_status(self, metadata, state, error=None):
        metadata.update(state=state, error=error, updated_at=utc_now())
        metadata.pop("pending", None)
        self._save(metadata)

    def _worker_state(self, metadata):
        value = read_json(self._directory(metadata) / "state.json", {})
        return value if isinstance(value, dict) else {}

    def _get(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id) or job_id not in self._jobs:
            raise APIError(404, "Unknown job")
        return self._jobs[job_id]

    def _allowed_input(self, value, *, directory=False):
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise APIError(400, "An absolute input path is required")
        try:
            candidate = Path(value).resolve(strict=True)
        except (OSError, ValueError):
            raise APIError(404, "Input path does not exist") from None
        if not any(inside(candidate, root) for root in self.data_roots) or inside(candidate, self.state_dir):
            raise APIError(403, "Path is outside the configured dataset roots")
        root = next(root for root in self.data_roots if inside(candidate, root))
        if any(part.casefold() in INPUT_EXCLUDED for part in candidate.relative_to(root).parts):
            raise APIError(403, "Excluded directories cannot be processed")
        if directory:
            if not candidate.is_dir():
                raise APIError(400, "Browse path must be a directory")
        elif not candidate.is_file() or candidate.suffix.casefold() not in {".mcp", ".vdf", ".fbx"}:
            raise APIError(400, "Select an existing MCP, VDF or FBX file")
        elif is_capture_dump(candidate.name):
            raise APIError(403, "Capture-time raw dumps are not take sequences and cannot be processed")
        return candidate

    def config(self):
        with self._lock:
            return {"data_roots": list(map(str, self.data_roots)), "output_root": str(self.state_dir),
                    "default_browse_path": str(self.data_roots[0]), "shogun_available": bool(self.executable and Path(self.executable).is_file()),
                    "shogun_path": self.executable, "blender_path": self.blender,
                    "max_workers": self.max_workers, "max_workers_limit": 8,
                    "viewer_base": "/pipeline-viewer/fbx_viewer.html", "csrf_token": self.csrf_token, "stages": STAGES}

    def configure(self, body):
        if set(body) != {"max_workers"} or type(body["max_workers"]) is not int or not 1 <= body["max_workers"] <= 8:
            raise APIError(400, "max_workers must be an integer from 1 to 8")
        with self._lock:
            self.max_workers = body["max_workers"]
            atomic_json(self.state_dir / "config.json", {"max_workers": self.max_workers})
            return self.config()

    def browse(self, value=None, recursive=False):
        base = self._allowed_input(value or str(self.data_roots[0]), directory=True)
        directories, sequences = [], []
        pending, visited = [base], set()
        scanned = 0
        truncated = False
        deadline = time.monotonic() + 3.0
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        scanned += 1
                        if scanned > 20000 or len(sequences) >= 2000 or len(directories) >= 2000 or time.monotonic() > deadline:
                            truncated = True
                            break
                        if entry.name.casefold() in EXCLUDED or entry.is_symlink():
                            continue
                        try:
                            path = Path(entry.path).resolve(strict=True)
                            if inside(path, self.state_dir) or inside(path, self.jobs_dir) or not inside(path, base):
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                if current == base:
                                    directories.append({"name": entry.name, "path": str(path)})
                                if recursive:
                                    pending.append(path)
                            elif entry.is_file(follow_symlinks=False) and path.suffix.casefold() in {".mcp", ".vdf", ".fbx"} and not is_capture_dump(path.name):
                                stat = entry.stat(follow_symlinks=False)
                                sequences.append({"id": hashlib.sha256(str(path).encode()).hexdigest()[:24],
                                                  "name": path.name, "path": str(path), "kind": path.suffix[1:].lower(),
                                                  "size_bytes": stat.st_size, "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                                                  "x2d_exists": path.with_suffix(".x2d").is_file() or path.with_suffix(".X2D").is_file()})
                        except OSError:
                            continue
            except OSError as exc:
                if current == base:
                    raise APIError(403, f"Cannot read directory: {exc}") from None
            if truncated:
                break
        parent = base.parent if base.parent != base and any(inside(base.parent, root) for root in self.data_roots) else None
        return {"path": str(base), "parent": str(parent) if parent else None, "roots": list(map(str, self.data_roots)),
                "directories": sorted(directories, key=lambda item: natural_key(item["name"])),
                "sequences": sorted(sequences, key=lambda item: natural_key(item["path"])), "truncated": truncated}

    def tree(self, value=None):
        """One Explorer level: authorized directories plus selectable sequences."""
        base = self._allowed_input(value or str(self.data_roots[0]), directory=True)
        entries, truncated = [], False
        try:
            with os.scandir(base) as scan:
                for entry in scan:
                    if len(entries) >= 500:
                        truncated = True
                        break
                    if entry.name.casefold() in EXCLUDED or entry.is_symlink():
                        continue
                    try:
                        path = Path(entry.path).resolve(strict=True)
                        if inside(path, self.state_dir) or inside(path, self.jobs_dir) or not inside(path, base):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            entries.append({"name": entry.name, "path": str(path), "kind": "dir"})
                        elif entry.is_file(follow_symlinks=False) and path.suffix.casefold() in {".mcp", ".vdf", ".fbx"} and not is_capture_dump(path.name):
                            stat = entry.stat(follow_symlinks=False)
                            entries.append({"name": entry.name, "path": str(path), "kind": path.suffix[1:].lower(),
                                            "size_bytes": stat.st_size,
                                            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                                            "x2d_exists": path.with_suffix(".x2d").is_file() or path.with_suffix(".X2D").is_file(),
                                            "deletable": self.deletable_output(path)})
                    except OSError:
                        continue
        except OSError as exc:
            raise APIError(403, f"Cannot read directory: {exc}") from None
        entries.sort(key=lambda item: (item["kind"] != "dir", natural_key(item["name"])))
        parent = base.parent if base.parent != base and any(inside(base.parent, root) for root in self.data_roots) else None
        return {"path": str(base), "parent": str(parent) if parent else None, "roots": list(map(str, self.data_roots)),
                "entries": entries, "truncated": truncated}

    def find(self, value, limit=50):
        """Bounded breadth-first substring search for the @ picker; honest caps."""
        if not isinstance(value, str) or not (query := value.strip()) or len(query) > 120:
            raise APIError(400, "A search query of 1-120 characters is required")
        needle = query.casefold()
        matches, pending, visited = [], [(root, 0) for root in self.data_roots if root.is_dir()], set()
        scanned, truncated = 0, False
        deadline = time.monotonic() + 4.0
        while pending:
            current, depth = pending.pop(0)
            if current in visited:
                continue
            visited.add(current)
            try:
                with os.scandir(current) as scan:
                    for entry in scan:
                        scanned += 1
                        if scanned > 20000 or time.monotonic() > deadline:
                            truncated = True
                            break
                        if entry.name.casefold() in EXCLUDED or entry.is_symlink():
                            continue
                        try:
                            path = Path(entry.path).resolve(strict=True)
                            if inside(path, self.state_dir) or inside(path, self.jobs_dir):
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                if depth < 8:
                                    # Directories whose own name matches are explored first so
                                    # deep dataset trees are reached before the scan budget dies.
                                    if needle in entry.name.casefold():
                                        pending.insert(0, (path, depth + 1))
                                    else:
                                        pending.append((path, depth + 1))
                            elif entry.is_file(follow_symlinks=False) and path.suffix.casefold() in {".mcp", ".vdf", ".fbx"} and not is_capture_dump(path.name):
                                name = path.name.casefold()
                                if needle in name or needle in str(path).casefold():
                                    stat = entry.stat(follow_symlinks=False)
                                    matches.append({"score": 0 if name.startswith(needle) else (1 if needle in name else 2),
                                                    "name": path.name, "path": str(path), "kind": path.suffix[1:].lower(),
                                                    "size_bytes": stat.st_size,
                                                    "x2d_exists": path.with_suffix(".x2d").is_file() or path.with_suffix(".X2D").is_file()})
                        except OSError:
                            continue
            except OSError:
                continue
            if truncated:
                break
        matches.sort(key=lambda item: (item["score"], natural_key(item["path"])))
        for item in matches:
            item.pop("score")
        return {"query": query, "matches": matches[:limit], "scanned": scanned, "truncated": truncated or len(matches) > limit}

    def inspect(self, value):
        """Roster analysis for one sequence: run reports and job states are read
        live on every call; only the expensive file-derived scans (FBX structure,
        MCP record scan) are cached against the file's size+mtime identity."""
        try:
            import sequence_inspect
        except ImportError:
            from . import sequence_inspect
        path = self._allowed_input(value)
        stat = path.stat()
        scan_key = (str(path), stat.st_size, stat.st_mtime_ns)
        kind = path.suffix[1:].lower()
        with self._lock:
            matched = [item for item in self._jobs.values() if item.get("input_path") == str(path)]
        matched.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        payload = {"path": str(path), "kind": kind, "size_bytes": stat.st_size,
                   "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                   "runs": [{"job_id": item["id"], "state": item.get("state"), "mode": item.get("mode"),
                             "created_at": item.get("created_at"), "updated_at": item.get("updated_at")}
                            for item in matched[:8]],
                   "x2d_exists": path.with_suffix(".x2d").is_file() or path.with_suffix(".X2D").is_file(),
                   "subjects": [], "objects": [], "stats": {}, "animation": [], "notes": [],
                   "frame_rate_hz": None, "dataset_family": sequence_inspect.dataset_family(path)}
        if payload["dataset_family"] in ("MP", "BODY"):
            payload["notes"].append("MP / body 是人体专用数据:处理开始后会删除场景里所有道具刚体,只保留人体;下面列出的物体只是文件里的定义记录。")
        roster: dict[tuple[str, str], dict] = {}

        def merge(group: str, name: str, source: str, detail: dict) -> None:
            entry = roster.setdefault((group, name), {"name": name, "sources": []})
            if source not in entry["sources"]:
                entry["sources"].append(source)
            entry.update({k: v for k, v in detail.items() if k not in ("name", "sources")})

        def cached_scan():
            with self._lock:
                scan = self._inspect_scans.get(scan_key)
            if scan is not None:
                return scan
            if kind == "fbx":
                try:
                    scan = {"fbx": sequence_inspect.parse_fbx_roster(path)}
                except (ValueError, OSError) as exc:
                    scan = {"fbx": {"error": str(exc)}}
            else:
                try:
                    scan = {"mcp": sequence_inspect.scan_mcp_roster(path)}
                except OSError as exc:
                    scan = {"mcp_error": str(exc)}
            with self._lock:
                if len(self._inspect_scans) >= 48:
                    self._inspect_scans.pop(next(iter(self._inspect_scans)))
                self._inspect_scans[scan_key] = scan
            return scan

        report = None
        for item in matched:
            candidate = read_json(self._directory(item) / "report.json")
            if isinstance(candidate, dict):
                report, payload["report_job"] = candidate, item["id"]
                break
        # Wall-clock capture window (octopus timecode alignment): MCP carries
        # Live's open time + 135MHz generator ticks; other formats fall back to
        # the source file's own mtime minus duration.
        try:
            window = None
            if kind == "mcp":
                frame_count = ((report or {}).get("range") or {}).get("frame_count")
                window = sequence_inspect.scan_mcp_capture_window(path, frame_count)
            if window is None and path.is_file():
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                window = {"capture_end": mtime.isoformat(), "source": "file_mtime",
                          "note": "非 MCP 源:按文件保存时间推算,take 结束≈文件保存时刻"}
            if window:
                payload["capture_wall"] = window
        except OSError:
            pass
        if report is not None:
            summary = sequence_inspect.summarize_report(report) or {}
            payload["report"] = {key: summary[key] for key in
                                 ("status", "pipeline_status_pending", "partial", "quality", "source_quality", "review_reasons")
                                 if key in summary}
            for row in summary.get("subjects", []):
                name = str(row.get("name") or "?")
                # Roster rows tagged as props in the report belong to the objects group.
                group = "object" if row.get("rig_class") == "PROP_OR_NONHUMAN" else "subject"
                merge(group, name, "run_report", {"rig_class": row.get("rig_class"), "classification": row.get("classification"),
                                                  "report": row})
            for row in summary.get("props", []):
                name = str(row.get("name") or "?")
                merge("object", name, "run_report", {"report": row})
        elif matched:
            payload["notes"].append("该序列已有任务但尚无报告；请打开任务查看实时日志与阶段。")
        if any(item.get("state") in ACTIVE for item in matched):
            payload["notes"].append("该序列有任务正在运行；名单与报告随阶段推进持续更新。")

        if kind == "fbx":
            parsed = cached_scan()["fbx"]
            if "error" in parsed:
                payload["notes"].append(f"FBX 结构解析不可用：{parsed['error']}")
            else:
                for group, rows in (("subject", parsed["subjects"]), ("object", parsed["objects"])):
                    for row in rows:
                        merge(group, row["name"], "fbx_parse", {"bones": row["bones"], "markers": row["markers"],
                                                                "meshes": row["meshes"], "nodes": row["nodes"],
                                                                "mesh_names": row["mesh_names"], "fbx_path": row["path"]})
                payload["stats"] = parsed["stats"]
                payload["animation"] = parsed["animation"]
                payload["frame_rate_hz"] = parsed["frame_rate_hz"]
                if parsed["unlisted_branches"]:
                    payload["stats"]["unlisted_branches"] = [
                        {"name": row["name"], "kind": row["kind"], "nodes": row["nodes"]}
                        for row in parsed["unlisted_branches"][:12]]
        elif kind == "mcp":
            scan = cached_scan()
            if "mcp" in scan:
                for row in scan["mcp"]["subject"]:
                    merge("subject", row["name"], "mcp_scan", {"occurrences": row["occurrences"]})
                for row in scan["mcp"]["rigid"]:
                    merge("object", row["name"], "mcp_scan", {"occurrences": row["occurrences"], "rigid": True})
                if scan["mcp"].get("clock"):
                    payload["capture_clock"] = dict(scan["mcp"]["clock"])
                if scan["mcp"]["rigid"] and report is None:
                    payload["notes"].append("名单是直接扫描 MCP 文件得到的:人物=文件里定义了骨架,物体=定义了标记点刚体(×n 是同一条记录出现的次数)。这只是「定义清单」,不代表每一帧都真的拍到;某个人物或道具在某段帧里有没有数据、质量好不好,以任务报告为准。")
                elif not scan["mcp"]["rigid"] and report is None:
                    payload["notes"].append("这个 MCP 文件里只扫到人物骨架,没有道具刚体定义;创建任务后,报告还会给出逐帧的实际有无与质量。")
            else:
                payload["notes"].append(f"MCP 扫描失败：{scan['mcp_error']}")
        elif kind == "vdf":
            if not matched:
                payload["notes"].append("VDF 文件本身不带名单;这个序列来自" + ("同名 X2D 离线重建。" if payload["x2d_exists"] else "已有处理产物。") + "人物与物体以创建任务后的报告为准。")

        payload["subjects"] = sorted((entry for (group, _), entry in roster.items() if group == "subject"),
                                     key=lambda entry: entry["name"].casefold())
        payload["objects"] = sorted((entry for (group, _), entry in roster.items() if group == "object"),
                                    key=lambda entry: entry["name"].casefold())
        return payload

    def delete(self, job_id):
        """Remove a finished job and its stored run artifacts. Never an active one."""
        with self._lock:
            metadata = self._get(job_id)
            if metadata.get("state") not in TERMINAL:
                raise APIError(409, "只能移除已结束任务(完成/需复核/失败/取消/中断);活动或排队中的任务请先取消")
            directory = Path(self._directory(metadata)).resolve()
            if directory.parent != self.jobs_dir or not directory.is_dir():
                raise APIError(403, "Job directory escapes the jobs store")
            shutil.rmtree(directory)
            self._jobs.pop(job_id, None)
            return {"removed": job_id}

    def dataset_progress(self):
        """Dataset processing progress: raw take counts grouped by family
        (BODY/MP/HOI) and actor/group, compared against existing
        03_exports deliverables (one usable FBX per take). Cached ~60s."""
        cached = getattr(self, "_progress_cache", None)
        if cached and time.time() - cached[0] < 60:
            return cached[1]
        payload = {"families": self._dataset_progress_scan()}
        self._progress_cache = (time.time(), payload)
        return payload

    def _dataset_progress_scan(self):
        families = {}
        for root in self.data_roots:
            for dirpath, dirnames, filenames in os.walk(root):
                relative = Path(dirpath).relative_to(root)
                dirnames[:] = [name for name in dirnames
                               if name.casefold() not in EXCLUDED
                               and not name.startswith(".")
                               and not re.match(r"^(02_processed|03_exports|jobs|.*_processing)$", name, re.I)]
                parts = list(relative.parts)
                if not parts or not parts[0].casefold().startswith("01_"):
                    continue
                for name in filenames:
                    if Path(name).suffix.casefold() not in (".mcp", ".vdf", ".fbx") or is_capture_dump(name):
                        continue
                    family, group, _ = self._classify_take(parts + [name])
                    entry = families.setdefault(family, {"total": 0, "done": 0, "groups": {}})
                    entry["total"] += 1
                    g = entry["groups"].setdefault(group, {"total": 0, "done": 0})
                    g["total"] += 1
                    tail = parts[1:] if parts[0].casefold().startswith("01_") else parts
                    stem = Path(name).stem
                    done = False
                    for out_name in (stem + ".fbx",):
                        for mirror in ("03_exports", "02_processed"):
                            if (root / mirror / Path(*tail) / out_name).is_file():
                                done = True
                                break
                        if done:
                            break
                    if done:
                        entry["done"] += 1
                        g["done"] += 1
        ordered = {}
        for family in sorted(families, key=lambda f: ({"BODY": 0, "MP": 1, "HOI": 2}.get(f, 3), f)):
            data = families[family]
            ordered[family] = {**data, "groups": [dict(name=name, **stats) for name, stats in
                                                  sorted(data["groups"].items(), key=lambda item: natural_key(item[0]))]}
        return ordered

    @staticmethod
    def _classify_take(rel_parts):
        """(family, group key) mirroring dataset_identity's rules: groupN/HOI|MP
        pairs -> that family keyed by the group directory; body tree -> BODY
        keyed by the actor directory."""
        for index, part in enumerate(rel_parts):
            if part.upper() in ("HOI", "MP") and index >= 1 and re.fullmatch(r"group[1-9]\d*", rel_parts[index - 1], re.IGNORECASE):
                return part.upper(), rel_parts[index - 1], part.upper()
            if part.lower() == "body":
                actor = rel_parts[index + 1] if len(rel_parts) > index + 1 else "(root)"
                return "BODY", actor, "BODY"
        top = rel_parts[1] if len(rel_parts) > 1 else "(root)"
        return "OTHER", top, "OTHER"

    def _purge_superseded(self, metadata):
        """队列不变式:每条序列只保留一个任务。当一条序列的处理刚刚定案
        (完成/需复核/失败等),同序列更早的终态任务被移除(释放其存储)。
        活动/排队任务从不被动移除。调用方必须持有 self._lock。"""
        removed = []
        for item in list(self._jobs.values()):
            if (item["id"] != metadata["id"] and item["input_path"] == metadata["input_path"]
                    and item.get("state") in TERMINAL):
                directory = self._directory(item)
                try:
                    if directory.parent == self.jobs_dir and directory.is_dir():
                        shutil.rmtree(directory)
                    else:
                        continue
                except OSError:
                    continue
                self._jobs.pop(item["id"], None)
                removed.append(item["id"])
        if removed:
            print("Superseded tasks purged for %s: %d" % (metadata["name"], len(removed)), file=sys.stderr)

    def clear_finished(self):
        removed = []
        with self._lock:
            ids = [item["id"] for item in self._jobs.values() if item.get("state") in TERMINAL]
        for job_id in ids:
            try:
                self.delete(job_id)
                removed.append(job_id)
            except APIError:
                continue
        return {"removed": removed, "count": len(removed)}

    @staticmethod
    def _options(value):
        if not isinstance(value, dict) or set(value) - DEFAULT_OPTIONS.keys():
            raise APIError(400, "Unknown or invalid processing options")
        options = {**DEFAULT_OPTIONS, **value}
        for key, default in DEFAULT_OPTIONS.items():
            item = options[key]
            if isinstance(default, bool):
                if type(item) is not bool:
                    raise APIError(400, f"{key} must be boolean")
            elif key.startswith("expected_"):
                if item is not None and (not isinstance(item, list) or len(item) > 200 or any(not isinstance(name, str) or not name.strip() or len(name) > 200 for name in item)):
                    raise APIError(400, f"{key} must be null or an array of names")
            elif type(item) not in (float, int) or not math.isfinite(item) or item < 0 or (key == "prop_tolerance" and item == 0):
                raise APIError(400, f"{key} must be a finite {'positive' if key == 'prop_tolerance' else 'nonnegative'} number")
        return options

    def enqueue(self, body):
        if set(body) - {"paths", "mode", "options"}:
            raise APIError(400, "Unknown job fields")
        paths, mode = body.get("paths"), body.get("mode", "manual")
        if not isinstance(paths, list) or not 1 <= len(paths) <= 200 or not isinstance(mode, str) or mode not in {"manual", "auto"}:
            raise APIError(400, "Provide 1..200 paths and manual or auto mode")
        inputs = list(dict.fromkeys(self._allowed_input(item) for item in paths))
        options = self._options(body.get("options", {}))
        with self._lock:
            if self._closed:
                raise APIError(503, "Manager is shutting down")
            result = []
            for source in inputs:
                job_id = uuid.uuid4().hex
                directory = self.jobs_dir / job_id
                directory.mkdir()
                now = utc_now()
                metadata = {"id": job_id, "name": source.name, "input_path": str(source), "input_kind": source.suffix[1:].lower(),
                            "mode": mode, "state": "QUEUED", "created_at": now, "updated_at": now, "error": None,
                            "options": options.copy(), "command_seq": 0}
                request = {"job_id": job_id, "input_path": str(source), "job_dir": str(directory),
                           "executable": self.executable, "blender": self.blender, "mode": mode,
                           "options": options, "log_path": str(directory / "processing.log"),
                           "data_roots": [str(root) for root in self.data_roots]}
                atomic_json(directory / "request.json", request)
                (directory / "commands.jsonl").touch(exist_ok=False)
                self._save(metadata)
                self._jobs[job_id] = metadata
                result.append(self._dto(metadata))
            return {"jobs": result}

    def outputs(self, limit=24):
        """Recently modified processing outputs under each data root's
        02_*/03_* result trees (bounded walk, newest first)."""
        found = []
        visited = 0
        for root in self.data_roots:
            try:
                first_level = [entry for entry in root.iterdir()
                               if entry.is_dir() and not entry.is_symlink()
                               and re.fullmatch(r"0[23]_[0-9A-Za-z]+", entry.name)]
            except OSError:
                continue
            for top in first_level:
                for directory, directories, files in os.walk(top):
                    directories.sort()
                    for name in files:
                        if Path(name).suffix.casefold() not in {".fbx", ".vdf", ".vsk"}:
                            continue
                        visited += 1
                        if visited > 20000:
                            break
                        try:
                            path = Path(directory) / name
                            stat = path.stat()
                        except OSError:
                            continue
                        found.append({"name": name, "path": str(path), "kind": Path(name).suffix[1:].lower(),
                                      "size_bytes": stat.st_size,
                                      "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                                      "deletable": True})
                    if visited > 20000:
                        break
                if visited > 20000:
                    break
            if visited > 20000:
                break
        found.sort(key=lambda item: item["modified_at"], reverse=True)
        return {"outputs": found[:max(1, min(int(limit or 24), 500))]}

    def deletable_output(self, path):
        """Files under a data root's 02_*/03_* result trees may be deleted from
        the UI; source data (01_data_raw and anything else) never."""
        if not path.is_file():
            return False
        for root in self.data_roots:
            if inside(path, root) and path != root:
                first = path.relative_to(root).parts[0] if path.relative_to(root).parts else ""
                if re.fullmatch(r"0[23]_[0-9A-Za-z]+", first):
                    return True
        return False

    def delete_output(self, path_value):
        if not isinstance(path_value, str) or not path_value.strip():
            raise APIError(400, "path required")
        try:
            path = Path(path_value).resolve(strict=True)
        except (OSError, ValueError):
            raise APIError(404, "file not found")
        if not self.deletable_output(path):
            raise APIError(403, "only files under 02_*/03_* result directories inside a data root may be deleted")
        with self._lock:
            active = {Path(item["input_path"]) for item in self._jobs.values() if item["state"] not in TERMINAL}
        if path in active:
            raise APIError(409, "file is the input of an active job")
        try:
            path.unlink()
        except OSError as exc:
            raise APIError(500, f"delete failed: {exc}")
        return {"deleted": str(path)}

    def clear_archive(self, job_id):
        """Delete only this run's archive copies under the data roots
        (02_processed / 03_exports mirrors). The job directory, source inputs
        and every non-archive artifact are never touched."""
        with self._lock:
            metadata = self._jobs.get(job_id)
            if metadata is None:
                raise APIError(404, "job not found")
            state = self._worker_state(metadata)
            source = Path(metadata["input_path"])
            removed = []
            for artifact in state.get("artifacts", []):
                if not isinstance(artifact, dict) or not str(artifact.get("role", "")).startswith("archive_"):
                    continue
                try:
                    path = Path(artifact["path"]).resolve(strict=True)
                except (KeyError, OSError, ValueError):
                    continue
                if not path.is_file() or path == source or not any(inside(path, root) for root in self.data_roots):
                    continue
                try:
                    path.unlink()
                    removed.append(str(path))
                except OSError:
                    continue
            return {"removed": removed}

    def _artifact_path(self, metadata, artifact):
        if not isinstance(artifact, dict) or not isinstance(artifact.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", artifact["id"]):
            return None
        try:
            path = Path(artifact["path"]).resolve(strict=True)
            source = Path(metadata["input_path"])
            if not path.is_file():
                return None
            if inside(path, self._directory(metadata)) or (path == source and metadata["input_kind"] == "fbx"):
                return path
            if str(artifact.get("role", "")).startswith("archive_") or str(artifact.get("role", "")) == "export_fbx":
                if any(inside(path, root) for root in self.data_roots):
                    return path
            return None
        except (KeyError, TypeError, OSError, ValueError):
            return None

    def _dto(self, metadata):
        worker = self._worker_state(metadata)
        dto = {key: metadata.get(key) for key in ("id", "name", "input_path", "input_kind", "mode", "state", "created_at", "updated_at", "error", "options", "review")}
        for key, default in (("current_stage", None), ("waiting_stage", None), ("stages", []), ("artifacts", [])):
            dto[key] = worker.get(key, default)
        # Queue/status polling must not copy multi-megabyte stage results.
        # Full stage reports remain available through the report/artifact endpoints.
        stage_fields = ("id", "index", "name", "label", "subject", "status",
                        "started_at", "finished_at", "elapsed_seconds", "error",
                        "checkpoint", "checkpoint_error", "reason")
        dto["stages"] = [{key: stage[key] for key in stage_fields if key in stage}
                         for stage in worker.get("stages", [])]
        for key in ("quality_status", "worker_pid"):
            if key in worker:
                dto[key] = worker[key]
        if metadata["state"] in ACTIVE and worker.get("state") in ACTIVE | TERMINAL:
            dto["state"] = worker["state"]
            dto["error"] = worker.get("error")
            dto["updated_at"] = worker.get("updated_at", dto["updated_at"])
        pending = metadata.get("pending")
        if pending and metadata["state"] not in TERMINAL:
            acknowledged = worker.get("command_seq", 0) >= pending["seq"]
            changed = worker.get("updated_at") != pending.get("worker_updated_at")
            # Older workers without command_seq acknowledge by a new state snapshot.
            if acknowledged or ("command_seq" not in worker and changed):
                metadata.pop("pending", None)
                self._save(metadata)
            else:
                if pending["action"] in {"step", "run", "preview"}:
                    dto["state"] = "RUNNING"
                elif pending["action"] == "pause":
                    dto["state"] = "PAUSE_REQUESTED"
        if metadata["state"] in TERMINAL:
            dto["state"], dto["error"] = metadata["state"], metadata.get("error")
            dto.pop("worker_pid", None)
        if metadata["id"] in self._owned and self._owned[metadata["id"]].get("cancel_at") is not None:
            dto["state"] = "PAUSE_REQUESTED"
        artifacts = []
        for artifact in worker.get("artifacts", []):
            path = self._artifact_path(metadata, artifact)
            if path is not None:
                item = {key: artifact[key] for key in ("id", "name", "role", "stage_id") if key in artifact}
                item.update(path=str(path), size_bytes=path.stat().st_size,
                            url=f"/api/processing/artifacts/{metadata['id']}/{quote(artifact['id'], safe='')}")
                artifacts.append(item)
        dto["artifacts"] = artifacts
        return dto

    def jobs(self):
        with self._lock:
            return {"jobs": [self._dto(item) for item in sorted(self._jobs.values(), key=lambda item: item["created_at"], reverse=True)]}

    def job(self, job_id):
        with self._lock:
            return {"job": self._dto(self._get(job_id))}

    def _command(self, metadata, action):
        metadata["command_seq"] += 1
        command = {"seq": metadata["command_seq"], "action": action}
        # Persist reservation first: a crash may skip a sequence, never reuse it.
        metadata["pending"] = {**command, "worker_updated_at": self._worker_state(metadata).get("updated_at")}
        metadata["updated_at"] = utc_now()
        self._save(metadata)
        with (self._directory(metadata) / "commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(command) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def action(self, job_id, body):
        action = body.get("action")
        allowed_actions = {"step", "run", "pause", "cancel", "preview", "retry", "approve"}
        note = None
        if action == "approve":
            if set(body) != {"action", "note"} or not isinstance(body.get("note"), str) or len(body["note"]) > 500:
                raise APIError(400, "approve expects an optional note up to 500 characters")
            note = body["note"].strip()
        elif set(body) != {"action"} or not isinstance(action, str) or action not in allowed_actions:
            raise APIError(400, "Unknown job action")
        with self._lock:
            metadata = self._get(job_id)
            state = self._dto(metadata)["state"]
            if action == "approve":
                if state != "NEEDS_REVIEW":
                    raise APIError(409, "Only a job needing manual review can be approved")
                self._set_status(metadata, "SUCCESS")
                metadata["review"] = {"approved_at": utc_now(), "note": note or ""}
                self._save(metadata)
                report_path = self._directory(metadata) / "report.json"
                try:
                    report = read_json(report_path)
                    if isinstance(report, dict):
                        report["status"] = "SUCCESS"
                        report["review"] = metadata["review"]
                        quality = report.get("quality")
                        if isinstance(quality, dict) and quality.get("usable") is False:
                            quality["usable"] = True
                            quality["scope"] = "approved_override"
                        else:
                            report["quality"] = {"basis": "post_processing", "usable": True, "scope": "approved_override"}
                        atomic_json(report_path, report)
                except OSError:
                    pass
                return {"job": self._dto(metadata)}
            if action == "retry":
                if state not in TERMINAL:
                    raise APIError(409, "Only a terminal job can be retried")
                result = self.enqueue({"paths": [metadata["input_path"]], "mode": metadata["mode"], "options": metadata["options"]})
                return {"job": result["jobs"][0]}
            if action == "cancel":
                if state in TERMINAL or (job_id in self._owned and self._owned[job_id].get("cancel_at") is not None):
                    raise APIError(409, "Job is terminal or cancellation is already requested")
                if state == "QUEUED":
                    self._set_status(metadata, "CANCELLED")
                elif job_id in self._owned:
                    self._command(metadata, "cancel")
                    self._owned[job_id]["cancel_at"] = time.monotonic()
                else:
                    raise APIError(409, "Job has no owned live worker")
            else:
                allowed = state == "WAITING" if action in {"step", "run", "preview"} else state == "RUNNING"
                if not allowed or job_id not in self._owned or metadata.get("pending"):
                    raise APIError(409, "Action is not valid at the current stage boundary")
                self._command(metadata, action)
            return {"job": self._dto(metadata)}

    def logs(self, job_id, offset):
        if type(offset) is not int or offset < 0:
            raise APIError(400, "offset must be a nonnegative byte offset")
        with self._lock:
            path = self._directory(self._get(job_id)) / "processing.log"
        try:
            with path.open("rb") as stream:
                stream.seek(min(offset, os.fstat(stream.fileno()).st_size))
                data = stream.read(65536)
                return {"text": data.decode("utf-8", errors="replace"), "next_offset": stream.tell()}
        except FileNotFoundError:
            return {"text": "", "next_offset": 0}

    def report(self, job_id):
        with self._lock:
            path = self._directory(self._get(job_id)) / "report.json"
        return {"report": read_json(path)}

    def artifact(self, job_id, artifact_id):
        with self._lock:
            metadata = self._get(job_id)
            for artifact in self._worker_state(metadata).get("artifacts", []):
                if artifact.get("id") == artifact_id:
                    path = self._artifact_path(metadata, artifact)
                    if path is not None:
                        return path
            raise APIError(404, "Artifact is not registered or is unavailable")

    def _launch(self, metadata):
        self._set_status(metadata, "STARTING")
        # Queue invariant enforced at start: a re-run immediately retires the
        # previous terminal tasks of the same sequence (only ever one per take).
        self._purge_superseded(metadata)
        log = None
        process = None
        owned = None
        try:
            if not self.worker_script.is_file():
                raise FileNotFoundError(f"Worker script not found: {self.worker_script}")
            # Revalidate in case a queued source was removed/replaced by a link.
            self._allowed_input(metadata["input_path"])
            log = (self._directory(metadata) / "processing.log").open("ab", buffering=0)
            process = subprocess.Popen([sys.executable, "-u", str(self.worker_script), "--request", str(self._directory(metadata) / "request.json")],
                                       cwd=str(self.worker_script.parent), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
                                       start_new_session=os.name != "nt")
            owned = {"process": process, "log": log, "job_handle": None, "cancel_at": None}
            self._owned[metadata["id"]] = owned
            if os.name == "nt":
                from pipeline_session import _kill_on_close_job
                owned["job_handle"] = _kill_on_close_job(process)
        except Exception as exc:
            if process is not None:
                self._terminate(owned)
                process.wait()
                self._release(owned)
                self._owned.pop(metadata["id"], None)
            elif log is not None:
                log.close()
            self._set_status(metadata, "FAILED", f"Worker launch failed: {exc}")

    @staticmethod
    def _terminate(owned):
        process = owned["process"]
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()

    @staticmethod
    def _release(owned):
        if owned.get("job_handle"):
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle(owned["job_handle"])
            owned["job_handle"] = None
        elif os.name != "nt":
            # Reap descendants left behind by a failed Python worker only.
            try:
                os.killpg(owned["process"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        owned["log"].close()

    def tick(self):
        with self._lock:
            for job_id, owned in list(self._owned.items()):
                process = owned["process"]
                cancel_at = owned.get("cancel_at")
                if cancel_at is not None and time.monotonic() - cancel_at >= self.cancel_grace and process.poll() is None:
                    self._terminate(owned)
                code = process.poll()
                if code is None:
                    continue
                metadata = self._jobs[job_id]
                worker = self._worker_state(metadata)
                if cancel_at is not None:
                    self._set_status(metadata, "CANCELLED", worker.get("error") if worker.get("state") == "CANCELLED" else None)
                elif worker.get("state") in TERMINAL:
                    self._set_status(metadata, worker["state"], worker.get("error"))
                else:
                    self._set_status(metadata, "FAILED", f"Worker exited ({code}) without a terminal state; partial diagnostics are retained.")
                self._release(owned)
                del self._owned[job_id]
                self._purge_superseded(metadata)
            if not self._stop.is_set():
                for metadata in sorted(self._jobs.values(), key=lambda item: item["created_at"]):
                    if len(self._owned) >= self.max_workers:
                        break
                    if metadata["state"] == "QUEUED":
                        self._launch(metadata)

    def start(self):
        if self._thread is not None:
            return
        def schedule():
            while not self._stop.wait(0.2):
                try:
                    self.tick()
                except Exception as exc:
                    print(f"Queue error: {exc}", file=sys.stderr, flush=True)
        self._thread = threading.Thread(target=schedule, name="mocap-queue", daemon=True)
        self._thread.start()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            for job_id, owned in self._owned.items():
                try:
                    self._command(self._jobs[job_id], "cancel")
                except OSError:
                    pass
            deadline = time.monotonic() + self.cancel_grace
            while time.monotonic() < deadline and any(item["process"].poll() is None for item in self._owned.values()):
                time.sleep(0.05)
            for job_id, owned in self._owned.items():
                self._terminate(owned)
                owned["process"].wait()
                worker = self._worker_state(self._jobs[job_id])
                if owned.get("cancel_at") is not None:
                    state, error = "CANCELLED", None
                elif worker.get("state") in TERMINAL - {"CANCELLED"}:
                    state, error = worker["state"], worker.get("error")
                else:
                    state, error = "INTERRUPTED", "Manager stopped; retry starts a new isolated run."
                self._set_status(self._jobs[job_id], state, error)
                self._release(owned)
            self._owned.clear()
        self._state_lock.close()


def allowed_authority(authority, ports):
    try:
        parsed = urlsplit("http://" + authority)
        return (parsed.hostname in {"localhost", "127.0.0.1", "::1"} and parsed.port in ports
                and parsed.username is None and parsed.password is None and parsed.path == "" and not parsed.query and not parsed.fragment)
    except ValueError:
        return False


class ProcessingServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, manager, additional_origins=()):
        if address[0] != "127.0.0.1":
            raise ValueError("Processing server must bind to 127.0.0.1")
        self.manager = manager
        self.additional_origins = frozenset(additional_origins)
        super().__init__(address, ProcessingHandler)


class ProcessingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, status, value):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _security(self):
        ports = {8767, self.server.server_address[1]}
        extras = self.server.additional_origins
        extra_hosts = {urlsplit(origin).netloc for origin in extras}
        host = self.headers.get("Host", "")
        if len(self.headers.get_all("Host", [])) != 1 or not (allowed_authority(host, ports) or host in extra_hosts):
            raise APIError(403, "Untrusted Host")
        origin = self.headers.get("Origin")
        if origin:
            try:
                parsed = urlsplit(origin)
            except ValueError:
                raise APIError(403, "Untrusted Origin") from None
            if origin not in extras and not (parsed.scheme in {"http", "https"} and allowed_authority(parsed.netloc, ports)
                                             and not parsed.path and not parsed.query and not parsed.fragment):
                raise APIError(403, "Untrusted Origin")
        if self.command == "POST":
            token = self.headers.get("X-Mocap-Token", "")
            if not token.isascii() or not secrets.compare_digest(token, self.server.manager.csrf_token):
                raise APIError(403, "Missing or invalid X-Mocap-Token")
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise APIError(415, "POST requests require application/json")

    def _body(self):
        if self.headers.get("Transfer-Encoding"):
            raise APIError(400, "Transfer-Encoding is not accepted")
        try:
            size = int(self.headers.get("Content-Length", "-1"))
            if not 0 <= size <= 1024 * 1024:
                raise ValueError()
            value = json.loads(self.rfile.read(size))
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except (ValueError, UnicodeError):
            raise APIError(400, "Expected a JSON object of at most 1MB") from None

    def _file(self, path):
        try:
            stream = path.open("rb")
        except OSError:
            raise APIError(404, "Artifact is unavailable") from None
        with stream:
            size = os.fstat(stream.fileno()).st_size
            start, end, partial = 0, size - 1, False
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=([0-9]{0,20})-([0-9]{0,20})", requested)
                if not match or not any(match.groups()):
                    self._range_error(size)
                    return
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = min(int(last), size - 1) if last else size - 1
                else:
                    count = int(last)
                    start = max(0, size - count)
                    end = size - 1
                    if count == 0:
                        start = size
                if start >= size or start > end:
                    self._range_error(size)
                    return
                partial = True
            count = max(0, end - start + 1)
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(count))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "private, no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Disposition", "inline; filename*=UTF-8''" + quote(path.name, safe=""))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command != "HEAD":
                stream.seek(start)
                while count:
                    data = stream.read(min(count, 1024 * 1024))
                    if not data:
                        break
                    self.wfile.write(data)
                    count -= len(data)

    def _range_error(self, size):
        self.send_response(416)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _dispatch(self):
        try:
            self._security()
            url = urlsplit(self.path)
            prefix = "/api/processing"
            if not url.path.startswith(prefix + "/"):
                raise APIError(404, "Unknown endpoint")
            route = unquote(url.path[len(prefix):])
            query = parse_qs(url.query)
            manager = self.server.manager
            get = self.command in {"GET", "HEAD"}
            if get and route == "/config":
                result = manager.config()
            elif get and route == "/browse":
                recursive = query.get("recursive", ["false"])[0]
                if recursive not in {"true", "false"}:
                    raise APIError(400, "recursive must be true or false")
                result = manager.browse(query.get("path", [None])[0], recursive == "true")
            elif get and route == "/tree":
                result = manager.tree(query.get("path", [None])[0])
            elif get and route == "/find":
                result = manager.find(query.get("q", [None])[0])
            elif get and route == "/inspect":
                result = manager.inspect(query.get("path", [None])[0])
            elif get and route == "/jobs":
                result = manager.jobs()
            elif get and (match := re.fullmatch(r"/jobs/([a-f0-9]{32})(?:/(logs|report))?", route)):
                job_id, detail = match.groups()
                if detail == "logs":
                    try:
                        offset = int(query.get("offset", ["0"])[0])
                    except ValueError:
                        raise APIError(400, "Invalid log offset") from None
                    result = manager.logs(job_id, offset)
                else:
                    result = manager.report(job_id) if detail == "report" else manager.job(job_id)
            elif get and (match := re.fullmatch(r"/artifacts/([a-f0-9]{32})/([A-Za-z0-9_-]{1,128})", route)):
                self._file(manager.artifact(*match.groups()))
                return
            elif get and route == "/output-file":
                value = (parse_qs(url.query).get("path") or [""])[0]
                try:
                    candidate = Path(value).resolve(strict=True)
                except (OSError, ValueError):
                    raise APIError(404, "file not found")
                if not manager.deletable_output(candidate):
                    raise APIError(403, "only files under 02_*/03_* result directories may be served")
                self._file(candidate)
                return
            elif self.command == "POST" and route == "/config":
                result = manager.configure(self._body())
            elif self.command == "POST" and route == "/jobs":
                result = manager.enqueue(self._body())
                self._json(201, result)
                return
            elif self.command == "POST" and (match := re.fullmatch(r"/jobs/([a-f0-9]{32})/actions", route)):
                result = manager.action(match[1], self._body())
            elif self.command == "POST" and (match := re.fullmatch(r"/jobs/([a-f0-9]{32})/delete", route)):
                result = manager.delete(match[1])
            elif self.command == "POST" and (match := re.fullmatch(r"/jobs/([a-f0-9]{32})/clear-archive", route)):
                result = manager.clear_archive(match[1])
            elif self.command == "POST" and route == "/delete-output":
                result = manager.delete_output(self._body().get("path"))
            elif get and route == "/outputs":
                result = manager.outputs(parse_qs(url.query).get("limit", [None])[0])
            elif get and route == "/dataset-progress":
                result = manager.dataset_progress()
            elif self.command == "POST" and route == "/jobs/clear-finished":
                result = manager.clear_finished()
            else:
                raise APIError(404, "Unknown endpoint or method")
            self._json(200, result)
        except APIError as exc:
            self.close_connection = True
            self._json(exc.status, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:
            self.close_connection = True
            print(f"HTTP error: {exc}", file=sys.stderr, flush=True)
            self._json(500, {"error": "Internal processing manager error; see manager log"})

    do_GET = _dispatch
    do_HEAD = _dispatch
    do_POST = _dispatch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default=os.environ.get("MOCAP_STATE_DIR", "H:/MocapData/jobs"))
    parser.add_argument("--jobs-dir", default=os.environ.get("MOCAP_JOBS_DIR", ""))
    parser.add_argument("--data-root", action="append", help="Allowed dataset root; repeat for multiple roots")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MOCAP_MANAGER_PORT", "8767")))
    parser.add_argument("--max-workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--shogun-cl", default=os.environ.get("MOCAP_SHOGUN_CL"))
    parser.add_argument("--blender", default=os.environ.get("MOCAP_BLENDER"))
    parser.add_argument("--allow-origin", action="append", default=[])
    args = parser.parse_args(argv)
    roots = args.data_root or [part for part in os.environ.get("MOCAP_DATA_ROOTS", "H:/MocapData").split(os.pathsep) if part]
    shogun_roots = [Path("D:/Software/Vicon"), Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Vicon"]
    executable = discover_executable(args.shogun_cl, "ShogunPostCL.exe", [(root, "ShogunPost*/ShogunPostCL.exe") for root in shogun_roots])
    blender = discover_executable(args.blender, "blender", [(Path("D:/Software/Blender"), "blender.exe"),
                                  (Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Blender Foundation", "Blender*/blender.exe")])
    origins = args.allow_origin + [item.strip() for item in os.environ.get("MOCAP_ALLOWED_ORIGINS", "").split(",") if item.strip()]
    manager = JobManager(args.state_dir, roots, executable=executable, blender=blender, max_workers=args.max_workers,
                         jobs_dir=args.jobs_dir or None)
    server = None
    try:
        server = ProcessingServer(("127.0.0.1", args.port), manager, origins)
        manager.start()
        def stop(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        print(f"Mocap manager listening on http://127.0.0.1:{server.server_address[1]}", flush=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
