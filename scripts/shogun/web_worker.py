"""Owned, command-gated processing worker; no discovery or SDK connection on import.

Each invocation consumes one immutable request.json and a sequenced commands.jsonl.
State/report replacement is atomic. Checkpoints are inspection artifacts, not restart
instructions: retry always means a new invocation against the original input.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid

from pipeline_processing import _stage, stage_observer


DEFAULT_OPTIONS = {
    "filter_markers": True, "expected_subjects": None, "expected_props": None,
    "dataset_mode": True, "attach_meshes": False, "lightweight": True,
    "drop_unused_labeling": True, "prop_jitter_max_correction": 0.0,
    "prop_tolerance": 2.0, "denoise_static": False,
}
TERMINAL = {"SUCCESS", "NEEDS_REVIEW", "FAILED", "CANCELLED", "INTERRUPTED"}
# Full-scene saves are expensive. Read-only inspection/capture/export stages
# retain report snapshots, while only successful scene mutations need a new VDF.
_CHECKPOINT_STAGES = {
    "load", "configure_dataset_scene", "reconstruct", "initial_label", "repair",
    "inspect_repair_props", "fix_occlusion", "filter_markers", "repair_prop_jitter",
    "repair_original_live_label_disagreement", "solve_reference_repaired_human",
    "solve_labeling", "solve_solving", "solve_rigid_props", "retarget",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                        allow_nan=False), encoding="utf-8")
        # The HTTP handler reads state/report files concurrently; on Windows a
        # reader without DELETE sharing can briefly block os.replace (WinError 5).
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05 + 0.05 * attempt)
    finally:
        temporary.unlink(missing_ok=True)


class Cancelled(BaseException):
    """Bypass process_take's ordinary-error recovery and further native calls."""


class CommandReader:
    """Read complete append-only records, retaining any partial final line."""

    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.pending = b""
        self.last_seq = 0

    def read(self):
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                data = stream.read()
                self.offset = stream.tell()
        except FileNotFoundError:
            return []
        lines = (self.pending + data).split(b"\n")
        self.pending = lines.pop()
        commands = []
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            seq = record.get("seq")
            if isinstance(seq, bool) or not isinstance(seq, int) or seq <= 0:
                raise ValueError("Command seq must be a positive integer")
            if seq > self.last_seq:
                commands.append(record)
                self.last_seq = seq
        return commands


def dataset_identity(source):
    """Require the actual groupN/HOI|MP ancestor pair, not filename tokens.

    Body captures live under a dedicated ``body`` tree (01_data_raw/body/
    actorXXX/<take>) and are human-only datasets like MP takes.
    """
    source = Path(source)
    for category in source.parents:
        if category.name.upper() in {"HOI", "MP"} and re.fullmatch(
                r"group[1-9]\d*", category.parent.name, re.IGNORECASE):
            return category.parent.parent, category.name.upper()
    for ancestor in source.parents:
        if ancestor.name.lower() == "body":
            return ancestor.parent, "BODY"
    return None, None


def preflight_fbx(path):
    from fbx_lightweight import _index
    import mesh_fbx as fbx
    source = Path(path)
    data = source.read_bytes()
    parser = fbx.load_parser()
    tops = fbx.parse_checked(parser, data, allow_unterminated_children=True)
    _, objects, models, links = _index(parser, data, tops)
    prop_markers = set()
    model_parents = {}
    for _, link in links:
        if link[0] == "OO" and link[1] in models and link[2] in models:
            model_parents[link[1]] = link[2]
    for uid, model in models.items():
        if model["kind"] == "Marker" and uid in model_parents:
            name = models[model_parents[uid]]["name"].split(":", 1)[0]
            if name != "Unlabeled_Markers":
                prop_markers.add(name)
    return {"source_bytes": len(data), "fbx_version": int.from_bytes(data[23:27], "little"),
            "object_count": len(objects), "model_count": len(models),
            "object_counts": dict(Counter(node.name.decode("ascii") for node in objects.values())),
            "model_counts": dict(Counter(model["kind"] for model in models.values())),
            "connection_count": len(links), "marker_parent_names": sorted(prop_markers),
            "physical_motion_repaired": False}


def source_quality(source):
    """Inherit only a sibling report that actually declares this exact FBX."""
    source = Path(source).resolve()
    for path in (source.parent / "report.json", source.parent.parent / "report.json"):
        if not path.is_file():
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8-sig"))
            declared = report.get("artifacts", {})
            if not isinstance(declared, dict):
                continue
            matches = any(isinstance(value, str) and
                          (base / value).resolve() == source
                          for value in declared.values()
                          for base in (path.parent, path.parent / "artifacts"))
            if matches and report.get("status") in {"SUCCESS", "NEEDS_REVIEW", "FAILED"}:
                status = report["status"]
                if status == "SUCCESS" and not report.get("quality", {}).get("usable"):
                    status = "NEEDS_REVIEW"
                return status, {"report_path": str(path), "status": report["status"],
                                "quality": report.get("quality"),
                                "review_reasons": report.get("review_reasons", [])}
        except (OSError, ValueError, TypeError):
            continue
    return "NEEDS_REVIEW", {"status": "UNKNOWN", "quality": {"usable": False},
                            "review_reasons": ["source_physical_quality_unverified"]}


class Worker:
    def __init__(self, request):
        self.request = request
        self.directory = Path(request["job_dir"]).resolve()
        self.source = Path(request["input_path"]).resolve(strict=True)
        if not self.source.is_file() or self.source.suffix.lower() not in {".mcp", ".vdf", ".fbx"}:
            raise ValueError("Input must be an existing MCP, VDF or FBX file")
        if self.source == self.directory or self.directory in self.source.parents:
            raise ValueError("Input must not be inside this run's output directory")
        if request.get("mode", "manual") not in {"manual", "auto"}:
            raise ValueError("mode must be manual or auto")
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "state.json").exists():
            raise ValueError("Refusing to resume an existing worker state; create a new run")
        self.options = {**DEFAULT_OPTIONS, **request.get("options", {})}
        self.condition = threading.Condition(threading.RLock())
        self.stopped = threading.Event()
        self.cancelled = False
        self.continuous = request.get("mode") == "auto"
        self.permits = 0
        self.preview_pending = False
        self.command_error = None
        self.reader = CommandReader(self.directory / "commands.jsonl")
        self.session = None
        self.loaded = False
        self.current_fbx = None
        self.active = []
        self.report = {"status": "RUNNING", "subjects": [], "props": [], "stages": [],
                       "artifacts": {}, "review_reasons": [], "job_id": request["job_id"],
                       "input_path": str(self.source), "input_kind": self.source.suffix[1:].lower()}
        self.state = {"state": "STARTING", "current_stage": None, "waiting_stage": None,
                      "stages": [], "artifacts": [], "error": None, "worker_pid": os.getpid(),
                      "command_seq": 0}
        self._write_state()
        self._snapshot()
        self.thread = threading.Thread(target=self._commands, name="processing-commands", daemon=True)

    def _write_state(self):
        self.state["updated_at"] = utc_now()
        atomic_json(self.directory / "state.json", self.state)

    def _event(self, kind, **data):
        with self.condition:
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
                line = json.dumps({"at": utc_now(), "type": kind, **data},
                                  ensure_ascii=False, allow_nan=False)
                stream.write(line + "\n")
            print(line, flush=True)

    def register(self, path, role, stage_id=None, *, source=False, archive=False):
        path = Path(path).resolve(strict=True)
        allowed = path.is_file() and (self.directory in path.parents or (source and path == self.source))
        if archive and not allowed:
            allowed = any(Path(root).resolve() in path.parents for root in self.request.get("data_roots", []))
        if not allowed:
            raise ValueError(f"Artifact escapes the run directory: {path}")
        identifier = hashlib.sha256((str(path) + "\0" + role).encode("utf-8")).hexdigest()[:24]
        artifact = {"id": identifier, "name": path.name, "role": role,
                    "path": str(path), "size_bytes": path.stat().st_size}
        if stage_id:
            artifact["stage_id"] = stage_id
        with self.condition:
            existing = next((item for item in self.state["artifacts"] if item["id"] == identifier), None)
            if existing is None:
                self.state["artifacts"].append(artifact)
            else:
                existing.update(artifact)
        return identifier

    def _snapshot(self, entry=None, *, final=False):
        native = self.directory / "native"
        if native.is_dir():
            self._register_native(self.report, native)
        snapshot = deepcopy(self.report)
        if not final:
            snapshot["status"] = "RUNNING"
            snapshot["partial"] = True
            snapshot["pipeline_status_pending"] = self.report.get("status")
        else:
            snapshot["partial"] = False
        snapshot.update(job_id=self.request["job_id"], input_path=str(self.source),
                        input_kind=self.source.suffix[1:].lower(), updated_at=utc_now())
        atomic_json(self.directory / "report.json", snapshot)
        self.register(self.directory / "report.json", "report")
        if entry:
            path = self.directory / "reports" / (entry["id"] + ".json")
            atomic_json(path, snapshot)
            self.register(path, "stage_report", entry["id"])
        with self.condition:
            self.state["stages"] = deepcopy(self.report["stages"])
            self._write_state()

    def _consume_commands(self):
        with self.condition:
            for command in self.reader.read():
                self.apply_command(command)

    def _commands(self):
        while not self.stopped.is_set():
            try:
                self._consume_commands()
            except Exception as exc:
                with self.condition:
                    self.command_error = f"Command stream failed: {type(exc).__name__}: {exc}"
                    self.continuous = False
                    self.condition.notify_all()
            self.stopped.wait(0.1)

    def apply_command(self, command):
        action = command.get("action")
        with self.condition:
            self.state["command_seq"] = max(self.state["command_seq"], command.get("seq", 0))
            if self.state["state"] in TERMINAL:
                self._write_state()
                return
            if action == "cancel":
                self.cancelled = True
                self.state["state"] = "PAUSE_REQUESTED"
            elif action == "pause":
                self.continuous = False
                self.permits = 0
                if self.state["state"] == "RUNNING":
                    self.state["state"] = "PAUSE_REQUESTED"
            elif action == "run":
                self.continuous = True
                self.permits = 0
                self.state["state"] = "RUNNING"
            elif action == "step" and self.state["state"] == "WAITING":
                self.continuous = False
                self.permits += 1
                self.state["state"] = "RUNNING"
            elif action == "preview" and self.state["state"] == "WAITING":
                self.preview_pending = True
                self.state["state"] = "RUNNING"
            else:
                self._event("command_rejected", command=command, state=self.state["state"])
                self._write_state()
                return
            self._event("command", command=command)
            self._write_state()
            self.condition.notify_all()

    def _check_stop(self):
        if self.cancelled:
            raise Cancelled()
        if self.command_error:
            raise RuntimeError(self.command_error)

    def before_stage(self, report, entry):
        self.report = report
        index = len(report["stages"])
        entry.update(id=f"{index:04d}-{entry['name']}", index=index - 1,
                     label=entry["name"].replace("_", " "), status="WAITING")
        self._snapshot()
        if entry["name"] == "parse":
            # The read-only roster parse always runs immediately: manual jobs
            # still get subjects/objects before the first human decision gate.
            entry.update(status="RUNNING", started_at=utc_now())
            self.active.append(entry["id"])
            self.state.update(state="RUNNING", current_stage=entry["id"], waiting_stage=None)
            self.state["stages"] = deepcopy(report["stages"])
            self._write_state()
            self._event("stage_started", stage_id=entry["id"])
            return
        while True:
            with self.condition:
                self._consume_commands()
                self._check_stop()
                if self.preview_pending:
                    self.preview_pending = False
                    self.state.update(state="RUNNING", current_stage=None, waiting_stage=entry["id"])
                    self._write_state()
                    preview = True
                elif self.continuous or self.permits:
                    if self.permits:
                        self.permits -= 1
                    entry.update(status="RUNNING", started_at=utc_now())
                    self.active.append(entry["id"])
                    self.state.update(state="RUNNING", current_stage=entry["id"], waiting_stage=None)
                    self.state["stages"] = deepcopy(report["stages"])
                    self._write_state()
                    self._event("stage_started", stage_id=entry["id"])
                    return
                else:
                    self.state.update(state="WAITING", current_stage=None, waiting_stage=entry["id"])
                    self._write_state()
                    self.condition.wait(0.2)
                    continue
            if preview:
                try:
                    self.export_preview(entry["id"])
                except Exception as exc:
                    error = f"Preview export failed: {type(exc).__name__}: {exc}"
                    with self.condition:
                        self.state["error"] = error
                    print(error, flush=True)
                    self._event("preview_failed", error=error)
                else:
                    with self.condition:
                        self.state["error"] = None
                with self.condition:
                    self.state.update(state="WAITING", current_stage=None,
                                      waiting_stage=entry["id"])
                self._snapshot()

    def after_stage(self, report, entry):
        self.report = report
        if "id" not in entry:
            index = len(report["stages"])
            entry.update(id=f"{index:04d}-{entry['name']}", index=index - 1,
                         label=entry["name"].replace("_", " "))
        entry["finished_at"] = utc_now()
        if entry["id"] in self.active:
            self.active.remove(entry["id"])
        if entry["status"] == "SUCCESS" and entry["name"] == "load":
            self.loaded = True
        if (entry["status"] == "SUCCESS" and self.loaded and self.session and not self.cancelled
                and entry["name"] in _CHECKPOINT_STAGES
                and not any(stage["status"] == "FAILED" for stage in report["stages"])):
            path = self.directory / "checkpoints" / (entry["id"] + ".vdf")
            path.parent.mkdir(exist_ok=True)
            try:
                self.session.api.SaveFile(path.as_posix())
                if not path.is_file() or not path.stat().st_size:
                    raise RuntimeError("Native checkpoint save did not produce a nonempty file")
                entry["checkpoint"] = str(path)
                self.register(path, "checkpoint_vdf", entry["id"])
            except Exception as exc:
                entry["checkpoint_error"] = f"{type(exc).__name__}: {exc}"
        with self.condition:
            # WAITING is published only inside the actual next-stage gate.
            self.state.update(state="RUNNING" if self.continuous else "PAUSE_REQUESTED",
                              current_stage=self.active[-1] if self.active else None,
                              waiting_stage=None)
        self._event("stage_finished", stage_id=entry["id"], status=entry["status"])
        self._snapshot(entry)
        self._check_stop()

    def export_preview(self, stage_id):
        if self.session and self.loaded:
            from pipeline_processing import _quote, _sdk_context
            path = self.directory / "previews" / (uuid.uuid4().hex + ".fbx")
            path.parent.mkdir(exist_ok=True)
            scene, _ = _sdk_context()
            scene.DeselectAllObjects()
            self.session.hsl("fbxExportOptions -exportActiveClipOnly true -collapseSubjects false "
                             "-exportPlayRangeOnly false -exportLabelingSetups true;")
            self.session.hsl(f"saveFile {_quote(path.as_posix())};")
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError("Native FBX export produced no nonempty file")
            self.register(path, "intermediate_fbx", stage_id)
        elif self.current_fbx:
            self.register(self.current_fbx, "intermediate_fbx", stage_id,
                          source=self.current_fbx == self.source)
        else:
            raise RuntimeError("No loaded scene or preflighted FBX is available yet")
        self._event("preview_exported", stage_id=stage_id)

    def _archive_outputs(self, native):
        """Copy final deliverables back into the dataset tree, mirroring the
        source layout. 02_processed receives the full working set
        (fbx/vdf/vsk); 03_exports receives ONE directly-usable FBX — the clean
        lightweight display trim when enabled (people + their markers only),
        otherwise the full FBX. Never overwrites the original input."""
        report = self.report
        if not any(stage.get("name") == "export_fbx" and stage.get("status") == "SUCCESS"
                   for stage in report.get("stages", [])):
            return
        roots = []
        for value in self.request.get("data_roots", []):
            try:
                roots.append(Path(value).resolve())
            except OSError:
                continue
        source = Path(report.get("input_path") or self.request["input_path"])
        root = next((item for item in roots if item == source or item in source.parents), None)
        if root is None:
            return
        relative = source.relative_to(root)
        parts = relative.parts[1:] if len(relative.parts) > 1 and relative.parts[0].casefold().startswith("01_") else relative.parts
        if not parts:
            return
        stem = Path(parts[-1]).stem
        tail = parts[:-1]
        processed = root / "02_processed" / Path(*tail) if tail else root / "02_processed"
        exports = root / "03_exports" / Path(*tail) if tail else root / "03_exports"
        art = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
        source_resolved = source.resolve(strict=False)

        def copy_to(local: Path, target: Path, role: str) -> None:
            if not local.is_file() or target.resolve(strict=False) == source_resolved:
                return
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.parent / (target.name + "." + uuid.uuid4().hex + ".tmp")
                shutil.copy2(local, temporary)
                os.replace(temporary, target)
                self.register(target, role, archive=True)
                report.setdefault("archived", []).append(str(target))
            except (OSError, ValueError) as exc:
                report.setdefault("archive_errors", []).append(f"{target}: {type(exc).__name__}: {exc}")

        for role, prefix in (("vdf", "archive_processed_vdf"), ("fbx", "archive_processed_fbx"), ("vsk", "archive_processed_vsk")):
            name = art.get(role)
            if isinstance(name, str) and name:
                copy_to(native / name, processed / f"{stem}{Path(name).suffix}", prefix)
        # 03_exports normally receives the trimmed FBX written directly by the
        # lightweight stage (export_fbx). Only when that stage did not run or
        # failed do we archive the full working FBX as the usable copy.
        light_ok = any(stage.get("name") == "lightweight_fbx" and stage.get("status") == "SUCCESS"
                       for stage in report.get("stages", []))
        if not light_ok:
            final_name = art.get("fbx")
            if isinstance(final_name, str) and final_name:
                copy_to(native / final_name, exports / f"{stem}.fbx", "archive_export_fbx")
        if report.get("archived"):
            self._event("archived", paths=list(report["archived"]))

    def _register_native(self, report, native):
        for role, value in report.get("artifacts", {}).items():
            if not isinstance(value, str):
                raise ValueError(f"Invalid native artifact declaration: {role}")
            path = (native / value).resolve()
            if native not in path.parents:
                raise ValueError(f"Native artifact escapes its workspace: {value}")
            if path.is_file():
                self.register(path, role)
        value = report.get("artifacts", {}).get("fbx")
        if value and (native / value).is_file():
            self.current_fbx = (native / value).resolve()

    def _native(self):
        from step1_process_takes import build_jobs, load_manifest
        from pipeline_processing import process_take
        from pipeline_session import ShogunSession
        executable = self.request.get("executable")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("A real ShogunPostCL executable is required for MCP/VDF processing")
        native = self.directory / "native"
        native.mkdir()
        job = build_jobs(self.source, self.directory / "plan", self.directory / "logs",
                         self.source.suffix[1:].lower(), load_manifest(None),
                         self.options["expected_subjects"], None, False,
                         expected_props=self.options["expected_props"])[0]
        root, category = dataset_identity(self.source)
        if category:
            # Human-only families (MP / BODY) force-disable all prop processing
            # regardless of the user's "dataset mode" toggle: no prop stage may run.
            # BODY takes are single-person shoots: an absent subject is tolerated.
            job.update(sequence_kind=category, props_enabled=category == "HOI",
                       tolerate_missing_subjects=category == "BODY")
        if self.options["dataset_mode"] and category:
            job.update(dataset_mode=True, split_subjects=True,
                       expected_human_count=(
                           1 if job["expected_subjects"] and len(job["expected_subjects"]) == 1
                           else None if category == "BODY" else 2),
                       relative_path=self.source.relative_to(root).as_posix())
        if job["preflight_errors"]:
            raise ValueError("; ".join(job["preflight_errors"]))
        policy = {key: self.options[key] for key in
                  ("filter_markers", "prop_jitter_max_correction", "prop_tolerance", "denoise_static")}
        # Session verifies listener ownership again before connecting: no GUI attachment.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self._consume_commands()
        self._check_stop()
        with ShogunSession(str(executable), native, self.directory / "shogun.log", port) as session:
            self.session = session
            self._event("session_ready", native_pid=session.pid, port=port, version=session.version)
            try:
                self.report = process_take(session, job, native, policy)
                self.report.update(job_id=self.request["job_id"], relative_path=job["relative_path"],
                                   expected_subjects=job["expected_subjects"], expected_props=job["expected_props"],
                                   session={"pid": session.pid, "port": port, "version": session.version})
                self._register_native(self.report, native)
                self._native_dir = native
            finally:
                self.session = None
        self._theme_audit(category)

    def _theme_audit(self, category=None):
        if category is None:
            _, category = dataset_identity(self.source)
        if category != "HOI":
            return
        from theme_roster import audit_take_mesh_completeness
        props = {row["name"] for row in self.report.get("props", []) if row.get("name")}
        if not props:
            props = set(self.report.get("preflight", {}).get("marker_parent_names", []))
        audit = audit_take_mesh_completeness(str(self.source), props, set())
        self.report["theme_roster_audit"] = audit
        # This audit concerns source marker roster, not proof of attached meshes.
        self.report["theme_roster_audit"]["evidence_scope"] = "source_marker_roster"
        missing = sorted({part for actor in audit["missing_by_actor"].values()
                          for part in actor["missing_scene_props"]})
        reasons = ["theme_roster_unverified"] if not audit["verified"] else []
        reasons.extend("theme_roster_missing_prop:" + name for name in missing)
        self.report["review_reasons"] = list(dict.fromkeys(self.report.get("review_reasons", []) + reasons))
        if reasons and self.report.get("status") == "SUCCESS":
            self.report["status"] = "NEEDS_REVIEW"
            self.report.setdefault("quality", {})["usable"] = False

    def _fbx(self):
        self.register(self.source, "source_fbx", source=True)
        # Stage 0001: read-only roster parse (subjects/objects/structure) before
        # any decision stage; a degraded scan is recorded, never fatal.
        with _stage(self.report, "parse") as entry:
            try:
                import sequence_inspect
                entry["result"] = sequence_inspect.parse_step(str(self.source))
            except Exception as exc:
                entry["result"] = {"warning": f"{type(exc).__name__}: {exc}"}
        status, inherited = source_quality(self.source)
        self.report.update(status=status, source_quality=inherited,
                           review_reasons=list(inherited.get("review_reasons", [])),
                           quality={"basis": "inherited_source_report", "usable": status == "SUCCESS",
                                    "physical_motion_repaired": False})
        with _stage(self.report, "fbx_preflight") as entry:
            self.report["preflight"] = preflight_fbx(self.source)
            entry["result"] = self.report["preflight"]
            self.current_fbx = self.source
        self._theme_audit()

    def _attach(self):
        with _stage(self.report, "attach_meshes") as entry:
            output = self.directory / "mounted"
            assets = Path(__file__).resolve().parents[4] / "3DObjects"
            if not assets.is_dir():
                raise FileNotFoundError(f"Mesh assets library not found: {assets}")
            blender = self.request.get("blender")
            if not blender:
                raise RuntimeError("Mesh attachment requires an installed Blender executable")
            command = [sys.executable, str(Path(__file__).with_name("step6_attach_prop_meshes.py")),
                       "--input", str(self.current_fbx), "--output-dir", str(output),
                       "--assets-root", str(assets), "--mapping",
                       str(Path(__file__).with_name("prop_meshes.json")), "--blender", str(blender),
                       "--export-stage", "props"]
            entry["command"] = command
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=sys.stdout,
                                       stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            owned_job = None
            try:
                if os.name == "nt":
                    from pipeline_session import _kill_on_close_job
                    owned_job = _kill_on_close_job(process)
                while process.poll() is None:
                    if self.cancelled:
                        raise Cancelled()
                    time.sleep(0.1)
                report_path = output / "mesh_batch_report.json"
                if not report_path.is_file():
                    raise RuntimeError(f"Mesh attachment exited {process.returncode} without its report")
                summary = json.loads(report_path.read_text(encoding="utf-8"))
                self.register(report_path, "mesh_report", entry["id"])
                self.report["mesh_attachment"] = summary
                rows = summary.get("files", [])
                if process.returncode not in {0, 2} or len(rows) != 1 or rows[0].get("status") == "FAILED":
                    raise RuntimeError(f"Mesh attachment failed (exit {process.returncode}): {rows}")
                mounted = rows[0].get("output")
                if not mounted:
                    raise RuntimeError("Requested mesh attachment produced no mounted FBX; inspect mesh report")
                mounted = Path(mounted).resolve(strict=True)
                self.register(mounted, "mounted_fbx", entry["id"])
                self.current_fbx = mounted
                entry["result"] = rows[0]
                if rows[0].get("status") != "SUCCESS":
                    self.report["status"] = "NEEDS_REVIEW"
                    self.report.setdefault("quality", {})["usable"] = False
                    self.report.setdefault("review_reasons", []).append("mesh_attachment_incomplete")
            finally:
                if owned_job is not None:
                    import ctypes
                    from ctypes import wintypes
                    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
                    kernel32.CloseHandle.restype = wintypes.BOOL
                    if not kernel32.CloseHandle(owned_job):
                        raise ctypes.WinError(ctypes.get_last_error())
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

    def _exports_target(self):
        """Where the single directly-usable (trimmed) FBX belongs:
        03_exports mirroring the source structure; None if not derivable."""
        roots = []
        for value in self.request.get("data_roots", []):
            try:
                roots.append(Path(value).resolve())
            except OSError:
                continue
        source = Path(self.report.get("input_path") or self.request["input_path"])
        root = next((item for item in roots if item == source or item in source.parents), None)
        if root is None:
            return None
        relative = source.relative_to(root)
        parts = relative.parts[1:] if len(relative.parts) > 1 and relative.parts[0].casefold().startswith("01_") else relative.parts
        if not parts:
            return None
        return root / "03_exports" / Path(*parts[:-1]) / (Path(parts[-1]).stem + ".fbx")

    def _lighten(self):
        from fbx_lightweight import lighten_file
        with _stage(self.report, "lightweight_fbx") as entry:
            # FBX graphs parse recursively; the default recursion limit sits at
            # the crash boundary for large scenes (occasional 0xC0000409).
            previous_limit = sys.getrecursionlimit()
            if previous_limit < 60000:
                sys.setrecursionlimit(60000)
            # The trimmed result goes straight to 03_exports (one usable copy,
            # no duplicate display.fbx in the job directory). Display trims
            # never need the auxiliary Labeling*/Solving skeleton trees: the
            # solved characters plus their markers are the usable data.
            output = self._exports_target()
            outside = output is not None
            if output is None:
                output = self.directory / "display.fbx"
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                try:
                    # Re-processing a take replaces its own export copy.
                    if output.exists():
                        output.unlink()
                except OSError:
                    pass
            report_path = self.directory / "lightweight_report.json"
            result = lighten_file(self.current_fbx, output, report_path=report_path,
                                  source_quality_status=self.report["status"],
                                  drop_unused_labeling=self.options["drop_unused_labeling"],
                                  drop_solver_trees=True)
            self.report["lightweight"] = result
            entry["result"] = result
            self.register(output, "export_fbx", entry["id"], archive=outside)
            self.register(report_path, "lightweight_report", entry["id"])
            if not outside:
                self.current_fbx = output

    def run(self):
        self.thread.start()
        try:
            with stage_observer(self):
                if self.source.suffix.lower() == ".fbx":
                    self._fbx()
                else:
                    self._native()
                if self.report["status"] != "FAILED" and self.current_fbx:
                    if self.options["attach_meshes"]:
                        self._attach()
                    if self.options["lightweight"]:
                        self._lighten()
                    self.register(self.current_fbx, "fbx", source=self.current_fbx == self.source)
                # Archive AFTER the display trim exists: 03_exports must get the
                # clean usable FBX, not the full working set.
                if getattr(self, "_native_dir", None) is not None:
                    self._archive_outputs(self._native_dir)
                self._consume_commands()
                self._check_stop()
            final = self.report["status"]
            if final not in {"SUCCESS", "NEEDS_REVIEW", "FAILED"}:
                raise RuntimeError(f"Invalid final processing status: {final}")
        except Cancelled:
            final = "CANCELLED"
            self.report.update(status=final, cancelled_at=utc_now())
        except Exception as exc:
            traceback.print_exc()
            final = "FAILED"
            self.report.update(status=final, error=f"{type(exc).__name__}: {exc}")
            self.report.setdefault("quality", {})["usable"] = False
        finally:
            self.stopped.set()
            self.thread.join(timeout=2)
        for entry in self.report["stages"]:
            if entry["status"] in {"RUNNING", "WAITING"}:
                entry.update(status="SKIPPED" if final == "CANCELLED" else "FAILED",
                             error="Processing cancelled" if final == "CANCELLED" else self.report.get("error"),
                             finished_at=utc_now())
        with self.condition:
            self.state.update(state=final, current_stage=None, waiting_stage=None,
                              error=self.report.get("error") or self.state.get("error"),
                              quality_status=final)
            self._event("finished", state=final)
            self._snapshot(final=True)
        return 1 if final == "FAILED" else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    request_path = args.request.resolve(strict=True)
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    directory = Path(request["job_dir"]).resolve()
    if request_path != directory / "request.json":
        raise ValueError("--request must name this run's job_dir/request.json")
    directory.mkdir(parents=True, exist_ok=True)
    log_path = Path(request.get("log_path") or directory / "processing.log")
    with log_path.open("a", encoding="utf-8", buffering=1) as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            return Worker(request).run()


if __name__ == "__main__":
    raise SystemExit(main())
