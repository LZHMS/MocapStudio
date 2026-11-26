"""CLI orchestration for isolated Shogun take processing and rigid prop checks."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid

from pipeline_run_registry import register_run
from pipeline_layout import in_run, initialize_layout, is_generated, load_layout, take_paths

_STARTUP_CLEANUP_GRACE = 30.0


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def names(value, label: str, *, allow_empty: bool = False) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or (not value and not allow_empty) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} must be a nonempty list of nonempty subject/prop/marker names")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicates")
    return value


def normalize_take_path(key: str) -> str:
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Manifest take keys must be nonempty relative paths")
    key = key.replace("\\", "/")
    if key.startswith("/") or ":" in key or ".." in key.split("/"):
        raise ValueError("Manifest take keys must be paths relative to DataPath's root")
    normalized = PurePosixPath(key).as_posix()
    if normalized == ".":
        raise ValueError("Manifest take keys must name a relative take path")
    return normalized.casefold()


def load_manifest(path: str | None) -> dict:
    if not path:
        return {"expected_subjects": None, "takes": {}, "core_markers": {},
                "expected_props": None, "prop_takes": {}}
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("SubjectManifest must contain an object")
    unknown = set(data) - {"expected_subjects", "takes", "core_markers", "expected_props", "prop_takes"}
    if unknown:
        raise ValueError(f"Unknown SubjectManifest fields: {sorted(unknown)}")
    data["expected_subjects"] = names(data.get("expected_subjects"), "expected_subjects")
    data["expected_props"] = names(data.get("expected_props"), "expected_props", allow_empty=True)
    for field in ("takes", "prop_takes", "core_markers"):
        mapping = data.setdefault(field, {})
        if not isinstance(mapping, dict):
            raise ValueError(f"{field} must contain an object")
        for key, value in mapping.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{field} keys must be nonempty strings")
            names(value, f"{field}.{key}", allow_empty=field == "prop_takes")
            if value is None:
                raise ValueError(f"{field}.{key} must be an explicit non-null list")
        if field in ("takes", "prop_takes"):
            normalized = {}
            for key, value in mapping.items():
                folded = normalize_take_path(key)
                if folded in normalized:
                    raise ValueError(f"Duplicate normalized manifest {field} path: {key}")
                normalized[folded] = value
            data[field] = normalized
    return data


def discover_shogun() -> Path | None:
    candidates: set[Path] = set()
    command = shutil.which("ShogunPostCL.exe")
    if command:
        candidates.add(Path(command))
    for key in ("ProgramFiles", "ProgramFiles(x86)"):
        if os.environ.get(key):
            candidates.update((Path(os.environ[key]) / "Vicon").glob("ShogunPost*/ShogunPostCL.exe"))
    # Shortcuts and running GUI locate nonstandard installations without a drive scan.
    query = r"""$paths=@(); Get-Process ShogunPost -ErrorAction SilentlyContinue | ForEach-Object { if ($_.Path) { $paths += Join-Path (Split-Path $_.Path) 'ShogunPostCL.exe' } }; $shell=New-Object -ComObject WScript.Shell; @([Environment]::GetFolderPath('CommonPrograms'),[Environment]::GetFolderPath('Programs')) | ForEach-Object { $menu=Join-Path $_ 'Vicon'; if(Test-Path -LiteralPath $menu) { Get-ChildItem -LiteralPath $menu -Filter '*.lnk' -File -Recurse | ForEach-Object { $target=$shell.CreateShortcut($_.FullName).TargetPath; if($target -and (Split-Path $target -Leaf) -ieq 'ShogunPost.exe') { $paths += Join-Path (Split-Path $target) 'ShogunPostCL.exe' } } } }; $paths | Sort-Object -Unique | ConvertTo-Json -Compress"""
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", query], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"Shogun discovery failed: {result.stderr.strip()}")
    if result.stdout.strip():
        found = json.loads(result.stdout)
        candidates.update(Path(item) for item in ([found] if isinstance(found, str) else found))
    candidates = {path.resolve() for path in candidates if path.is_file()}
    for path in sorted(candidates):
        print(f"[FOUND] {path}")
    def version(path: Path):
        import re
        return tuple(int(part) for part in re.findall(r"\d+", path.parent.name))
    return max(candidates, key=version) if candidates else None


def inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def dataset_root(data_path: Path) -> Path:
    """Keep group-relative identities when selecting a subgroup or one take."""
    selected = data_path.parent if data_path.is_file() else data_path
    for ancestor in (selected, *selected.parents):
        if ancestor.name.casefold() in ("processed", "_processing"):
            return ancestor.parent
    for ancestor in (selected, *selected.parents):
        if re.fullmatch(r"group[1-9]\d*", ancestor.name, re.IGNORECASE):
            return ancestor.parent
    return selected


def dataset_sequence(relative: Path) -> tuple[int | None, str | None]:
    """Only an immediate groupN/MP|HOI hierarchy establishes a take's mode."""
    parts = relative.parts
    group = re.fullmatch(r"group([1-9]\d*)", parts[0], re.IGNORECASE) if parts else None
    number = int(group[1]) if group else None
    category = parts[1].upper() if len(parts) >= 3 else None
    return number, category if group and category in ("MP", "HOI") else None


def dataset_order(source: Path, root: Path) -> tuple:
    relative = source.relative_to(root)
    group, category = dataset_sequence(relative)
    return (group if group is not None else math.inf, {"MP": 0, "HOI": 1}.get(category, 2),
            relative.as_posix().casefold(), relative.as_posix())


def build_jobs(data_path: Path, output: Path, logs: Path, input_type: str, manifest: dict,
               expected: list[str] | None, repair_subject: str | None, overwrite: bool,
               expected_props: list[str] | None = None, dataset_layout: bool = False,
               input_list: str | Path | None = None) -> list[dict]:
    """Plan inventory or only InputList's exact MCP/VDF paths (never both).

    InputList names a JSON file; relative entries resolve under DataPath's
    directory, while DatasetLayout identities still use the full dataset root.
    """
    data_path, output, logs = data_path.resolve(), output.resolve(), logs.resolve()
    if dataset_layout and repair_subject:
        raise ValueError("DatasetLayout cannot be combined with RepairSubject; every human must be processed and split")
    for bucket in ("takes", "_needs_review", ".work", "_failed"):
        protected = output / bucket
        if inside(logs, protected) or inside(protected, logs):
            raise ValueError("LogDir must be separate from output artifacts/workspaces; use a dedicated logs directory")
    if not data_path.exists():
        raise ValueError(f"DataPath not found: {data_path}")
    if data_path.is_file():
        if data_path.suffix.lower() not in (".mcp", ".vdf"):
            raise ValueError("DataPath must be a .mcp/.vdf file or a directory")
        root, inputs = data_path.parent, [data_path]
    else:
        if repair_subject:
            raise ValueError("RepairSubject requires a single VDF file")
        root = data_path
        inputs = [] if dataset_layout or input_list is not None else sorted(
            path.resolve() for path in root.rglob("*" + "." + input_type)
            if path.is_file() and not inside(path.resolve(), output))
    if dataset_layout:
        root = dataset_root(data_path)
        controls = (root / "_processing").resolve()
        if inside(output, root) and (output == controls or not inside(output, controls)):
            raise ValueError("DatasetLayout OutputDir within the dataset must be below _processing")
        if inside(logs, root) and not inside(logs, controls):
            raise ValueError("DatasetLayout LogDir within the dataset must be below _processing")
        if is_generated(data_path, root):
            raise ValueError("DatasetLayout DataPath must be raw input, not processed/_processing")
        if data_path.is_dir() and input_list is None:
            inputs = [path.resolve() for path in data_path.rglob("*")
                      if path.is_file() and path.suffix.lower() == "." + input_type
                      and not is_generated(path, root) and not is_generated(path.resolve(), root)]
        inputs.sort(key=lambda source: dataset_order(source, root))
    if input_list is not None:
        selected = json.loads(Path(input_list).expanduser().read_text(encoding="utf-8-sig"))
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(item, str) or not item.strip() for item in selected)):
            raise ValueError("InputList must contain a nonempty JSON array of take paths")
        selection_root = data_path.parent if data_path.is_file() else data_path
        generated_root = dataset_root(data_path)
        inputs = []
        seen: set[str] = set()
        for value in selected:
            path = Path(value.replace("\\", "/")).expanduser()
            path = path if path.is_absolute() else selection_root / path
            source = path.resolve()
            if not inside(source, selection_root) or (data_path.is_file() and source != data_path):
                raise ValueError(f"InputList path is outside DataPath: {value}")
            if source.suffix.lower() not in (".mcp", ".vdf"):
                raise ValueError(f"InputList path must be an MCP or VDF take: {value}")
            if (is_generated(path, generated_root) or is_generated(source, generated_root)
                    or inside(source, output) or inside(source, logs)):
                raise ValueError(f"InputList path is generated output, not raw input: {value}")
            if not source.is_file():
                raise ValueError(f"InputList take does not exist or is not a file: {value}")
            identity = str(source).casefold()
            if identity in seen:
                raise ValueError(f"Duplicate InputList take: {value}")
            seen.add(identity)
            inputs.append(source)
        if dataset_layout:
            inputs.sort(key=lambda source: dataset_order(source, root))
    if not inputs:
        raise ValueError(f"No {input_type.upper()} inputs found: {data_path}")
    jobs = []
    destinations: set[str] = set()
    for source in inputs:
        relative = source.relative_to(root)
        paths = take_paths(output, relative, data_root=root if dataset_layout else None)
        job_dir = paths["take_root"]
        destination = paths["destination"]
        review_destination = paths["review_destination"]
        log_dir = paths["log_dir"] if dataset_layout or load_layout(output) else logs / relative
        key = hashlib.sha256(relative.as_posix().casefold().encode("utf-8")).hexdigest()[:16]
        take_key = normalize_take_path(relative.as_posix())
        roster = manifest["takes"].get(take_key, manifest["expected_subjects"])
        if expected is not None:
            roster = expected
        props = manifest.get("prop_takes", {}).get(take_key, manifest.get("expected_props"))
        if expected_props is not None:
            props = expected_props
        errors = []
        kind = source.suffix.lower()[1:]
        sequence_kind = None
        if dataset_layout:
            _, sequence_kind = dataset_sequence(relative)
            if sequence_kind is None:
                errors.append(f"Unclassified DatasetLayout input; expected groupN/MP|HOI/<take>: {relative.as_posix()}")
        if repair_subject and kind != "vdf":
            errors.append("RepairSubject requires VDF input; MCP would reconstruct the scene")
        if source.stat().st_size == 0:
            errors.append(f"Empty input: {source}")
        if kind == "mcp":
            x2d = source.with_suffix(".x2d")
            if not x2d.is_file() or x2d.stat().st_size == 0:
                errors.append(f"Missing or empty same-name X2D: {x2d}")
        if output == root or destination.resolve() == source.parent:
            errors.append("Output directory must not be the source directory, even with Overwrite")
        identity = str(job_dir).casefold()
        if identity in destinations:
            errors.append(f"Output collision: {destination / source.stem}")
        destinations.add(identity)
        report_path = job_dir / "report.json"
        if not overwrite:
            if report_path.exists():
                errors.append(f"Output exists (use Overwrite or another OutputDir): {report_path}")
            for parent in (destination, review_destination):
                if dataset_layout:
                    # Subject split names are only known after native loading.
                    # Existing nested artifacts still make a no-overwrite run unsafe.
                    for target in sorted(parent.rglob("*")):
                        if target.is_file():
                            errors.append(f"Output exists (use Overwrite or another OutputDir): {target}")
                else:
                    for suffix in (".vdf", ".fbx", ".vsk", ".json"):
                        target = parent / (source.stem + suffix)
                        if target.exists():
                            errors.append(f"Output exists (use Overwrite or another OutputDir): {target}")
        jobs.append({"job_id": key, "input_path": str(source), "input_kind": kind,
                     "relative_path": relative.as_posix(), "expected_subjects": roster, "expected_props": props,
                     "destination": str(destination), "review_destination": str(review_destination),
                     "report_path": str(report_path), "log_path": str(log_dir / "processing.log"),
                     "native_log_path": str(log_dir / "shogun.log"), "preflight_errors": errors})
        if dataset_layout:
            expected_count = 2
            if roster is not None and len(roster) == 1:
                expected_count = 1
            jobs[-1].update(dataset_mode=True, sequence_kind=sequence_kind, split_subjects=True,
                            props_enabled=sequence_kind == "HOI", expected_human_count=expected_count,
                            layout_data_root=str(root))
    return jobs


def worker(job: dict, stage: str, executable: str, port: int, profile: str | None,
           startup_timeout: float, policy: dict) -> None:
    stage_path = Path(stage)
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "FAILED", "subjects": [], "props": [], "stages": [], "artifacts": {},
              "quality": {"schema_version": 2, "basis": "post_processing", "usable": False}}
    with log_path.open("w", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            from pipeline_session import ShogunSession
            from pipeline_processing import process_take
            with ShogunSession(executable, stage_path, job["native_log_path"], port, profile, startup_timeout) as session:
                session_info = {"pid": session.pid, "port": session.port, "version": session.version,
                                "started_at": datetime.now(timezone.utc).isoformat(),
                                "job_id": job.get("job_id"), "input_path": job.get("input_path"),
                                "policy": policy}
                write_json(stage_path / ".control" / "session-ready.json", session_info)
                try:
                    report = process_take(session, job, stage_path, policy)
                finally:
                    session_info["finished_at"] = datetime.now(timezone.utc).isoformat()
                    report["session"] = session_info
        except BaseException as error:
            traceback.print_exc()
            if report.get("status") == "FAILED" and report.get("error"):
                report["cleanup_error"] = f"{type(error).__name__}: {error}"
            else:
                report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            report["quality"]["usable"] = False
        report.update(schema_version=1, job_id=job["job_id"], input_path=job["input_path"],
                      relative_path=job["relative_path"], input_kind=job["input_kind"],
                      expected_subjects=job["expected_subjects"], expected_props=job.get("expected_props"),
                      policy={**policy, **report.get("policy", {})}, identity_validated=False,
                      logs={"processing": job["log_path"], "shogun": job["native_log_path"]})
        if job.get("dataset_mode"):
            report.update({field: job[field] for field in
                           ("dataset_mode", "sequence_kind", "split_subjects", "props_enabled", "expected_human_count")})
        if job.get("review_repair"):
            for field in ("review_repair", "source_output", "source_report", "source_report_sha256",
                          "checkpoint_fingerprint", "checkpoint_completion", "roster_resolution",
                          "confirmed_subject_roster", "priority"):
                if field in job:
                    report[field] = job[field]
        write_json(stage_path / ".control" / "result.json", report)


def free_port(reserved: set[int]) -> int:
    for _ in range(100):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        if port not in reserved:
            return port
    raise RuntimeError("Unable to allocate a distinct control port")


def publish(job: dict, stage: Path, output: Path, overwrite: bool) -> dict:
    result_path = stage / ".control" / "result.json"
    if result_path.is_file():
        report = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        report = {"status": "FAILED", "error": "Worker exited without a result report", "artifacts": {},
                  "job_id": job["job_id"], "input_path": job["input_path"], "subjects": [], "props": [],
                  "relative_path": job["relative_path"], "input_kind": job["input_kind"],
                  "expected_subjects": job["expected_subjects"], "expected_props": job.get("expected_props")}
    status = report.get("status")
    if status not in ("SUCCESS", "NEEDS_REVIEW", "FAILED"):
        report.update(status="FAILED", error=f"Invalid worker status: {status}")
    try:
        if report["status"] != "FAILED":
            products = report.get("artifacts", {})
            required = {"vdf", "fbx", "vsk"} if job["input_kind"] == "mcp" else {"vdf", "fbx"}
            if not required <= products.keys():
                raise ValueError(f"Missing fresh native artifacts: {sorted(required - products.keys())}")
            expected_suffixes = {"vdf": ".vdf", "fbx": ".fbx", "vsk": ".vsk"}
            for role, relative in products.items():
                if not isinstance(relative, str):
                    raise ValueError(f"Artifact {role} must be a relative file path")
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(f"Artifact escapes staging: {relative}")
                if role in expected_suffixes and relative_path.suffix.lower() != expected_suffixes[role]:
                    raise ValueError(f"Unexpected artifact type {role}: {relative}")
                path = (stage / relative_path).resolve()
                if not inside(path, stage.resolve()) or path == stage.resolve() or not path.is_file() or not path.stat().st_size:
                    raise ValueError(f"Invalid or missing fresh artifact {role}: {relative}")
            target_root = Path(job["destination"] if report["status"] == "SUCCESS" else job["review_destination"])
            target_root.mkdir(parents=True, exist_ok=True)
            for relative in products.values():
                if (target_root / relative).exists() and not overwrite:
                    raise FileExistsError(f"Output appeared while processing: {target_root / relative}")
            published = {}
            for role, relative in products.items():
                target = target_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if overwrite:
                    os.replace(stage / relative, target)
                else:
                    # On Windows rename refuses to replace a target created concurrently.
                    os.rename(stage / relative, target)
                published[role] = str(target)
            report["artifacts"] = published
            for entry in report.get("subject_exports", []):
                entry["path"] = published[entry["artifact_role"]]
            # CL may generate a same-stem timecode sidecar. It is not a success prerequisite.
            sidecar = stage / (Path(job["input_path"]).stem + ".json")
            if sidecar.is_file():
                sidecar_target = target_root / sidecar.name
                if not sidecar_target.exists() or overwrite:
                    os.replace(sidecar, sidecar_target)
                    report["artifacts"]["timecode"] = str(sidecar_target)
    except Exception as error:
        report.update(status="FAILED", error=f"Publish failed: {error}")
    if report["status"] == "FAILED":
        if "quality" in report:
            report["quality"]["usable"] = False
        for entry in report.get("subject_exports", []):
            if "quality" in entry:
                entry["quality"]["usable"] = False
        failed = take_paths(output, job["relative_path"], data_root=job.get("layout_data_root"))["failed_root"] / (job["job_id"] + "-" + uuid.uuid4().hex[:8])
        failed.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage, failed)
        report["diagnostic_directory"] = str(failed)
        diagnostic_artifacts = {}
        for role, relative in report.get("artifacts", {}).items():
            if isinstance(relative, str) and not Path(relative).is_absolute() and ".." not in Path(relative).parts:
                candidate = failed / relative
                if candidate.is_file():
                    diagnostic_artifacts[role] = str(candidate)
        report["artifacts"] = diagnostic_artifacts
    else:
        shutil.rmtree(stage)
    if job.get("dataset_mode"):
        report.update({field: job[field] for field in
                       ("dataset_mode", "sequence_kind", "split_subjects", "props_enabled", "expected_human_count")})
    write_json(Path(job["report_path"]), report)
    return report


def terminate_owned(process: mp.Process, close: bool = True) -> None:
    if not process.is_alive():
        process.join()
        if close:
            process.close()
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        process.terminate()
    process.join(timeout=10)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    if process.is_alive():
        raise RuntimeError(f"Unable to stop owned worker PID {process.pid}")
    if close:
        process.close()


def run_jobs(jobs: list[dict], output: Path, executable: Path, max_workers: int,
             profile: str | None, startup_timeout: float, policy: dict, overwrite: bool) -> list[dict]:
    output = output.resolve()
    if os.name == "nt":
        worker_python = Path(sys.executable).with_name("pythonw.exe")
        if not worker_python.is_file():
            raise ValueError(f"Windowless worker interpreter missing: {worker_python}")
        mp.set_executable(str(worker_python))
    context = mp.get_context("spawn")
    pending = iter(job for job in jobs if not job["preflight_errors"])
    results = []
    active: dict[int, tuple[mp.Process, dict, Path, int]] = {}
    startup_deadlines: dict[int, float] = {}
    for job in jobs:
        if job["preflight_errors"]:
            result = {"job_id": job["job_id"], "input_path": job["input_path"], "status": "FAILED",
                      "error": "; ".join(job["preflight_errors"]), "artifacts": {}, "subjects": [], "props": [],
                      "expected_subjects": job["expected_subjects"], "expected_props": job.get("expected_props")}
            result.update(relative_path=job["relative_path"], input_kind=job["input_kind"])
            if job.get("dataset_mode"):
                result.update({field: job[field] for field in
                               ("dataset_mode", "sequence_kind", "split_subjects", "props_enabled", "expected_human_count")})
            results.append(result)
            report_path = Path(job["report_path"])
            if not report_path.exists() or overwrite:
                write_json(report_path, result)
            print(f"[FAILED] {job['input_path']}: {result['error']}", flush=True)
    exhausted = False
    try:
        while active or not exhausted:
            while len(active) < max_workers and not exhausted:
                job = next(pending, None)
                if job is None:
                    exhausted = True
                    break
                work_root = take_paths(output, job["relative_path"], data_root=job.get("layout_data_root"))["work_root"]
                stage = work_root / (job["job_id"] + "-" + uuid.uuid4().hex)
                stage.mkdir(parents=True)
                port = free_port({entry[3] for entry in active.values()})
                process = context.Process(target=worker, args=(job, str(stage), str(executable), port,
                                                               profile, startup_timeout, policy), name="shogun-" + job["job_id"])
                process.start()
                active[process.sentinel] = (process, job, stage, port)
                startup_deadlines[process.sentinel] = time.monotonic() + startup_timeout + _STARTUP_CLEANUP_GRACE
                print(f"[RUN] worker={process.pid} port={port} {job['input_path']}", flush=True)
            if not active:
                continue
            completed = list(wait(list(active), timeout=0.5))
            for sentinel, (process, job, stage, _) in list(active.items()):
                if sentinel not in completed and not (stage / ".control" / "session-ready.json").is_file() and time.monotonic() > startup_deadlines[sentinel]:
                    # A native SDK call may not release Python's GIL. The parent
                    # can still enforce startup bounds by stopping this owned tree.
                    terminate_owned(process, close=False)
                    write_json(stage / ".control" / "result.json", {
                        "status": "FAILED", "job_id": job["job_id"], "input_path": job["input_path"],
                        "relative_path": job["relative_path"], "input_kind": job["input_kind"],
                        "error": f"Worker did not acknowledge SDK readiness within {startup_timeout}s plus cleanup grace",
                        "artifacts": {}, "subjects": [], "props": [],
                        "expected_subjects": job["expected_subjects"], "expected_props": job.get("expected_props"),
                        "logs": {"processing": job["log_path"], "shogun": job["native_log_path"]},
                    })
                    completed.append(sentinel)
            for sentinel in completed:
                process, job, stage, _ = active.pop(sentinel)
                startup_deadlines.pop(sentinel, None)
                process.join()
                try:
                    report = publish(job, stage, output, overwrite)
                finally:
                    process.close()
                results.append(report)
                print(f"[{report['status']}] {job['input_path']} -> {job['report_path']}", flush=True)
                if report.get("error"):
                    print(f"  {report['error']}", flush=True)
    finally:
        for process, _, _, _ in active.values():
            terminate_owned(process)
        for work_root in {take_paths(output, job["relative_path"], data_root=job.get("layout_data_root"))["work_root"] for job in jobs}:
            if work_root.exists() and not any(work_root.iterdir()):
                work_root.rmdir()
    return results


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Reconstruct, inspect, conditionally repair and solve isolated Shogun takes.")
    for name in ("DataPath", "OutputDir", "LogDir", "ShogunCL", "RepairSubject", "ExpectedSubjects", "ExpectedProps", "SubjectManifest", "Profile", "InputList"):
        result.add_argument("-" + name)
    result.add_argument("-MaxWorkers", type=int, default=None)
    result.add_argument("-InputType", type=str.lower, choices=("mcp", "vdf"), default="mcp")
    result.add_argument("-MinCoverage", type=float, default=1.0)
    result.add_argument("-PropTolerance", type=float, default=2.0)
    result.add_argument("-PropMaxStep", type=float, default=200.0)
    result.add_argument("-PropJitterMaxCorrection", type=float, default=5.0)
    result.add_argument("-PropJitterMaxGap", type=int, default=5)
    result.add_argument("-PropJitterWindow", type=int, default=3)
    result.add_argument("-StartupTimeout", type=float, default=90.0)
    for name in ("IgnoreLabelingClusters", "CheckOnly", "DryRun", "Overwrite", "Resume", "DatasetLayout", "FilterMarkers"):
        result.add_argument("-" + name, action="store_true")
    result.add_argument("-Help", action="help")
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        options = parser().parse_args(argv)
    except SystemExit as exc:
        # argparse's default usage-error code 2 is reserved for NEEDS_REVIEW.
        return 0 if exc.code == 0 else 1
    if options.MaxWorkers is None:
        options.MaxWorkers = 1 if options.DatasetLayout else 2
    if options.MaxWorkers < 1:
        raise ValueError("MaxWorkers must be positive")
    if not 0 < options.MinCoverage <= 1:
        raise ValueError("MinCoverage must be greater than 0 and at most 1")
    if options.StartupTimeout <= 0:
        raise ValueError("StartupTimeout must be positive")
    if not math.isfinite(options.PropTolerance) or options.PropTolerance <= 0:
        raise ValueError("PropTolerance must be positive and finite (mm)")
    if not math.isfinite(options.PropMaxStep) or options.PropMaxStep <= 0:
        raise ValueError("PropMaxStep must be positive and finite (mm/frame)")
    if not math.isfinite(options.PropJitterMaxCorrection) or options.PropJitterMaxCorrection < 0:
        raise ValueError("PropJitterMaxCorrection must be nonnegative and finite (mm)")
    if options.PropJitterMaxGap <= 0 or options.PropJitterWindow <= 0:
        raise ValueError("PropJitterMaxGap and PropJitterWindow must be positive frame counts")
    if options.Resume and options.Overwrite:
        raise ValueError("Use Resume or Overwrite, not both")
    if options.CheckOnly and options.DryRun:
        raise ValueError("Use CheckOnly or DryRun, not both")
    if options.DatasetLayout and options.RepairSubject:
        raise ValueError("DatasetLayout cannot be combined with RepairSubject; every human must be processed and split")
    if options.ExpectedSubjects is not None and options.SubjectManifest:
        raise ValueError("Use ExpectedSubjects or SubjectManifest, not both")
    if options.ExpectedProps is not None and options.SubjectManifest:
        raise ValueError("Use ExpectedProps or SubjectManifest, not both")
    if options.RepairSubject and (options.ExpectedSubjects is not None or options.SubjectManifest):
        raise ValueError("RepairSubject cannot be combined with an expected-subject roster")
    if options.RepairSubject and options.ExpectedProps is not None:
        raise ValueError("RepairSubject cannot be combined with ExpectedProps")
    if options.InputList is not None and not options.DataPath:
        raise ValueError("InputList requires DataPath")
    expected = names([item.strip() for item in options.ExpectedSubjects.split(",")], "ExpectedSubjects") if options.ExpectedSubjects is not None else None
    expected_props = names([item.strip() for item in options.ExpectedProps.split(",")], "ExpectedProps") if options.ExpectedProps is not None else None
    manifest = load_manifest(options.SubjectManifest)
    executable = None
    # Dataset inventory/planning must be usable without a licensed native install.
    if not (options.DatasetLayout and options.DryRun):
        executable = Path(options.ShogunCL).expanduser().resolve() if options.ShogunCL else discover_shogun()
        if not executable or not executable.is_file() or executable.name.lower() != "shogunpostcl.exe":
            raise ValueError("ShogunPostCL.exe not found; pass ShogunCL explicitly")
        sdk = executable.parent / "SDK" / "Win64" / "vicon_shogun_post.py"
        if not sdk.is_file():
            raise ValueError(f"Shogun SDK missing: {sdk}")
        if options.Profile:
            options.Profile = str(Path(options.Profile).expanduser().resolve())
            if not Path(options.Profile).is_file():
                raise ValueError(f"Profile not found: {options.Profile}")
        from pipeline_session import check_sdk
        check_sdk(str(executable))
        import numpy as np
        print(f"[OK] NumPy {np.__version__} (rigid prop geometry)")
        print(f"[OK] Python {sys.version.split()[0]}; SDK={sdk.parent}; Shogun={executable}")
        print("[INFO] Workers use separate SDK connections/control ports; licence and execution require a real run.")
    if not options.DataPath:
        if not options.CheckOnly:
            raise ValueError("Pass DataPath <directory|take.mcp|take.vdf>")
        print("[OK] Tools checked; no Shogun processes started and no files written.")
        return 0
    data = Path(options.DataPath).expanduser().resolve()
    root = dataset_root(data) if options.DatasetLayout else data.parent if data.is_file() else data
    vdf_mode = options.RepairSubject or (data.is_file() and data.suffix.lower() == ".vdf") or (data.is_dir() and options.InputType == "vdf")
    default_output = root / "_processing" / "processing" if options.DatasetLayout else root / ("repaired_shogun" if vdf_mode else "processed_shogun")
    output = Path(options.OutputDir).expanduser().resolve() if options.OutputDir else default_output
    if output.resolve() == root:
        raise ValueError("OutputDir must be separate from the source directory, even with Overwrite")
    logs = Path(options.LogDir).expanduser().resolve() if options.LogDir else output / "logs"
    build_overwrite = True if options.Resume else options.Overwrite
    jobs = build_jobs(data, output, logs, options.InputType, manifest, expected, options.RepairSubject,
                      build_overwrite, expected_props, options.DatasetLayout, input_list=options.InputList)
    selection = None
    if options.Resume:
        from pipeline_resume import select_resume_jobs
        selection = select_resume_jobs(jobs, output)
        print(f"[RESUME] total={len(jobs)} preserved={len(selection['preserved'])} pending={len(selection['pending'])} "
              f"retries={len(selection['retry_reports'])} recoverable={len(selection['recoverable'])}")
    for job in (selection["pending"] if selection else jobs):
        print(f"[PLAN] {job['input_path']} -> {job['destination']}; "
              f"expected_subjects={job['expected_subjects']}; expected_props={job['expected_props']}"
              + (f"; sequence_kind={job['sequence_kind']}; props_enabled={job['props_enabled']}; split_subjects=True"
                 if options.DatasetLayout else ""))
        for error in job["preflight_errors"]:
            print(f"[ERROR] {error}")
    print(f"[PLAN] inputs={len(selection['pending'] if selection else jobs)} max_workers={options.MaxWorkers} min_coverage={options.MinCoverage}; "
          f"prop_tolerance={options.PropTolerance}mm prop_max_step={options.PropMaxStep}mm/frame; "
          f"prop_jitter_max_correction={options.PropJitterMaxCorrection}mm "
          f"prop_jitter_max_gap={options.PropJitterMaxGap}frames prop_jitter_window={options.PropJitterWindow}frames; "
          f"filter_markers={options.FilterMarkers}; output={output}")
    if options.RepairSubject:
        print("[INFO] RepairSubject is human target-only: prop detection, repair and quality gates are skipped.")
    if options.DatasetLayout:
        print("[INFO] DatasetLayout: MP is human-only; HOI includes rigid props; every take splits all humans.")
        print("[INFO] Explicit InputList selects only listed takes; otherwise extension-only inventory ignores .capture and backup extensions.")
    if options.CheckOnly or options.DryRun:
        print("[INFO] No files written; no Shogun process started.")
        active_jobs = selection["pending"] if selection else jobs
        return 1 if any(job["preflight_errors"] for job in active_jobs) else 0
    output.mkdir(parents=True, exist_ok=True)
    if options.DatasetLayout:
        initialize_layout(output, root)
    lock_path = output / ".pipeline.lock"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError(f"Another run or stale lock exists: {lock_path}. Inspect its owning PID before removing it.") from None
    try:
        with lock:
            json.dump({"pid": os.getpid(), "run_id": run_id}, lock)
            lock.flush()
            register_run(output, kind="human_target" if options.RepairSubject else "processing",
                         run_id=run_id, data_path=data, planned_total=len(jobs),
                         subject=options.RepairSubject,
                         input_path=data if options.RepairSubject and data.is_file() else None)
            dispatch_jobs = jobs
            if options.Resume:
                from pipeline_resume import select_resume_jobs
                selection = select_resume_jobs(jobs, output)
                if selection["recoverable"]:
                    for rec_job, rec_stage in selection["recoverable"]:
                        publish(rec_job, rec_stage, output, False)
                    selection = select_resume_jobs(jobs, output)
                if selection["retry_reports"]:
                    for item in selection["retry_reports"]:
                        rep_p = Path(item["report_path"])
                        matching = next(job for job in jobs if job["job_id"] == item["job_id"])
                        paths = take_paths(output, matching["relative_path"])
                        relative = rep_p.relative_to(paths["take_root"]) if load_layout(output) else rep_p.relative_to(output)
                        write_json(paths["history_root"] / run_id / relative, item["report_data"])
                        sidecar = rep_p.with_name("independent_recheck.json")
                        if sidecar.is_file():
                            if not in_run(output, sidecar):
                                raise ValueError("Independent recheck sidecar escapes this run")
                            shutil.copy2(sidecar, paths["history_root"] / run_id / relative.with_name(sidecar.name))
                dispatch_jobs = selection["pending"]
            policy = {"repair_subject": options.RepairSubject, "ignore_labeling_clusters": options.IgnoreLabelingClusters,
                      "min_coverage": options.MinCoverage, "core_markers": manifest["core_markers"],
                      "prop_tolerance": options.PropTolerance, "prop_max_step": options.PropMaxStep,
                      "prop_jitter_max_correction": options.PropJitterMaxCorrection,
                      "prop_jitter_max_gap": options.PropJitterMaxGap,
                      "prop_jitter_window": options.PropJitterWindow,
                      "dataset_layout": options.DatasetLayout, "filter_markers": options.FilterMarkers}
            results = run_jobs(dispatch_jobs, output, executable, options.MaxWorkers, options.Profile,
                               options.StartupTimeout, policy, False if options.Resume else options.Overwrite)
            counts = {status: sum(row["status"] == status for row in results) for status in ("SUCCESS", "NEEDS_REVIEW", "FAILED")}
            summary = {"schema_version": 1, "run_id": run_id, "counts": counts, "jobs": results,
                       "identity_validated": False, "shogun": str(executable),
                       "max_workers": options.MaxWorkers, "policy": policy}
            if options.Resume:
                summary["resume"] = {
                    "preserved_count": len(selection["preserved"]),
                    "retry_count": len(selection["retry_reports"]),
                    "pending_count": len(dispatch_jobs),
                }
            write_json(output / ("batch-" + run_id + ".json"), summary)
            write_json(output / "batch_report.json", summary)
            if options.Resume:
                print(f"[SUMMARY] newly_processed: {' '.join(f'{key}={value}' for key, value in counts.items())}; preserved={len(selection['preserved'])}")
            else:
                print("[SUMMARY] " + " ".join(f"{key}={value}" for key, value in counts.items()))
            return 1 if counts["FAILED"] else 2 if counts["NEEDS_REVIEW"] else 0
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    mp.freeze_support()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[INTERRUPTED] Owned workers stopped; inspect retained work directories before restarting.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
