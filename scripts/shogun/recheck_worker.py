"""Run each quality recheck in a fresh, windowless Python/SDK process."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import faulthandler
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid

from step1_process_takes import write_json
from pipeline_layout import in_run, report_relative, take_paths


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stop_worker(process: mp.Process) -> None:
    """Stop just this spawned worker/tree, including a CLI still starting up."""
    if process.is_alive():
        if os.name == "nt":
            # Match the pipeline's owned-tree cleanup without opening a console.
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True, check=False, timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            except (OSError, subprocess.SubprocessError):
                pass  # Fall back to the original worker handle below.
        if process.is_alive():
            process.terminate()
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        if process.is_alive():
            raise RuntimeError(f"Unable to stop owned recheck worker PID {process.pid}")
    else:
        process.join()


def _worker(request: dict, attempt: str, temporary: str) -> None:
    """Only this entry point imports and touches the process-global SDK."""
    directory = Path(attempt)
    state = dict(request, worker_pid=os.getpid(), phase="starting")
    session = None
    native_process = None
    exit_code = 1
    with Path(state["logs"]["processing"]).open("a", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                print(f"[recheck-worker] PID={os.getpid()} report={request['report']}", flush=True)
                faulthandler.enable(file=log, all_threads=True)
                write_json(directory / "attempt.json", state)
                from step1_process_takes import free_port
                from pipeline_session import ShogunSession
                from step4_recheck_quality import recheck_report, recover_quality_transition

                session = ShogunSession(
                    request["executable"], Path(temporary),
                    Path(state["logs"]["shogun"]), free_port(set()), None, 90,
                )
                state.update(phase="connecting", control_port=session.port)
                write_json(directory / "attempt.json", state)
                with session:
                    native_process = session._process
                    state.update(phase="processing", native_pid=session.pid, session_ready_at=_now())
                    write_json(directory / "attempt.json", state)
                    operation = recover_quality_transition if request["checkpoint_recovery"] else recheck_report
                    result = operation(session, Path(request["report"]), Path(request["output"]))
                    if result.get("status") not in ("SUCCESS", "NEEDS_REVIEW"):
                        raise RuntimeError(result.get("error") or f"Recheck returned {result.get('status')!r}")
                    summary = {
                        "status": result["status"], "relative_path": result["relative_path"],
                        "usable": result["quality"]["usable"],
                        "checkpoint_recovery": request["checkpoint_recovery"],
                    }
                    state.update(phase="closing")
                    write_json(directory / "attempt.json", state)
                state.update(phase="completed", finished_at=_now())
                response = {"ok": True, "result": summary}
                exit_code = 0
            except BaseException as error:
                details = traceback.format_exc()
                print(details, file=log, flush=True)
                state.update(phase="failed", finished_at=_now(),
                             error=f"{type(error).__name__}: {error}", traceback=details)
                response = {"ok": False, "error": state["error"], "traceback": details}
            finally:
                if session is not None:
                    state["native_pid"] = session.pid
                if native_process is not None:
                    state["native_exitcode"] = native_process.poll()
                write_json(directory / "attempt.json", state)
                write_json(directory / "result.json", response)
                faulthandler.disable()
    raise SystemExit(exit_code)


def run_isolated_recheck(
    report_path: Path, output: Path, executable: str, recovery: bool, timeout: float = 900,
) -> dict:
    """Return a completed summary, or raise with a persistent attempt directory.

    The parent enforces the deadline even if a native SDK call holds the child's
    GIL. Session scratch space is disposable; logs and atomic attempt/results are
    retained in the take's layout-aware recheck bucket on every outcome.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Recheck timeout must be finite and positive")
    report_path = Path(report_path).absolute()
    output = Path(output).resolve()
    relative = report_relative(output, report_path)
    source = json.loads(report_path.read_bytes())
    if Path(source.get("relative_path", "")) != relative:
        raise ValueError("Canonical report location differs from raw relative identity")
    if os.name == "nt":
        worker_python = Path(sys.executable).with_name("pythonw.exe")
        if not worker_python.is_file():
            raise ValueError(f"Windowless worker interpreter missing: {worker_python}")
        mp.set_executable(str(worker_python))
    directory = take_paths(output, relative)["recheck_root"] / uuid.uuid4().hex
    if not in_run(output, directory):
        raise ValueError(f'Recheck attempt destination escapes this run: {directory}')
    directory.mkdir(parents=True)
    state = {
        "report": str(report_path), "output": str(output), "executable": str(executable),
        "checkpoint_recovery": bool(recovery), "timeout_seconds": timeout,
        "parent_pid": os.getpid(), "worker_pid": None, "native_pid": None,
        "native_exitcode": None, "started_at": _now(), "phase": "pending",
        "diagnostic_directory": str(directory),
        "logs": {"processing": str(directory / "processing.log"),
                 "shogun": str(directory / "native.log")},
    }
    write_json(directory / "attempt.json", state)
    for role, log_path in state["logs"].items():
        Path(log_path).write_text(
            f"[recheck-parent] {role} log; report={report_path}; started={state['started_at']}\n",
            encoding="utf-8",
        )
    temporary = None
    process = None
    worker_pid = None
    worker_exitcode = None
    failure = None
    timed_out = False
    interrupted = None
    cleanup_errors = []
    scratch_removed = True
    try:
        temporary = tempfile.mkdtemp(prefix="shogun-quality-")
        scratch_removed = False
        process = mp.get_context("spawn").Process(
            target=_worker, args=(state, str(directory), temporary),
            name="shogun-quality-recheck",
        )
        process.start()
        worker_pid = process.pid
        process.join(timeout)
        if process.is_alive():
            timed_out = True
            failure = f"Recheck exceeded its {timeout:g}s process deadline"
            _stop_worker(process)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        state["parent_traceback"] = traceback.format_exc()
        if not isinstance(error, Exception):
            interrupted = error
    finally:
        if process is not None and process.pid is not None:
            worker_pid = process.pid
            try:
                if process.is_alive():
                    _stop_worker(process)
                worker_exitcode = process.exitcode
            except Exception as error:
                cleanup_errors.append(f"Worker cleanup: {type(error).__name__}: {error}")
            finally:
                try:
                    process.close()
                except Exception as error:
                    cleanup_errors.append(f"Worker handle cleanup: {type(error).__name__}: {error}")
        elif process is not None:
            process.close()
        if temporary is not None:
            try:
                shutil.rmtree(temporary)
                scratch_removed = True
            except Exception as error:
                cleanup_errors.append(f"Session scratch cleanup: {type(error).__name__}: {error}")

    # A worker may have died before it could publish a result or session PID.
    # Read its last atomic state, then add parent-observed exit/deadline facts.
    try:
        state.update(json.loads((directory / "attempt.json").read_text(encoding="utf-8")))
    except Exception as error:
        failure = failure or f"Cannot read worker attempt: {type(error).__name__}: {error}"
    if not state.get("native_pid"):
        native_log = Path(state["logs"]["shogun"])
        if native_log.is_file():
            with native_log.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    match = re.search(r"Owned native PID=(\d+)", line)
                    if match:
                        state["native_pid"] = int(match.group(1))
                        break
    state.update(worker_pid=worker_pid, worker_exitcode=worker_exitcode,
                 finished_at=_now(), timed_out=timed_out, scratch_removed=scratch_removed)
    response = None
    try:
        response = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    except Exception as error:
        failure = failure or f"Worker produced no readable result: {type(error).__name__}: {error}"
    if worker_exitcode != 0:
        failure = failure or (response or {}).get("error") or f"Recheck worker exited with code {worker_exitcode}"
    if response is not None and response.get("ok") is not True:
        failure = failure or response.get("error") or "Recheck worker reported failure"
    if cleanup_errors:
        state["cleanup_errors"] = cleanup_errors
        failure = "; ".join(([failure] if failure else []) + cleanup_errors)
    if failure:
        state.update(phase="timed_out" if timed_out else "failed", error=failure)
        response = {"ok": False, "error": failure, "worker_response": response}
    write_json(directory / "attempt.json", state)
    write_json(directory / "result.json", dict(response or {}, diagnostics=state))
    if interrupted is not None:
        interrupted.add_note(f"Recheck diagnostics: {directory}")
        raise interrupted
    if failure:
        exception = TimeoutError if timed_out else RuntimeError
        raise exception(f"{failure}; worker exit code={worker_exitcode}; diagnostics: {directory}")
    return dict(response["result"], worker_pid=worker_pid, native_pid=state["native_pid"],
                worker_exitcode=worker_exitcode, diagnostic_directory=str(directory), logs=state["logs"])
