"""Own one Shogun Post CL and its process-global SDK connection per worker.

Importing this module never imports the native SDK or starts an application.
The SDK's DLL search handles intentionally survive individual sessions: Python
cannot unload its extension modules safely between jobs in the same process.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import importlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType
from typing import Any


_SESSION_LOCK = threading.Lock()
_SDK_LOCK = threading.Lock()
_SDK_MODULE: ModuleType | None = None
_SDK_DIRECTORY: Path | None = None
_DLL_HANDLES: list[Any] = []
_SDK_POISONED: str | None = None


class ShogunSessionError(RuntimeError):
    """A native-session prerequisite, startup, ownership, or cleanup failure."""


def _file_fingerprint(path: Path) -> dict:
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino):
        raise ShogunSessionError(f"Native load source changed while hashing: {path}")
    return {"size": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest}


def load_native_file(session, path: str | Path, *, native_call=None, audit=None) -> dict:
    """Load MCP in place; load VDF from a verified session-owned local snapshot.

    Post 1.21 rejected a published VDF path but loaded its identical bytes at a
    short local path. This chooses the compatible path before the first request;
    it does not retry SDK errors or assume a particular native filename limit.
    Snapshots stay alive until the owning native session has stopped.
    """
    source = Path(path).resolve(strict=True)
    kind = source.suffix.lower()
    if kind not in {".mcp", ".vdf"} or not source.is_file():
        raise ValueError(f"Native scene source must be an MCP or VDF file: {source}")
    entry = {"source_path": str(source), "source_size": source.stat().st_size,
             "method": "verified_local_vdf_snapshot" if kind == ".vdf" else "original_mcp",
             "status": "PREPARING"}
    if audit is not None:
        audit.append(entry)
    loads = getattr(session, "native_load_audit", None)
    if loads is None:
        loads = session.native_load_audit = []
    if loads is not audit:
        loads.append(entry)

    def record():
        note = getattr(session, "_note", None)
        if note is not None:
            note("Native load " + json.dumps(entry, ensure_ascii=False, allow_nan=False))

    def invoke(load_path):
        if native_call is None:
            session.api.LoadFile(load_path.as_posix())
        else:
            native_call("LoadFile", session.api.LoadFile, load_path.as_posix())

    record()
    try:
        if kind == ".mcp":
            entry["status"] = "LOADING"
            record()
            invoke(source)
        else:
            before = _file_fingerprint(source)
            entry["source_before"] = before
            directory = getattr(session, "_native_load_directory", None)
            if directory is None:
                directory = session._native_load_directory = tempfile.TemporaryDirectory(prefix="sv_")
            snapshot_dir = Path(directory.name) / str(len(loads))
            snapshot_dir.mkdir()
            snapshot = snapshot_dir / "input.vdf"
            entry["snapshot_path"] = str(snapshot)
            shutil.copyfile(source, snapshot)
            copied = _file_fingerprint(snapshot)
            entry["snapshot_before"] = copied
            entry["source_after_copy"] = _file_fingerprint(source)
            if (entry["source_after_copy"] != before
                    or (copied["size"], copied["sha256"]) != (before["size"], before["sha256"])):
                raise ShogunSessionError(f"Native VDF source changed or snapshot differs: {source}")
            entry["status"] = "LOADING"
            record()
            try:
                invoke(snapshot)
            finally:
                entry["source_after"] = _file_fingerprint(source)
                entry["snapshot_after"] = _file_fingerprint(snapshot)
                if (entry["source_after"] != before or entry["snapshot_after"] != copied):
                    raise ShogunSessionError(f"Native VDF source or snapshot changed during load: {source}")
        entry["status"] = "SUCCESS"
    except BaseException as error:
        entry.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        record()
        raise
    record()
    return entry


def _executable_details(executable: str) -> tuple[Path, str]:
    if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise ShogunSessionError("Shogun Post requires 64-bit Windows and Python.")
    exe = Path(executable).expanduser().resolve(strict=True)
    if not exe.is_file() or exe.name.lower() != "shogunpostcl.exe":
        raise ShogunSessionError(f"Executable must be ShogunPostCL.exe: {exe}")

    class FixedInfo(ctypes.Structure):
        _fields_ = [(name, wintypes.DWORD) for name in (
            "signature", "structure_version", "file_ms", "file_ls",
            "product_ms", "product_ls", "flags_mask", "flags", "os",
            "file_type", "subtype", "date_ms", "date_ls",
        )]

    version_dll = ctypes.WinDLL("version", use_last_error=True)
    version_dll.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    version_dll.GetFileVersionInfoSizeW.restype = wintypes.DWORD
    version_dll.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    version_dll.GetFileVersionInfoW.restype = wintypes.BOOL
    version_dll.VerQueryValueW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.UINT)]
    version_dll.VerQueryValueW.restype = wintypes.BOOL
    ignored = wintypes.DWORD()
    size = version_dll.GetFileVersionInfoSizeW(str(exe), ctypes.byref(ignored))
    if size:
        data = ctypes.create_string_buffer(size)
        pointer = ctypes.c_void_p()
        length = wintypes.UINT()
        if (version_dll.GetFileVersionInfoW(str(exe), 0, size, data)
                and version_dll.VerQueryValueW(data, "\\", ctypes.byref(pointer), ctypes.byref(length))
                and length.value >= ctypes.sizeof(FixedInfo)):
            translations = ctypes.c_void_p()
            translation_size = wintypes.UINT()
            if version_dll.VerQueryValueW(data, "\\VarFileInfo\\Translation", ctypes.byref(translations), ctypes.byref(translation_size)):
                pairs = ctypes.cast(translations, ctypes.POINTER(ctypes.c_ushort))
                for index in range(0, translation_size.value // 2 - 1, 2):
                    field = f"\\StringFileInfo\\{pairs[index]:04x}{pairs[index + 1]:04x}\\ProductVersion"
                    text_pointer = ctypes.c_void_p()
                    text_size = wintypes.UINT()
                    if version_dll.VerQueryValueW(data, field, ctypes.byref(text_pointer), ctypes.byref(text_size)) and text_size.value:
                        product_version = ctypes.wstring_at(text_pointer).strip()
                        if re.match(r"^\d+\.\d+", product_version):
                            return exe, product_version
            info = ctypes.cast(pointer, ctypes.POINTER(FixedInfo)).contents
            if info.signature == 0xFEEF04BD:
                ms, ls = info.product_ms, info.product_ls
                return exe, f"{ms >> 16}.{ms & 65535}.{ls >> 16}.{ls & 65535}"
    directory_version = re.fullmatch(r"ShogunPost(\d+\.\d+(?:\.\d+)*)", exe.parent.name, re.IGNORECASE)
    if directory_version:
        return exe, directory_version.group(1)
    raise ShogunSessionError(f"Cannot determine the version of {exe}; no usable version resource or versioned installation directory.")


def _load_sdk(exe: Path) -> ModuleType:
    global _SDK_MODULE, _SDK_DIRECTORY, _SDK_POISONED
    sdk_dir = exe.parent / "SDK" / "Win64"
    if not (sdk_dir / "vicon_shogun_post.py").is_file():
        raise ShogunSessionError(f"The selected installation has no SDK/Win64/vicon_shogun_post.py: {exe.parent}")
    with _SDK_LOCK:
        if _SDK_POISONED:
            raise ShogunSessionError(f"This Python worker cannot reuse the SDK: {_SDK_POISONED}. Start a fresh worker process.")
        if _SDK_MODULE is not None:
            if _SDK_DIRECTORY != sdk_dir:
                raise ShogunSessionError("A different installation's process-global Shogun SDK is already loaded; use a fresh worker process.")
            return _SDK_MODULE
        for name in ("vicon_shogun_post", "ViconShogunPostSDK"):
            existing = sys.modules.get(name)
            if existing is not None:
                module_file = getattr(existing, "__file__", None)
                if module_file is None or not Path(module_file).resolve().is_relative_to(sdk_dir):
                    raise ShogunSessionError(f"A different {name} module is already imported; use a fresh worker process.")
        try:
            for directory in (exe.parent, sdk_dir, sdk_dir / "ViconShogunPostSDK"):
                _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
            sys.path.insert(0, str(sdk_dir))
            try:
                module = importlib.import_module("vicon_shogun_post")
            finally:
                sys.path.remove(str(sdk_dir))
            if Path(module.__file__).resolve() != sdk_dir / "vicon_shogun_post.py":
                raise ShogunSessionError("SDK import did not resolve to the selected executable's sibling SDK.")
        except Exception as exc:
            _SDK_POISONED = f"SDK import failed: {exc}"
            raise ShogunSessionError(f"Unable to import {sdk_dir} with Python {sys.version.split()[0]}: {exc}") from exc
        _SDK_DIRECTORY = sdk_dir
        _SDK_MODULE = module
        return module


def check_sdk(executable: str) -> dict[str, str]:
    """Import the exact installed SDK without launching or connecting to Shogun."""
    exe, version = _executable_details(executable)
    _load_sdk(exe)
    return {
        "executable": str(exe),
        "version": version,
        "sdk_dir": str(exe.parent / "SDK" / "Win64"),
        "python_version": sys.version.split()[0],
    }


def _listener_owners(port: int) -> set[int]:
    """Read both Windows TCP listener tables without connecting to any server."""
    class Row4(ctypes.Structure):
        _fields_ = [(name, wintypes.DWORD) for name in (
            "state", "local_address", "local_port", "remote_address", "remote_port", "pid",
        )]

    class Row6(ctypes.Structure):
        _fields_ = [
            ("local_address", ctypes.c_ubyte * 16), ("local_scope", wintypes.DWORD),
            ("local_port", wintypes.DWORD), ("remote_address", ctypes.c_ubyte * 16),
            ("remote_scope", wintypes.DWORD), ("remote_port", wintypes.DWORD),
            ("state", wintypes.DWORD), ("pid", wintypes.DWORD),
        ]

    iphlpapi = ctypes.WinDLL("iphlpapi", use_last_error=True)
    get_table = iphlpapi.GetExtendedTcpTable
    get_table.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.BOOL, wintypes.ULONG, ctypes.c_int, wintypes.ULONG]
    get_table.restype = wintypes.DWORD
    owners: set[int] = set()
    for family, row_type in ((socket.AF_INET, Row4), (socket.AF_INET6, Row6)):
        size = wintypes.DWORD()
        result = get_table(None, ctypes.byref(size), False, family, 3, 0)
        if result not in (0, 122):  # ERROR_INSUFFICIENT_BUFFER
            raise OSError(result, "GetExtendedTcpTable size query failed")
        for _ in range(4):  # The table can grow between its size query and copy.
            data = ctypes.create_string_buffer(max(size.value, ctypes.sizeof(wintypes.DWORD)))
            result = get_table(data, ctypes.byref(size), False, family, 3, 0)
            if result != 122:
                break
        if result:
            raise OSError(result, "GetExtendedTcpTable listener query failed")
        count = wintypes.DWORD.from_buffer(data).value
        row_size = ctypes.sizeof(row_type)
        offset = ctypes.sizeof(wintypes.DWORD)
        if offset + count * row_size > len(data):
            raise ShogunSessionError("Windows returned an invalid TCP owner table.")
        for index in range(count):
            row = row_type.from_buffer(data, offset + index * row_size)
            if socket.ntohs(row.local_port & 65535) == port:
                owners.add(int(row.pid))
    return owners


def _private_profile(exe: Path, version: str, work_dir: Path, supplied: str | None) -> Path:
    if supplied is not None:
        source = Path(supplied).expanduser().resolve(strict=True)
    else:
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        shell32.SHGetFolderPathW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR]
        shell32.SHGetFolderPathW.restype = ctypes.c_long
        buffer = ctypes.create_unicode_buffer(32768)
        if shell32.SHGetFolderPathW(None, 0x002E, None, 0, buffer) != 0:  # CSIDL_COMMON_DOCUMENTS
            raise ShogunSessionError("Cannot locate Windows Public Documents; provide an explicit Shogun profile.")
        major_minor = ".".join(version.split(".")[:2])
        source = Path(buffer.value) / "Vicon" / f"ShogunPost{major_minor}" / "default.ini"
    if not source.is_file():
        raise ShogunSessionError(f"A real installed profile is required: {source}. Supply profile explicitly; empty-profile validity is not established.")
    destination = work_dir / "profile.ini"
    if source.resolve() == destination.resolve():
        raise ShogunSessionError("Profile source must differ from the private work-directory profile copy.")
    # Do not inherit a read-only installed profile's permissions.
    shutil.copyfile(source, destination)
    # This section/key is present in the installed Shogun profile, not an
    # invented setting. Never let an inherited GUI autosave policy write back
    # to the source scene while a worker is inspecting or processing it.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetPrivateProfileStringW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,
        wintypes.LPWSTR, wintypes.DWORD, wintypes.LPCWSTR,
    ]
    kernel32.GetPrivateProfileStringW.restype = wintypes.DWORD
    kernel32.WritePrivateProfileStringW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,
    ]
    kernel32.WritePrivateProfileStringW.restype = wintypes.BOOL
    value = ctypes.create_unicode_buffer(64)
    missing = "__missing_autosave__"
    kernel32.GetPrivateProfileStringW("MainFrame", "AutoSave", missing, value, len(value), str(destination))
    if value.value == missing:
        raise ShogunSessionError(f"Profile {source} has no observed MainFrame/AutoSave setting; cannot establish a safe autosave policy.")
    if not kernel32.WritePrivateProfileStringW("MainFrame", "AutoSave", "0", str(destination)):
        raise ctypes.WinError(ctypes.get_last_error())
    # Microsoft documents zero for a cache flush as well as for errors.
    # Verify the actual setting below instead of interpreting this as failure.
    kernel32.WritePrivateProfileStringW(None, None, None, str(destination))
    kernel32.GetPrivateProfileStringW("MainFrame", "AutoSave", missing, value, len(value), str(destination))
    if value.value != "0":
        raise ShogunSessionError(f"Could not disable autosave in private profile {destination}.")
    return destination


def _kill_on_close_job(process: subprocess.Popen[bytes]) -> Any:
    """Attach only our new native process to a private crash-cleanup job."""
    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
            ("flags", wintypes.DWORD), ("minimum_working_set", ctypes.c_size_t),
            ("maximum_working_set", ctypes.c_size_t), ("active_process_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD),
        ]

    class IOCounts(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "read_operations", "write_operations", "other_operations",
            "read_bytes", "write_bytes", "other_bytes",
        )]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("basic", BasicLimits), ("io", IOCounts),
            ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
            ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    limits.basic.flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    try:
        if not kernel32.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.AssignProcessToJobObject(handle, wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())
    except BaseException:
        kernel32.CloseHandle(handle)
        raise
    return handle


class ShogunSession:
    """A private normal-mode CLI, never an attachment to an existing GUI/CLI."""

    def __init__(
        self,
        executable: str,
        work_dir: str | Path,
        log_path: str | Path,
        port: int,
        profile: str | None = None,
        startup_timeout: float = 90,
    ) -> None:
        self.executable = executable
        self.work_dir = Path(work_dir).resolve()
        self.log_path = Path(log_path).resolve()
        self.port = port
        self.profile = profile
        self.startup_timeout = startup_timeout
        self.api: Any = None
        self.pid: int | None = None
        self.version: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._log: Any = None
        self._job: Any = None
        self._sdk: ModuleType | None = None
        self._sdk_thread: threading.Thread | None = None
        self._lock_owned = False
        self._entered = False
        self._native_load_directory: tempfile.TemporaryDirectory | None = None
        self.native_load_audit: list[dict] = []

    def _note(self, message: str) -> None:
        if self._log is not None:
            self._log.write((f"[pipeline-session] {message}\n").encode("utf-8", errors="replace"))

    def _require_owned_listener(self) -> bool:
        process = self._process
        if process is None or process.poll() is not None:
            code = None if process is None else process.returncode
            raise ShogunSessionError(f"Owned Shogun CLI exited before readiness (exit code {code}).")
        owners = _listener_owners(self.port)
        if owners and owners != {process.pid}:
            raise ShogunSessionError(f"Control port {self.port} belongs to PID(s) {sorted(owners)}, not owned PID {process.pid}; refusing SDK connection.")
        return owners == {process.pid}

    def __enter__(self) -> ShogunSession:
        if self._entered:
            raise ShogunSessionError(f"A ShogunSession object may only be entered once. Native log: {self.log_path}")
        self._entered = True
        if not _SESSION_LOCK.acquire(blocking=False):
            raise ShogunSessionError(f"Only one process-global Shogun SDK session is allowed per Python worker. Native log: {self.log_path}")
        self._lock_owned = True
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("ab", buffering=0)
            if not isinstance(self.port, int) or isinstance(self.port, bool) or not 1 <= self.port <= 65535:
                raise ShogunSessionError("Control port must be an explicit integer from 1 through 65535.")
            if not math.isfinite(self.startup_timeout) or self.startup_timeout <= 0:
                raise ShogunSessionError("startup_timeout must be finite and positive.")
            deadline = time.monotonic() + self.startup_timeout
            exe, self.version = _executable_details(self.executable)
            self._sdk = _load_sdk(exe)
            if self._sdk._Client.TheClient.IsConnected():
                raise ShogunSessionError("The SDK already has an external connection; refusing to disconnect or replace it.")
            private_profile = _private_profile(exe, self.version, self.work_dir, self.profile)
            temp_dir = self.work_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            owners = _listener_owners(self.port)
            if owners:
                raise ShogunSessionError(f"Requested control port {self.port} is already owned by PID(s) {sorted(owners)}.")
            # Windows' exclusive bind catches non-listening reservations as well.
            # Release only immediately before launch; ownership checks below close
            # the bind-to-spawn race without attaching to a competing application.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
                reservation.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                reservation.bind(("0.0.0.0", self.port))
            environment = os.environ.copy()
            environment.update(TEMP=str(temp_dir), TMP=str(temp_dir), TMPDIR=str(temp_dir))
            command = [str(exe), "-profile", str(private_profile), "-controlStreamPort", str(self.port)]
            self._note(f"Launching version={self.version} port={self.port} command={subprocess.list2cmdline(command)}")
            self._process = subprocess.Popen(
                command, cwd=self.work_dir, env=environment,
                stdin=subprocess.PIPE, stdout=self._log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW, close_fds=True,
            )
            self.pid = self._process.pid
            self._job = _kill_on_close_job(self._process)
            self._note(f"Owned native PID={self.pid}; private kill-on-close job assigned")
            while not self._require_owned_listener():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Shogun did not listen on explicit control port {self.port} before startup timeout.")
                time.sleep(min(0.1, remaining))
            errors: list[BaseException] = []

            def connect() -> None:
                try:
                    while True:
                        if not self._require_owned_listener():
                            raise ShogunSessionError("Owned control listener disappeared before SDK connection.")
                        try:
                            self.api = self._sdk.ViconShogunPost("127.0.0.1", self.port)
                            self.api.GetSubjectNames()
                            self._require_owned_listener()
                            return
                        except Exception:
                            if time.monotonic() >= deadline:
                                raise
                            time.sleep(0.1)
                except BaseException as exc:
                    errors.append(exc)

            self._sdk_thread = threading.Thread(target=connect, name="shogun-sdk-startup", daemon=True)
            self._sdk_thread.start()
            self._sdk_thread.join(max(0.0, deadline - time.monotonic()))
            if self._sdk_thread.is_alive():
                raise TimeoutError("Shogun SDK readiness exceeded startup timeout.")
            if errors:
                raise errors[0]
            if time.monotonic() > deadline:
                raise TimeoutError("SDK connection completed after the startup deadline.")
            if not self._require_owned_listener():
                raise ShogunSessionError("Owned control listener disappeared after SDK readiness.")
            self._note(f"SDK ready; verified native PID={self.pid} exclusively owns control port={self.port}")
            return self
        except BaseException as exc:
            cleanup_error = None
            try:
                self._close()
            except Exception as cleanup_exc:
                cleanup_error = cleanup_exc
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            suffix = f"; cleanup: {cleanup_error}" if cleanup_error else ""
            raise ShogunSessionError(f"Shogun startup failed: {exc}{suffix}. Native log: {self.log_path}") from exc

    def hsl(self, command: str) -> str:
        if self.api is None or not self._lock_owned or not self._require_owned_listener():
            raise ShogunSessionError(f"Shogun session is not ready. Native log: {self.log_path}")
        return self.api.HSL(command)

    def _close(self) -> None:
        global _SDK_POISONED
        problems: list[str] = []
        try:
            process = self._process
            if process is not None:
                try:
                    if process.poll() is None:
                        self._note(f"Requesting graceful exit of owned PID={process.pid}")
                        try:
                            process.stdin.write(b"exit\n")
                            process.stdin.flush()
                        except (BrokenPipeError, OSError):
                            pass
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self._note(f"Graceful exit timed out; terminating only owned PID={process.pid}")
                            process.kill()  # Uses Popen's original process HANDLE, not a PID lookup.
                            process.wait(timeout=5)
                except Exception as exc:
                    problems.append(f"owned CLI stop failed: {exc}")
                finally:
                    if process.stdin is not None:
                        try:
                            process.stdin.close()
                        except OSError as exc:
                            problems.append(f"native stdin close failed: {exc}")
            # Closing this non-inherited handle also covers a native process that
            # failed to stop above, and owned descendants, never an existing GUI.
            if self._job is not None:
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
                kernel32.CloseHandle.restype = wintypes.BOOL
                if not kernel32.CloseHandle(self._job):
                    problems.append(f"private job close failed: {ctypes.WinError(ctypes.get_last_error())}")
                self._job = None
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    problems.append(f"owned PID {process.pid} did not exit after private job closure")
            if self._sdk_thread is not None:
                self._sdk_thread.join(5)
                if self._sdk_thread.is_alive():
                    _SDK_POISONED = "SDK startup thread did not return after its owned CLI was stopped"
                    problems.append(_SDK_POISONED)
            if self._process is not None and not (self._sdk_thread and self._sdk_thread.is_alive()):
                disconnect_errors: list[BaseException] = []

                def disconnect() -> None:
                    try:
                        self._sdk._Client.TheClient.Disconnect()
                    except BaseException as exc:
                        disconnect_errors.append(exc)

                thread = threading.Thread(target=disconnect, name="shogun-sdk-disconnect", daemon=True)
                thread.start()
                thread.join(5)
                if thread.is_alive():
                    _SDK_POISONED = "SDK disconnect did not return after its owned CLI was stopped"
                    problems.append(_SDK_POISONED)
                elif disconnect_errors:
                    _SDK_POISONED = f"SDK disconnect failed: {disconnect_errors[0]}"
                    problems.append(_SDK_POISONED)
            if self._native_load_directory is not None:
                if process is not None and process.poll() is None:
                    problems.append(f"Native snapshots retained for still-running PID {process.pid}: "
                                    f"{self._native_load_directory.name}")
                else:
                    self._native_load_directory.cleanup()
                    self._native_load_directory = None
            self.api = None
            self._process = None
            self._note(f"Session closed; owned PID={self.pid}; cleanup_errors={problems}")
        finally:
            try:
                if self._log is not None:
                    self._log.close()
                    self._log = None
            finally:
                if self._lock_owned:
                    self._lock_owned = False
                    _SESSION_LOCK.release()
        if problems:
            raise ShogunSessionError(f"{'; '.join(problems)}. Native log: {self.log_path}")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self._close()
        except Exception as cleanup_exc:
            if exc is None:
                raise
            exc.add_note(str(cleanup_exc))
