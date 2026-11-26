"""Standalone loopback MocapStudio: processing workbench.

The canonical Shogun pipeline remains in MOCAP_REPO_SCRIPTS; no capture-site
server, proxy, or Node runtime is used. Only explicit public assets are served.
"""
from __future__ import annotations

import argparse
import ctypes
from http.server import BaseHTTPRequestHandler
import os
from pathlib import Path
import platform as _platform
import shutil
import signal
import subprocess
import sys
from urllib.parse import urlsplit

import dashboard

# dashboard bootstraps the repo-scripts path convention shared with the
# canonical queue engine in web_manager.
dashboard.ensure_repo_scripts_path()
from dashboard_data import open_scoped  # noqa: E402
from web_manager import APIError, JobManager, ProcessingHandler, ProcessingServer, SmplManager, discover_executable  # noqa: E402

APP_ROOT = Path(__file__).resolve().parents[1]
ASSET_SUFFIXES = frozenset({".js", ".mjs", ".css", ".svg", ".png", ".jpg", ".jpeg", ".webp", ".ico", ".woff", ".woff2", ".ttf"})
WORKBENCH_CSP = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' data: https://fonts.gstatic.com; img-src 'self' data: blob:; connect-src 'self'; frame-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
VIEWER_CSP = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; connect-src 'self' blob:; worker-src 'self' blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'self'; form-action 'none'"


_HARDWARE: dict | None = None


def _cpu_info():
    name = None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            name = str(winreg.QueryValueEx(key, "ProcessorName0")[0]).strip()
    except OSError:
        name = _platform.processor() or None
    if not name or "Family" in name:
        # OEM images sometimes leave the raw family string in the registry; WMI carries the marketing brand.
        try:
            completed = subprocess.run(["powershell", "-NoProfile", "-Command",
                                        "Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name"],
                                       capture_output=True, text=True, timeout=15)
            brand = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else None
            if brand:
                name = brand
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    return {"name": name or _platform.machine(), "logical_cores": os.cpu_count()}


def _memory_total():
    try:
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except (AttributeError, OSError):
        pass
    try:
        page, pages = os.sysconf("SC_PAGE_SIZE"), os.sysconf("SC_PHYS_PAGES")
        if page > 0 and pages > 0:
            return int(page) * int(pages)
    except (ValueError, OSError):
        pass
    return None


def _gpus():
    result = []
    try:
        completed = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=8)
        for line in completed.stdout.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) >= 3:
                try:
                    memory_mb = int(fields[2])
                except ValueError:
                    memory_mb = None
                result.append({"vendor": "NVIDIA", "name": fields[0], "driver": fields[1], "memory_mb": memory_mb})
    except (OSError, subprocess.SubprocessError):
        pass
    if not result:
        try:
            completed = subprocess.run(["powershell", "-NoProfile", "-Command",
                                        "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                                       capture_output=True, text=True, timeout=15)
            for line in completed.stdout.splitlines():
                if line.strip():
                    result.append({"vendor": "unknown", "name": line.strip(), "driver": None, "memory_mb": None})
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def hardware_report():
    global _HARDWARE
    if _HARDWARE is None:
        _HARDWARE = {
            "system": {"platform": _platform.system(), "release": _platform.release(), "machine": _platform.machine()},
            "cpu": _cpu_info(),
            "memory_total_bytes": _memory_total(),
            "gpus": _gpus(),
            "python": _platform.python_version(),
        }
    return _HARDWARE


_METRICS_PREV = None


def _memory_status():
    """(total, available) bytes, or (None, None)."""
    try:
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
    except (AttributeError, OSError):
        pass
    try:
        page, pages = os.sysconf("SC_PAGE_SIZE"), os.sysconf("SC_PHYS_PAGES")
        if page > 0 and pages > 0:
            return int(page) * pages, None
    except (ValueError, OSError):
        pass
    return None, None


def _cpu_percent():
    """Delta-based CPU busy percent between successive polls (None on first poll)."""
    global _METRICS_PREV
    try:
        class FileTime(ctypes.Structure):
            _fields_ = [("lo", ctypes.c_ulong), ("hi", ctypes.c_ulong)]

        def value(ft):
            return (ft.hi << 32) | ft.lo
        idle, kernel, user = FileTime(), FileTime(), FileTime()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        now = (value(idle), value(kernel), value(user))
    except (AttributeError, OSError):
        try:
            now = (os.times().idle, os.times().idle + os.times().system, os.times().idle + os.times().user)
        except OSError:
            return None
    previous, _METRICS_PREV = _METRICS_PREV, now
    if previous is None:
        return None
    idle_delta = now[0] - previous[0]
    total = (now[1] - previous[1]) + (now[2] - previous[2])  # kernel time includes idle on Windows
    if total <= 0:
        return None
    return round(100.0 * (total - idle_delta) / total, 1)


def _uptime_seconds():
    try:
        return round(ctypes.windll.kernel32.GetTickCount64() / 1000, 1)
    except (AttributeError, OSError):
        try:
            with open("/proc/uptime", encoding="ascii") as handle:
                return round(float(handle.read().split()[0]), 1)
        except (OSError, ValueError):
            return None


def metrics_report(data_roots):
    total, available = _memory_status()
    disks = []
    for root in data_roots:
        try:
            usage = shutil.disk_usage(str(root))
            disks.append({"path": str(root), "total_bytes": usage.total, "free_bytes": usage.free})
        except OSError:
            continue
    return {
        "cpu_percent": _cpu_percent(),
        "memory_used_bytes": (total - available) if total is not None and available is not None else None,
        "memory_total_bytes": total,
        "disk": disks,
        "uptime_seconds": _uptime_seconds(),
    }


def explicit_origins(values):
    """Require exact HTTP(S) origins, not URLs, wildcard hosts, or credentials."""
    result = set()
    for value in values:
        try:
            parsed = urlsplit(value)
            valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                     and parsed.username is None and parsed.password is None
                     and not parsed.path and not parsed.query and not parsed.fragment
                     and parsed.port != 0 and value == f"{parsed.scheme}://{parsed.netloc}")
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f"Expected an exact HTTP(S) origin: {value!r}")
        result.add(value)
    return frozenset(result)


class AppServer(ProcessingServer):
    """One listener and one queue lock; closes only this manager's workers."""
    def __init__(self, port, manager, *, app_root=APP_ROOT,
                 workbench_root=None, viewer_root=None, additional_origins=(), smpl=None):
        self.static_root = Path(app_root).resolve()
        self.workbench_root = Path(workbench_root or self.static_root / "workbench" / "dist").resolve()
        self.viewer_root = Path(viewer_root or self.static_root / "viewer").resolve()
        self._resources_closed = False
        super().__init__(("127.0.0.1", port), manager, explicit_origins(additional_origins), smpl=smpl)
        self.smpl = smpl
        self.RequestHandlerClass = AppHandler

    def start(self):
        self.manager.start()
        if self.smpl is not None:
            self.smpl.start()

    def server_close(self):
        try:
            super().server_close()
        finally:
            if not self._resources_closed:
                self._resources_closed = True
                if self.smpl is not None:
                    self.smpl.close()
                self.manager.close()


class AppHandler(ProcessingHandler, dashboard.DashboardHandler):
    server_version = "MocapStudio/1"
    sys_version = ""

    def _cors_origin(self):
        """Echo back an explicitly allowlisted dev origin; never a wildcard."""
        origin = self.headers.get("Origin")
        additional = getattr(self.server, "additional_origins", frozenset())
        return origin if origin and origin in additional else None

    def end_headers(self):
        policy = getattr(self, "_policy", "workbench")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Frame-Options", "SAMEORIGIN" if policy == "viewer" else "DENY")
        csp = VIEWER_CSP if policy == "viewer" else WORKBENCH_CSP
        if policy == "viewer":
            # The workbench dev origin embeds the viewer in an iframe; mirror the
            # explicit allowlist instead of hard-locking to 'self'. Production
            # launches add nothing, so the directive stays 'self' there.
            additional = sorted(getattr(self.server, "additional_origins", frozenset()))
            if additional:
                csp = csp.replace("frame-ancestors 'self'", "frame-ancestors 'self' " + " ".join(additional))
        self.send_header("Content-Security-Policy", csp)
        cors = self._cors_origin()
        if cors:
            self.send_header("Access-Control-Allow-Origin", cors)
            self.send_header("Vary", "Origin")
        # Neither parent's end_headers may add a second, conflicting CSP.
        BaseHTTPRequestHandler.end_headers(self)

    def do_OPTIONS(self):
        cors = self._cors_origin()
        if cors and urlsplit(self.path).path.startswith("/api/processing/"):
            self.send_response(204)
            # end_headers() already echoes the allowlisted Access-Control-Allow-Origin;
            # emitting it here too would duplicate the header and fail the browser's
            # single-value check on the preflight response.
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Mocap-Token")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()
            return
        self.close_connection = True
        self.send_response(403)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _local_request(self, *, processing=False):
        port = self.server.server_address[1]
        authorities = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            authorities.update({"127.0.0.1", "localhost"})
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0].lower() not in authorities:
            raise APIError(403, "Only this local MocapStudio Host is allowed")
        origins = self.headers.get_all("Origin", [])
        allowed = {f"http://{authority}" for authority in authorities}
        if processing:
            allowed.update(self.server.additional_origins)
        if len(origins) > 1 or (origins and origins[0] not in allowed):
            raise APIError(403, "Cross-origin requests are not allowed")

    def allowed_request(self):
        try:
            self._local_request()
            return True
        except APIError as exc:
            self.close_connection = True
            self.error_json(exc.status, str(exc))
            return False

    def _security(self):
        # Tighten the engine's shared-server trust policy to this exact listener.
        # Keep its token, JSON content type and body semantics unchanged.
        self._local_request(processing=True)
        ProcessingHandler._security(self)

    def _url(self):
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc or parsed.fragment:
            raise APIError(400, "Only origin-form request paths are accepted")
        path = parsed.path
        if path in {"/monitor/", "/workbench/"}:
            path = path[:-1]
        return parsed, dashboard.path_segments(path)

    def _processing(self):
        self._policy = "workbench"
        ProcessingHandler._dispatch(self)

    def _asset(self, root, relative):
        opened = open_scoped(root, relative, nonempty=False)
        self.send_file(opened, download=False)

    def do_GET(self):
        self._policy = "workbench"
        try:
            _, segments = self._url()
            if segments[:2] == ["api", "processing"]:
                if segments[2:] == ["hardware"]:
                    self._security()
                    self._json(200, {"hardware": hardware_report()})
                    return
                if segments[2:] == ["metrics"]:
                    self._security()
                    self._json(200, {"metrics": metrics_report(self.server.manager.data_roots)})
                    return
                self._processing()
                return
            self._local_request()
            if not segments or segments in (["index.html"], ["monitor"], ["monitor", "index.html"]):
                # The former dashboard and read-only monitor pages are gone;
                # every former entry point lands on the workbench.
                self.send_response(302)
                self.send_header("Location", "/workbench/")
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif segments in (["workbench"], ["workbench", "index.html"]):
                if not (self.server.workbench_root / "index.html").is_file():
                    raise APIError(503, "Workbench build is missing. Run npm ci and npm run build in MocapStudio/workbench, or launch with -Build.")
                self._asset(self.server.workbench_root, "index.html")
            elif segments[0] == "workbench":
                if len(segments) < 3 or segments[1] != "assets" or Path(segments[-1]).suffix.lower() not in ASSET_SUFFIXES:
                    raise APIError(404, "Unknown workbench asset")
                self._asset(self.server.workbench_root / "assets", "/".join(segments[2:]))
            elif segments[0] == "pipeline-viewer":
                self._policy = "viewer"
                relative = segments[1:]
                if relative != ["fbx_viewer.html"] and not (len(relative) >= 4 and relative[:2] == ["vendor", "three"] and Path(relative[-1]).suffix.lower() in ASSET_SUFFIXES):
                    raise APIError(404, "Unknown viewer asset")
                self._asset(self.server.viewer_root, "/".join(relative))
            else:
                raise APIError(404, "Unknown MocapStudio resource")
        except APIError as exc:
            self.close_connection = True
            self.error_json(exc.status, str(exc))
        except (FileNotFoundError, NotADirectoryError, KeyError):
            self.error_json(404, "Resource not found")
        except PermissionError:
            self.error_json(403, "Resource is inaccessible")
        except (ValueError, UnicodeError, RuntimeError) as exc:
            self.close_connection = True
            self.error_json(400, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except OSError:
            self.error_json(404, "Resource is unavailable")

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        self._policy = "workbench"
        try:
            _, segments = self._url()
            if segments[:2] != ["api", "processing"]:
                self.method_not_allowed()
                return
            self._processing()
        except APIError as exc:
            self.close_connection = True
            self.error_json(exc.status, str(exc))
        except (ValueError, UnicodeError) as exc:
            self.close_connection = True
            self.error_json(400, str(exc))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--port", type=dashboard.port_number, default=8787)
    result.add_argument("--app-root", default=str(APP_ROOT), help="MocapStudio public installation root")
    result.add_argument("--workbench-root", help="Built workbench dist directory")
    result.add_argument("--viewer-root", help="Copied standalone viewer directory")
    result.add_argument("--state-dir", default=os.environ.get("MOCAP_STATE_DIR", "H:/MocapData/jobs"))
    result.add_argument("--jobs-dir", default=os.environ.get("MOCAP_JOBS_DIR", "H:/MocapData/jobs"))
    result.add_argument("--data-root", action="append", help="Authorized processing dataset root; repeatable")
    result.add_argument("--max-workers", type=int, choices=range(1, 9), default=1)
    result.add_argument("--shogun-cl", default=os.environ.get("MOCAP_SHOGUN_CL"))
    result.add_argument("--blender", default=os.environ.get("MOCAP_BLENDER"))
    result.add_argument("--smpl-workers", type=int, choices=range(1, 5), default=int(os.environ.get("MOCAP_SMPL_WORKERS", "2")))
    result.add_argument("--wsl-distro", default=os.environ.get("MOCAP_WSL_DISTRO", "Ubuntu-22.04"))
    result.add_argument("--dirty-repo", default=os.environ.get("MOCAP_DIRTY_REPO", "F:/projects/4DHOI/DirtyMoCap"))
    result.add_argument("--smplh-pkl", default=os.environ.get("MOCAP_SMPLH_PKL", "F:/projects/4DHOI/SMPL/SMPLH/merged/SMPLH_NEUTRAL.pkl"))
    result.add_argument("--allow-origin", action="append", default=[], help="Explicit additional processing origin (does not relax Host validation)")
    return result


def main(argv=None):
    # pythonw / detached service: no console means sys.stdout/stderr are None.
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8", errors="replace"))
    arguments = parser().parse_args(argv)
    origins = arguments.allow_origin + [value.strip() for value in os.environ.get("MOCAP_ALLOWED_ORIGINS", "").split(",") if value.strip()]
    try:
        origins = explicit_origins(origins)
    except ValueError as exc:
        parser().error(str(exc))
    roots = arguments.data_root or [part for part in os.environ.get("MOCAP_DATA_ROOTS", "H:/MocapData").split(os.pathsep) if part]
    program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
    shogun_roots = [Path("D:/Software/Vicon"), program_files / "Vicon"]
    executable = discover_executable(arguments.shogun_cl, "ShogunPostCL.exe", [(root, "ShogunPost*/ShogunPostCL.exe") for root in shogun_roots])
    blender = discover_executable(arguments.blender, "blender", [(Path("D:/Software/Blender"), "blender.exe"), (program_files / "Blender Foundation", "Blender*/blender.exe")])
    manager = None
    server = None
    previous_signals = {}
    try:
        manager = JobManager(arguments.state_dir, roots, executable=executable, blender=blender, max_workers=arguments.max_workers,
                             jobs_dir=arguments.jobs_dir or None)
        smpl = SmplManager(arguments.jobs_dir or arguments.state_dir, roots, blender=blender,
                           max_workers=arguments.smpl_workers, wsl_distro=arguments.wsl_distro,
                           dirty_repo=arguments.dirty_repo, smplh_pkl=arguments.smplh_pkl)
        server = AppServer(arguments.port, manager, app_root=arguments.app_root,
                           workbench_root=arguments.workbench_root, viewer_root=arguments.viewer_root,
                           additional_origins=origins, smpl=smpl)
        def stop(signum, frame):
            raise KeyboardInterrupt()
        for name in (signal.SIGINT, signal.SIGTERM):
            previous_signals[name] = signal.signal(name, stop)
        server.start()
        print(f"MocapStudio workbench: http://127.0.0.1:{server.server_address[1]}/workbench/", flush=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Cannot start MocapStudio: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        for name, previous in previous_signals.items():
            signal.signal(name, previous)
        if server is not None:
            server.server_close()
        elif manager is not None:
            manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
