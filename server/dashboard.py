"""Serve the independent, read-only Shogun dashboard on 127.0.0.1 only.

Example: python server/dashboard.py --output-dir H:/MocapData_processed
         --data-path H:/MocapData --port 8787 --refresh-seconds 15
The server never imports the SDK, starts Post, changes queues or writes outputs.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, quote, unquote, urlsplit

# dashboard_data imports read-only modules (check_health, step1_process_takes)
# from the 4DHOI scripts directory at import time; MOCAP_REPO_SCRIPTS overrides
# the location, so this bootstrap must run before that import.
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

from dashboard_data import DashboardData, open_scoped, safe_component  # noqa: E402
from dashboard_runs import MultiRunCache  # noqa: E402


STATIC_ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 1024 * 1024


def path_segments(raw_path):
    """Decode once per segment; encoded separators are never routing syntax."""
    if not raw_path.startswith("/") or raw_path.startswith("//"):
        raise ValueError("invalid request path")
    if raw_path == "/":
        return []
    result = []
    for component in raw_path[1:].split("/"):
        if re.search(r"%(?![0-9a-fA-F]{2})", component):
            raise ValueError("malformed URL encoding")
        component = unquote(component, encoding="utf-8", errors="strict")
        if not component or component in (".", "..") or any(value in component for value in ("/", "\\", ":", "\x00")):
            raise ValueError("invalid path component")
        result.append(component)
    return result


def byte_range(value, size):
    if value is None:
        return 0, size - 1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not match or not any(match.groups()) or size <= 0:
        raise ValueError("unsupported or unsatisfiable range")
    first, last = match.groups()
    if not first:
        length = int(last)
        if length <= 0:
            raise ValueError("unsatisfiable range")
        return max(0, size - length), size - 1, True
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise ValueError("unsatisfiable range")
    return start, end, True


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port, cache, static_root=STATIC_ROOT):
        self.cache = cache
        self.static_root = Path(static_root).resolve()
        # No configurable bind address: the read-only service is local-only.
        super().__init__(("127.0.0.1", port), DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ShogunDashboard/1"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def __getattr__(self, name):
        if name.startswith("do_"):
            return self.method_not_allowed
        raise AttributeError(name)

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        super().end_headers()

    def allowed_request(self):
        port = self.server.server_address[1]
        authorities = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            authorities.update(("127.0.0.1", "localhost"))
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0].lower() not in authorities:
            self.error_json(403, "Only the local dashboard Host is allowed")
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin.lower() not in {f"http://{host}" for host in authorities}:
            self.error_json(403, "Cross-origin requests are not allowed")
            return False
        return True

    def error_json(self, code, message, *, headers=None):
        self.json_response({"error": message}, code, headers=headers)

    def json_response(self, value, status=200, *, headers=None):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if headers:
            for name, item in headers.items():
                self.send_header(name, str(item))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def method_not_allowed(self):
        # Never consume or execute posted content; close to avoid treating its
        # body as another request on a persistent connection.
        self.close_connection = True
        if self.allowed_request():
            self.error_json(405, "Read-only service; use GET or HEAD", headers={"Allow": "GET, HEAD", "Connection": "close"})

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self.allowed_request():
            return
        try:
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc or parsed.fragment:
                raise ValueError("only origin-form paths are accepted")
            segments = path_segments(parsed.path)
            query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=4)
            manager = self.server.cache
            if segments == ["api", "runs"]:
                if query:
                    raise ValueError("query parameters are not allowed for this resource")
                self.json_response(manager.runs())
                return
            api_resource = segments[:2] in (["api", "overview"], ["api", "takes"], ["api", "artifacts"])
            if api_resource:
                allowed = {"run", "refresh"} if segments == ["api", "overview"] else {"run"}
                if set(query) - allowed or any(len(values) != 1 for values in query.values()):
                    raise ValueError("invalid API query parameters")
                if "refresh" in query and query["refresh"] != ["1"]:
                    raise ValueError("only refresh=1 is supported")
                identifier = query.get("run", [None])[0]
                cache = manager.for_run(identifier) if isinstance(manager, MultiRunCache) else manager
                if identifier is not None and not isinstance(manager, MultiRunCache):
                    raise KeyError(identifier)
                if segments == ["api", "overview"]:
                    self.json_response(manager.overview(identifier, refresh="refresh" in query) if isinstance(manager, MultiRunCache)
                                       else cache.overview(refresh="refresh" in query))
                    return
            elif query:
                is_doc_run = (segments in ([], ["index.html"]) and set(query) == {"run"}
                              and len(query["run"]) == 1 and bool(re.fullmatch(r"[0-9a-f]{16}", query["run"][0])))
                is_version = (set(query) == {"v"} and len(query["v"]) == 1
                              and bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", query["v"][0])))
                if not (is_doc_run or is_version):
                    raise ValueError("invalid query parameters for this static resource")
            if len(segments) == 3 and segments[:2] == ["api", "takes"]:
                if not safe_component(segments[2]):
                    self.error_json(404, "Unknown take")
                    return
                snapshot = cache.snapshot()
                if snapshot is None:
                    self.error_json(404, "Take unavailable before the first snapshot")
                    return
                self.json_response(cache.data.detail(snapshot, segments[2]))
                return
            if len(segments) == 4 and segments[:2] == ["api", "artifacts"]:
                if not safe_component(segments[2]) or not safe_component(segments[3]):
                    self.error_json(404, "Unknown artifact")
                    return
                snapshot = cache.snapshot()
                if snapshot is None:
                    self.error_json(404, "Unknown artifact")
                    return
                opened = cache.data.open_artifact(snapshot, segments[2], segments[3])
                self.send_file(opened, download=True)
                return
            if segments and segments[0] == "api":
                self.error_json(404, "Unknown API resource")
                return
            relative = "/".join(segments) if segments else "index.html"
            opened = open_scoped(self.server.static_root, relative, nonempty=False)
            self.send_file(opened, download=False)
        except (KeyError, FileNotFoundError, NotADirectoryError):
            self.error_json(404, "Resource not found")
        except PermissionError:
            self.error_json(403, "Resource is inaccessible")
        except (ValueError, UnicodeError, RuntimeError) as exc:
            self.error_json(400, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except OSError:
            self.error_json(404, "Resource is unavailable")

    def send_file(self, opened, *, download):
        stream, info, path = opened
        with stream:
            try:
                start, end, partial = byte_range(self.headers.get("Range"), info.st_size)
            except (ValueError, OverflowError):
                self.error_json(416, "Only one satisfiable byte range is supported",
                                headers={"Content-Range": f"bytes */{info.st_size}", "Accept-Ranges": "bytes"})
                return
            length = max(0, end - start + 1)
            self.send_response(206 if partial else 200)
            content_type = "application/octet-stream" if download else mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if not download and path.suffix.lower() in (".html", ".css", ".js"):
                content_type += "; charset=utf-8"
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store" if download else "no-cache")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{info.st_size}")
            if download:
                suffix = path.suffix if re.fullmatch(r"\.[A-Za-z0-9]+", path.suffix) else ""
                self.send_header("Content-Disposition", f"attachment; filename=artifact{suffix}; filename*=UTF-8''{quote(path.name, safe='')}")
            self.end_headers()
            if self.command == "HEAD":
                return
            stream.seek(start)
            remaining = length
            while remaining:
                chunk = stream.read(min(CHUNK_BYTES, remaining))
                if not chunk:
                    self.close_connection = True
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def port_number(value):
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return number


def existing_dir(value):
    if not Path(value).is_dir():
        raise argparse.ArgumentTypeError(f"not an existing directory: {value}")
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", default="H:/MocapData/_processing/MocapData_processed",
                        help="Existing run control directory; never written (sequence-local manifest or generic layout)")
    result.add_argument("--data-path",
                        help="Optional separate raw MCP file/dataset for coverage; defaults to manifest data_root, not download authorization")
    result.add_argument("--port", type=port_number, default=8787)
    result.add_argument("--refresh-seconds", type=positive_seconds, default=15)
    result.add_argument("--static-root", type=existing_dir, default=str(STATIC_ROOT),
                        help="Frontend directory containing index.html; defaults to the MocapStudio root above server/")
    result.add_argument("--plan-dir", help="p2_hoi plan directory (导演台本 + notes.json); default <web root>/data/p2_hoi")
    result.add_argument("--run-registry", help="Explicit run catalog directory; defaults to MOCAP_RUN_REGISTRY or H:/MocapStudio/data/runs")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    data = DashboardData(arguments.output_dir, arguments.data_path, arguments.refresh_seconds, plan_dir=arguments.plan_dir)
    if not data.output.is_dir():
        parser().error(f"Output directory does not exist or is inaccessible: {data.output}")
    cache = MultiRunCache(data, arguments.run_registry)
    try:
        server = DashboardServer(arguments.port, cache, arguments.static_root)
    except OSError as exc:
        print(f"Cannot bind local dashboard: {exc}", file=sys.stderr)
        return 1
    cache.start()
    print(f"Read-only Shogun dashboard: http://127.0.0.1:{arguments.port}/", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
