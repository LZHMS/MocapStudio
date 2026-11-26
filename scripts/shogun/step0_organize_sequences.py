"""Safely relocate stopped centralized runs into sequence-local storage.

Dry-run is the default. Required source roots must exist; --optional-source-output
roots may be missing. Standalone diagnostic roots without canonical take reports
are inventoried but never relocated unless explicit supplemental mappings are supplied.
No SDK, processing, links, or binary copies.

JSON rewriting uses a validating streaming lexer, not json.load on batch reports:
64 KiB input chunks, at most 8 MiB per string token, and nesting at most 256.
ijson, when installed, streams canonical/attempt metadata too. Without it, only
identity documents use a documented 64 MiB bound (never batch JSON). Historical
NaN/Infinity numeric diagnostics are preserved; strict-parser identity failures
also use that bounded fallback. Original JSON bytes are streamed into
_migration_history before any move. JSON hash dependencies are
resolved before moving, so independent recheck and source-report hashes follow
rewritten evidence. Binary mtimes are preserved by same-volume rename, hence
checkpoint size/mtime fingerprints remain unchanged. Log text is historical and
is deliberately never rewritten. The journal/map is audit data, not an alias API.

The caller must stop all writers first. Moves extract children before their source
parents, but install destination ancestors before descendant attempts/history.
These prerequisites are planned before mutation; directories are never merged.
A durable journal makes interrupted moves and JSON replacement resumable by
invoking the exact same command again. Raw file identity/size/mtime baselines
are checked before moving and after applying.
Legacy FAILED reports lacking relative_path remain canonical only when their
explicit existing raw input_path agrees with the complete old takes location;
the journal records that evidence without synthesizing missing report fields.
Registry files outside the selected runs are not edited; control_mapping in the
journal is the explicit handoff for their owner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import sys

from pipeline_layout import initialize_layout, is_generated, take_paths

try:
    import ijson
    import ijson.backends.python
    # ijson 3.6.0's default yajl2_c backend has an out-of-bounds readinto
    # bug (ICRAR/ijson #187, fixed only in unreleased dev) that surfaces as
    # a nondeterministic access violation while parsing large reports.
    # The pure-python backend is ~10x slower but stable; streaming keeps
    # memory flat either way.
    _PARSE = ijson.backends.python.parse
except ImportError:
    ijson = None
    _PARSE = None


SCHEMA_VERSION = 1
CHUNK = 64 * 1024
MAX_TOKEN = 8 * 1024 * 1024
MAX_IDENTITY = 64 * 1024 * 1024
# Resumable parse cache for this unstable host: large batch reports on H:
# crash long-running processes nondeterministically (machine-level AVs), so
# every parsed report is journaled to a local-NVMe cache keyed by
# path+size+mtime_ns. Re-running the same command skips finished reports.
META_CACHE = Path(__file__).with_name(".reloc_cache")
HASH = re.compile(r"[0-9a-fA-F]{64}\Z")
NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
HASH_PATTERN = re.compile(r'"([0-9a-fA-F]{64})"')
BUCKETS = {".work": "work_root", "_failed": "failed_root",
           ".quality_recheck_attempts": "recheck_root"}


class MigrationError(ValueError):
    """Unsafe or inconsistent migration; no speculative recovery is allowed."""


def _key(value: str | Path) -> str:
    # Windows dataset identity is case-insensitive, including in portable tests.
    return str(value).replace("\\", "/").rstrip("/").casefold()


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _absolute(value: object) -> bool:
    return isinstance(value, str) and (Path(value).is_absolute()
                                      or PureWindowsPath(value).is_absolute())


_SAFE_DIRS: set[str] = set()


def _safe(path: Path) -> None:
    """Reject symlinks/junctions in all existing components, without resolving."""
    # Verified directories are memoized: the walk re-validates every ancestor
    # of every file, which otherwise costs lstat(depth) per entry.  Only
    # directories are cached, so a file component is still lstat'd fresh.
    for part in (path, *path.parents):
        key = _key(part)
        if key in _SAFE_DIRS:
            continue
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400):
            raise MigrationError(f"Symlink/reparse mapping is forbidden: {part}")
        if stat.S_ISDIR(info.st_mode):
            _SAFE_DIRS.add(key)


_STAT_CACHE: dict[str, tuple] = {}


def _fingerprint(path: Path) -> dict:
    key = _key(path)
    cached = _STAT_CACHE.get(key)
    if cached is not None:
        return {"device": cached[0], "inode": cached[1],
                "size": cached[2], "mtime_ns": cached[3]}
    info = path.stat()
    row = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    _STAT_CACHE[key] = row
    return {"device": row[0], "inode": row[1], "size": row[2], "mtime_ns": row[3]}



def _path(path: str | Path) -> Path:
    result = Path(os.path.abspath(path))
    _safe(result)
    return result




def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()



def _stat_row(entry, is_link: bool) -> list:
    # os.stat, not DirEntry.stat: Windows fakes st_ino differently between
    # the two, and journal rows must equal what _fingerprint records later.
    info = os.stat(entry.path)
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, 1 if is_link else 0]



def _walk(root: Path) -> list[Path]:
    # Enumerate with per-directory journals keyed by the directory's stat.
    # A crash mid-scan resumes at the next directory instead of re-walking
    # the whole root; scandir costs one listing per directory.  Journals also
    # carry the scandir stat row per file so inventory fingerprinting costs
    # no extra H: calls.  Once every directory journal is hot, a combined
    # per-root manifest (walkroot) lets a later attempt read the whole file
    # list in one JSON parse instead of thousands of small reads.
    _safe(root)
    if root.is_file():
        return [root]
    try:
        root_key = _cache_key_extra(root, "walkroot", "")
        if root_key.is_file():
            files, stats = json.loads(root_key.read_text(encoding="utf-8"))
            for value, row in zip(files, stats):
                if not row[4]:
                    _STAT_CACHE[_key(value)] = tuple(row[:4])
            return [Path(v) for v in files]
    except (OSError, ValueError):
        root_key = None
    all_rows: list[tuple[str, list]] = []

    def walk_dir(directory: Path) -> list[Path]:
        key = None
        try:
            key = _cache_key_extra(directory, "walkd", "")
            if key.is_file():
                dirs, files, stats = json.loads(key.read_text(encoding="utf-8"))
                for value, row in zip(files, stats):
                    all_rows.append((value, row))
                    if not row[4]:
                        _STAT_CACHE[_key(value)] = tuple(row[:4])
                return [Path(v) for v in files] + \
                    [found for d in (Path(v) for v in dirs) for found in walk_dir(d)]
        except (OSError, ValueError):
            key = None
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise MigrationError(f"Cannot list {directory}: {error}") from error
        subdirs, names, stats = [], [], []
        for entry in entries:
            candidate = Path(entry.path)
            _safe(candidate)
            if entry.is_symlink():
                # os.walk(followlinks=False) ignores linked directories and
                # keeps linked files, matching the previous scan exactly.
                if not entry.is_dir(follow_symlinks=True):
                    names.append(candidate); row = _stat_row(entry, True)
                    stats.append(row); all_rows.append((candidate.as_posix(), row))
                continue
            if entry.is_dir(follow_symlinks=False):
                subdirs.append(candidate)
            elif entry.is_file(follow_symlinks=False):
                names.append(candidate); row = _stat_row(entry, False)
                stats.append(row); all_rows.append((candidate.as_posix(), row))
            else:
                raise MigrationError(f"Non-regular migration entry: {candidate}")
        result = list(names)
        for candidate in subdirs:
            result.extend(walk_dir(candidate))
        if key is not None:
            try:
                key.parent.mkdir(parents=True, exist_ok=True)
                tmp = key.with_suffix(".tmp")
                tmp.write_text(json.dumps([[p.as_posix() for p in subdirs],
                                           [p.as_posix() for p in names], stats]),
                               encoding="utf-8")
                os.replace(tmp, key)
            except OSError:
                pass
        return result

    result = walk_dir(root)
    if root_key is not None:
        try:
            root_key.parent.mkdir(parents=True, exist_ok=True)
            tmp = root_key.with_suffix(".tmp")
            tmp.write_text(json.dumps([[v for v, _ in all_rows],
                                       [r for _, r in all_rows]]), encoding="utf-8")
            os.replace(tmp, root_key)
        except OSError:
            pass
    return result


def _raw_baseline(data_root: Path) -> dict:
    result = {}
    for directory, dirs, names in os.walk(data_root, followlinks=False):
        base = Path(directory)
        dirs[:] = sorted(name for name in dirs if not is_generated(base / name, data_root))
        for name in dirs:
            _safe(base / name)
        for name in sorted(names):
            path = base / name
            if is_generated(path, data_root):
                continue
            _safe(path)
            if not path.is_file():
                raise MigrationError(f"Non-regular raw file: {path}")
            result[path.relative_to(data_root).as_posix()] = _fingerprint(path)
    return result


def _relative(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise MigrationError("Missing raw relative_path identity")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (path.is_absolute() or ":" in normalized
            or any(part in ("", ".", "..") for part in normalized.split("/"))
            or path == PurePosixPath(".") or path.suffix.casefold() != ".mcp"):
        raise MigrationError(f"Unsafe raw relative_path: {value!r}")
    return Path(*path.parts)


def _safe_version(value) -> bool:
    return (isinstance(value, str) and bool(value) and value not in (".", "..")
            and not any(char in value for char in '/\\:<>"|?*')
            and value.rstrip(". ") == value)


def _identity(document: dict, data_root: Path, *, require_both=False) -> Path | None:
    relative = _relative(document["relative_path"]) if document.get("relative_path") else None
    value = document.get("input_path")
    if value is not None:
        if not _absolute(value):
            raise MigrationError(f"Raw input_path must be absolute: {value!r}")
        raw = _path(value)
        if not _inside(raw, data_root) or is_generated(raw, data_root) or not raw.is_file():
            raise MigrationError(f"Raw input is missing/outside the raw dataset: {value}")
        actual = raw.relative_to(data_root)
        _relative(actual.as_posix())
        if relative is not None and _key(relative) != _key(actual):
            raise MigrationError(f"input_path and relative_path disagree: {value} / {relative}")
        relative = actual
    elif require_both:
        raise MigrationError("Canonical report is missing input_path")
    if require_both and not document.get("relative_path"):
        raise MigrationError("Canonical report is missing relative_path")
    if relative is not None:
        raw = data_root / relative
        _safe(raw)
        if not raw.is_file() or is_generated(raw, data_root):
            raise MigrationError(f"Relative identity is not an existing raw MCP: {relative}")
    return relative


def _small_json(path: Path) -> dict:
    if path.stat().st_size > MAX_IDENTITY:
        raise MigrationError(f"Identity JSON exceeds the 64 MiB fallback bound; install ijson: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeError) as error:
        raise MigrationError(f"Invalid identity JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise MigrationError(f"Identity document must be an object: {path}")
    return value


def _metadata_stream(path: Path) -> dict:
    """Read just canonical identity/artifacts/quality, never large subject arrays."""
    if ijson is None or path.stat().st_size <= MAX_IDENTITY:
        document = _small_json(path)
        return {key: document[key] for key in
                ("input_path", "relative_path", "job_id", "status", "artifacts", "quality")
                if key in document}
    if os.environ.get("MIG_DEBUG_METADATA"):
        print(f"[meta] {path}", file=sys.stderr, flush=True)
    result, builders = {}, {}
    try:
        with path.open("rb") as stream:
            for prefix, event, value in _PARSE(stream, use_float=True):
                if prefix in ("input_path", "relative_path") and event in ("start_map", "start_array"):
                    raise MigrationError(f"Invalid {prefix} metadata type: {path}")
                if prefix in ("input_path", "relative_path", "job_id", "status") and event in ("string", "null", "number", "boolean"):
                    result[prefix] = value
                for field in ("artifacts", "quality"):
                    if prefix == field and event in ("start_map", "start_array"):
                        builders[field] = ijson.common.ObjectBuilder()
                    if field in builders and (prefix == field or prefix.startswith(field + ".")):
                        builders[field].event(event, value)
                        if prefix == field and event in ("end_map", "end_array"):
                            result[field] = builders.pop(field).value
                    elif prefix == field and event in ("null", "string", "number", "boolean"):
                        result[field] = value
    except MigrationError:
        raise
    except (ValueError, ijson.JSONError):
        # Historical diagnostics may contain Python's NaN/Infinity extension.
        document = _small_json(path)
        return {key: document[key] for key in
                ("input_path", "relative_path", "job_id", "status", "artifacts", "quality")
                if key in document}
    return result


def _metadata(path: Path) -> dict:
    return _meta_cached(path, "meta", lambda: _metadata_stream(path))


def _identity_documents_stream(path: Path):
    """Yield object-local explicit identity fields without retaining other arrays."""
    fields = {"input_path", "relative_path", "job_id", "report", "report_path"}
    if ijson is None or path.stat().st_size <= MAX_IDENTITY:
        pending = [_small_json(path)]
        while pending:
            row = pending.pop()
            if isinstance(row, dict):
                yield {key: value for key, value in row.items() if key in fields}
                pending.extend(value for value in row.values() if isinstance(value, (dict, list)))
            elif isinstance(row, list):
                pending.extend(row)
        return
    stack = []
    try:
        with path.open("rb") as stream:
            for prefix, event, value in _PARSE(stream, use_float=True):
                if event == "start_map":
                    stack.append({})
                elif event == "end_map":
                    row = stack.pop()
                    if row:
                        yield row
                elif event in ("string", "null") and stack:
                    field = prefix.rsplit(".", 1)[-1]
                    if field in fields:
                        stack[-1][field] = value
    except (ValueError, ijson.JSONError):
        pending = [_small_json(path)]
        while pending:
            row = pending.pop()
            if isinstance(row, dict):
                yield {key: value for key, value in row.items() if key in fields}
                pending.extend(value for value in row.values() if isinstance(value, (dict, list)))
            elif isinstance(row, list):
                pending.extend(row)


def _meta_cache_paths(path: Path, flavour: str):
    st = path.stat()
    digest = hashlib.sha1(
        f"{flavour}|{path}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8", "surrogateescape")
    ).hexdigest()
    return META_CACHE / digest[:2] / (digest[2:] + ".json")


def _meta_cached(path: Path, flavour: str, compute):
    try:
        key = _meta_cache_paths(path, flavour)
        if key.is_file():
            return json.loads(key.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        key = None
    value = compute()
    if key is not None:
        try:
            key.parent.mkdir(parents=True, exist_ok=True)
            tmp = key.with_suffix(".tmp")
            tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, key)
        except OSError:
            pass
    return value


def _cache_key_extra(path: Path, flavour: str, extra: str) -> Path:
    st = path.stat()
    digest = hashlib.sha1(
        f"{flavour}|{path}|{st.st_size}|{st.st_mtime_ns}|{extra}".encode("utf-8", "surrogateescape")
    ).hexdigest()
    return META_CACHE / digest[:2] / (digest[2:] + ".json")


def _json_identity_fast(path: Path):
    # Identity token streaming reproduces the file's exact bytes (chunk
    # boundaries only re-split tokens), so its output digest equals a raw
    # C-speed file digest.  Hash references appear only as complete JSON
    # strings, which the quoted 64-hex pattern matches verbatim.
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    refs = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        carry = ""
        while True:
            chunk = stream.read(2 * 1024 * 1024)
            if not chunk:
                break
            text = carry + chunk
            for match in HASH_PATTERN.finditer(text):
                refs.add(match.group(1).lower())
            carry = text[-72:]
    return sha, refs


def _json_identity_cached(path: Path):
    """Identity sha256 + hash references of one JSON, journaled across crashes."""
    key = _cache_key_extra(path, "j1", "")
    if key.is_file():
        try:
            sha, refs = json.loads(key.read_text(encoding="utf-8"))
            return sha, set(refs)
        except (OSError, ValueError):
            pass
    if path.stat().st_size >= 1024 * 1024:
        sha, refs = _json_identity_fast(path)
    else:
        sha, refs = _json_stream(path, lambda value: value)
    try:
        key.parent.mkdir(parents=True, exist_ok=True)
        tmp = key.with_suffix(".tmp")
        tmp.write_text(json.dumps([sha, sorted(refs)], ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, key)
    except OSError:
        pass
    return sha, refs


def _identity_documents(path: Path):
    def compute():
        if os.environ.get("MIG_DEBUG_METADATA"):
            print(f"[ident] {path}", file=sys.stderr, flush=True)
        return list(_identity_documents_stream(path))
    yield from _meta_cached(path, "ident", compute)


def _tokens(path: Path):
    """Yield exact JSON tokens; compact only at input-chunk boundaries."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        buffer, cursor, eof, first = "", 0, False, True
        while True:
            if cursor == len(buffer):
                if eof:
                    return
                buffer, cursor = stream.read(CHUNK), 0
                eof = not buffer
                if not buffer:
                    return
            if first:
                first = False
                if buffer.startswith("\ufeff"):
                    yield "bom", "\ufeff"
                    cursor = 1
                    continue
            char = buffer[cursor]
            if char in "{}[]:,":
                yield char, char
                cursor += 1
                continue
            if char in " \t\r\n":
                end = cursor + 1
                while end < len(buffer) and buffer[end] in " \t\r\n":
                    end += 1
                yield "space", buffer[cursor:end]
                cursor = end
                continue
            quoted = char == '"'
            end = cursor + 1 if quoted else cursor
            escaped = False
            while True:
                complete = False
                while end < len(buffer):
                    current = buffer[end]
                    if quoted:
                        if current == '"' and not escaped:
                            end += 1
                            complete = True
                            break
                        escaped = current == "\\" and not escaped
                    elif current in " \t\r\n{}[]:,":
                        complete = True
                        break
                    end += 1
                if end - cursor > MAX_TOKEN:
                    raise MigrationError(f"JSON token exceeds 8 MiB bound: {path}")
                if complete or eof:
                    break
                chunk = stream.read(CHUNK)
                if chunk:
                    buffer = buffer[cursor:] + chunk
                    end -= cursor
                    cursor = 0
                else:
                    eof = True
            text, cursor = buffer[cursor:end], end
            if quoted:
                yield "string", text
            elif text in ("true", "false", "null", "NaN", "Infinity", "-Infinity") or NUMBER.fullmatch(text):
                yield "scalar", text
            else:
                raise MigrationError(f"Invalid JSON token in {path}: {text[:80]!r}")


def _json_stream(path: Path, transform, sink=None) -> tuple[str, set[str]]:
    """Validate structure, transform string values, and hash exact output bytes."""
    digest = hashlib.sha256()
    hashes = set()
    # States distinguish an empty container from a trailing comma.
    stack = [["root", "value"]]
    for kind, text in _tokens(path):
        if kind not in ("space", "bom"):
            container, state = stack[-1]
            if kind == ":":
                if container != "object" or state != "colon":
                    raise MigrationError(f"Unexpected JSON colon: {path}")
                stack[-1][1] = "value"
            elif kind == ",":
                if container not in ("object", "array") or state != "end":
                    raise MigrationError(f"Unexpected JSON comma: {path}")
                stack[-1][1] = "key" if container == "object" else "value"
            elif kind in ("}", "]"):
                valid = ((kind == "}" and container == "object" and state in ("empty-key", "end"))
                         or (kind == "]" and container == "array" and state in ("empty-value", "end")))
                if not valid:
                    raise MigrationError(f"Unexpected JSON container end: {path}")
                stack.pop()
            elif container == "object" and state in ("key", "empty-key"):
                if kind != "string":
                    raise MigrationError(f"Expected JSON object key: {path}")
                stack[-1][1] = "colon"
            else:
                if state not in ("value", "empty-value"):
                    raise MigrationError(f"Unexpected JSON value: {path}")
                stack[-1][1] = "end"
                if kind in ("{", "["):
                    if len(stack) >= 256:
                        raise MigrationError(f"JSON exceeds nesting bound: {path}")
                    stack.append(["object", "empty-key"] if kind == "{" else ["array", "empty-value"])
                elif kind not in ("string", "scalar"):
                    raise MigrationError(f"Invalid JSON value: {path}")
            if kind == "string":
                try:
                    value = json.loads(text)
                except ValueError as error:
                    raise MigrationError(f"Invalid JSON string in {path}: {error}") from error
                if HASH.fullmatch(value):
                    hashes.add(value.lower())
                replacement = transform(value)
                if replacement != value:
                    text = json.dumps(replacement, ensure_ascii=True)
        payload = text.encode("utf-8")
        digest.update(payload)
        if sink is not None:
            sink.write(payload)
    if stack != [["root", "end"]]:
        raise MigrationError(f"Incomplete JSON document: {path}")
    return digest.hexdigest(), hashes


STRING_TOKEN = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"')


def _json_stream_fast(path: Path, transform, sink=None) -> tuple[str, set[str]]:
    # Large-file twin of _json_stream: byte-identical output, but string
    # tokens are found by the C regex engine and spans are hashed in bulk
    # chunks instead of one byte loop.  Chunked so 480 MB reports stream at
    # bounded memory.  Structural validation is intentionally skipped above
    # the size threshold; these documents are machine-made.
    digest = hashlib.sha256()
    hashes = set()

    def emit(payload: str) -> None:
        encoded = payload.encode("utf-8")
        digest.update(encoded)
        if sink is not None:
            sink.write(encoded)

    carry = ""
    with path.open("r", encoding="utf-8", newline="") as stream:
        while True:
            chunk = stream.read(16 * 1024 * 1024)
            text = carry + chunk
            previous = 0
            for match in STRING_TOKEN.finditer(text):
                value = json.loads(match.group(0))
                if HASH.fullmatch(value):
                    hashes.add(value.lower())
                replacement = transform(value)
                inter = text[previous:match.start()]
                if replacement != value:
                    emit(inter + json.dumps(replacement, ensure_ascii=True))
                else:
                    emit(inter + match.group(0))
                previous = match.end()
            if not chunk:
                emit(text[previous:])
                return digest.hexdigest(), hashes
            carry = text[previous:]


def _json_stream_big(path: Path, transform, sink=None) -> tuple[str, set[str]]:
    if path.stat().st_size >= 1024 * 1024:
        return _json_stream_fast(path, transform, sink)
    return _json_stream(path, transform, sink)


class Mappings(list):
    # Prefix table with a compiled view: the inventory and rewrite passes
    # probe tens of thousands of paths, and re-normalizing every mapping
    # source per probe dominated the plan phase.
    def append(self, item):
        list.append(self, item)
        self._compiled = None

    def sort(self, *args, **kwargs):
        list.sort(self, *args, **kwargs)
        self._compiled = None

    @property
    def compiled(self):
        if getattr(self, "_compiled", None) is None:
            entries = []
            for entry in self:
                source = entry["source"].replace("\\", "/").rstrip("/")
                entries.append((_key(source), len(source), str(entry["target"])))
            entries.sort(key=lambda item: -len(item[0]))
            self._compiled = entries
        return self._compiled


def _rewrite(value: str, mappings, digests: dict[str, str]) -> str:
    if HASH.fullmatch(value) and value.lower() in digests:
        return digests[value.lower()]
    if not _absolute(value):
        return value
    normalized = os.path.normpath(value.replace("\\", "/")).replace("\\", "/")
    key = _key(normalized)
    if isinstance(mappings, Mappings):
        entries = mappings.compiled
        for prefix, source_length, target in entries:
            if key == prefix or key.startswith(prefix + "/"):
                suffix = normalized.rstrip("/")[source_length:].lstrip("/")
                return str(Path(target).joinpath(*suffix.split("/"))) if suffix else target
        return value
    for entry in mappings:  # Pre-sorted longest prefix first.
        prefix = _key(entry["source"])
        if key == prefix or key.startswith(prefix + "/"):
            source_length = len(entry["source"].replace("\\", "/").rstrip("/"))
            suffix = normalized.rstrip("/")[source_length:].lstrip("/")
            return str(Path(entry["target"]).joinpath(*suffix.split("/"))) if suffix else entry["target"]
    return value


def _mapped(path: Path, mappings: list[dict]) -> Path:
    return Path(_rewrite(str(path), mappings, {}))


def _rename_no_replace(source: Path, target: Path) -> None:
    """Atomic no-clobber rename, including on POSIX where os.rename overwrites."""
    if os.name == "nt":
        os.rename(source, target)
        return
    import ctypes
    library = ctypes.CDLL(None, use_errno=True)
    operation = getattr(library, "renameat2", None)
    if operation is None:
        raise MigrationError("This platform lacks atomic renameat2(RENAME_NOREPLACE); apply on Windows/Linux")
    operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    operation.restype = ctypes.c_int
    if operation(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def _transform_object(value, mappings, digests):
    if isinstance(value, str):
        return _rewrite(value, mappings, digests)
    if isinstance(value, list):
        return [_transform_object(item, mappings, digests) for item in value]
    if isinstance(value, dict):
        return {key: _transform_object(item, mappings, digests) for key, item in value.items()}
    return value


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _save(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    _safe(path)
    _safe(temporary)
    # This temporary belongs solely to this journal; interrupted saves may leave it.
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _configuration(data_root, sources, control_root, optional_sources, supplemental=None) -> dict:
    supplemental = supplemental or {}
    moves = []
    for entry in supplemental.get("moves", []):
        moves.append({"source": str(_path(entry["source"])), "target": str(_path(entry["target"]))})
    controls = []
    for entry in supplemental.get("controls", []):
        controls.append({"source": str(_path(entry["source"])), "target": str(_path(entry["target"])),
                         "version_key": entry["version_key"]})
    return {"data_root": str(_path(data_root)), "control_root": str(_path(control_root)),
            "source_outputs": sorted({str(_path(path)) for path in sources}, key=_key),
            "optional_source_outputs": sorted({str(_path(path)) for path in optional_sources}, key=_key),
            "supplemental": {"moves": sorted(moves, key=lambda row: _key(row["source"])),
                             "controls": sorted(controls, key=lambda row: _key(row["source"])),
                             "warnings": list(supplemental.get("warnings", [])),
                             "mesh_reports": list(supplemental.get("mesh_reports", []))}}


def _journal_path(config: dict) -> Path:
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return Path(config["control_root"]) / "_migration_history" / identity / "journal.json"


def _order_moves(operations: list[dict]) -> list[dict]:
    """Respect both source extraction and destination installation prerequisites."""
    sources = [Path(row.get("runtime_source", row["source"])) for row in operations]
    targets = [Path(row["target"]) for row in operations]
    prerequisites = [set() for _ in operations]
    for index, operation in enumerate(operations):
        for other in range(len(operations)):
            if index == other:
                continue
            # An extracted child must leave before its source ancestor moves.
            if _inside(sources[index], sources[other]):
                prerequisites[other].add(index)
            # Otherwise mkdir for a child target would occupy its ancestor's
            # rename destination. Install that ancestor atomically first.
            if _inside(targets[index], targets[other]):
                prerequisites[index].add(other)
            # Deferred unassigned entries travel with a root, then move out of
            # its destination; their runtime source does not exist beforehand.
            if operation.get("runtime_source") and _inside(sources[index], targets[other]):
                prerequisites[index].add(other)
    ordered, visiting, finished = [], set(), set()

    def finish(index):
        if index in finished:
            return
        if index in visiting:
            raise MigrationError(f"Cyclic move prerequisites: {operations[index]['source']}")
        visiting.add(index)
        for predecessor in sorted(prerequisites[index]):
            finish(predecessor)
        visiting.remove(index)
        finished.add(index)
        ordered.append(operations[index])

    for index in range(len(operations)):
        finish(index)
    return ordered


def plan_migration(data_root, source_outputs, control_root, optional_source_outputs=(), *, supplemental=None) -> dict:
    """Read-only plan. Supplemental moves bind before JSON dependency hashing.

    supplemental = {"moves": [{"source": absolute, "target": absolute}, ...],
                    "controls": [{"source": old_control, "target": new_control,
                                  "version_key": unique_basename}, ...],
                    "warnings": ["explicit unresolved evidence", ...],
                    "mesh_reports": [{"report_path": new_canonical_path,
                                      "input_path": raw_mcp, "relative_path": raw_relative,
                                      "purpose": "mesh", "artifacts": {...}, ...}, ...]}
    Extra move sources must lie under explicitly selected source roots. Every
    extra root's complete content is inventoried; child moves override its prefix.
    A whole standalone root may move to a local diagnostics directory while its
    separate, empty metadata control is initialized at controls[].target.
    Identity attribution is the supplemental inventory owner's responsibility;
    this engine still validates containment, collisions, bytes and resumability.
    """
    config = _configuration(data_root, source_outputs, control_root, optional_source_outputs, supplemental)
    journal = _journal_path(config)
    if journal.exists():
        raise MigrationError(f"A journal already exists; resume through migrate(): {journal}")
    data_root, control_root = Path(config["data_root"]), Path(config["control_root"])
    if not data_root.is_dir():
        raise MigrationError(f"Missing raw data root: {data_root}")
    if (_inside(control_root, data_root) and not is_generated(control_root, data_root)) or control_root == data_root:
        raise MigrationError("Control/history root must be outside raw roots or inside _processing/processed")
    warnings, diagnostics, runs, roots = [], [], [], []
    warnings.extend(config["supplemental"]["warnings"])
    selected_roots = []
    metadata_roots = set()
    candidates = [(Path(value), False) for value in config["source_outputs"]]
    candidates += [(Path(value), True) for value in config["optional_source_outputs"]]
    if not candidates:
        raise MigrationError("At least one source-output is required")
    seen_sources = set()
    for root, optional in candidates:
        if _key(root) in seen_sources:
            raise MigrationError(f"Repeated source root: {root}")
        seen_sources.add(_key(root))
        if not root.exists():
            if optional:
                warnings.append(f"Optional source root is absent and untouched: {root}")
                continue
            raise MigrationError(f"Required source root is missing: {root}")
        if not root.is_dir():
            raise MigrationError(f"Source root is not a directory: {root}")
        if (_inside(root, data_root) and not is_generated(root, data_root)) or _inside(data_root, root):
            raise MigrationError(f"Source root overlaps original raw tree: {root}")
        if _inside(control_root, root) or _inside(root, control_root):
            raise MigrationError(f"Source/control overlap is forbidden: {root}")
        all_files = _walk(root)
        if os.environ.get("MIG_DEBUG_METADATA"):
            print(f"[scan] {root} files={len(all_files)}", file=sys.stderr, flush=True)
        selected_roots.append(root)
        controls = [root]
        index = 0
        while index < len(controls):
            repairs = controls[index] / "_repair_runs"
            if repairs.exists():
                for child in sorted(repairs.iterdir()):
                    if child.is_dir():
                        controls.append(child)
            index += 1
        reports = {control: sorted(path for path in all_files
                                  if path.name == "report.json"
                                  and path.parent.suffix.casefold() == ".mcp"
                                  and _inside(path, control / "takes"))
                   for control in controls}
        if not any(reports.values()):
            diagnostics.append({"root": str(root), "optional": optional,
                                "files": len(all_files), "bytes": sum(p.stat().st_size for p in all_files)})
            warnings.append(f"Standalone diagnostic root has no canonical take reports; left untouched: {root}")
            continue
        if any(_inside(root, previous) or _inside(previous, root) for previous in roots):
            raise MigrationError(f"Overlapping source run roots: {root}")
        roots.append(root)
        metadata_roots.add(root)
        target_root = control_root / root.name
        for control in controls:
            target = target_root / control.relative_to(root)
            if not _safe_version(control.name):
                raise MigrationError(f"Unsafe version basename: {control.name}")
            if (control / ".sequence-layout.json").exists():
                raise MigrationError(f"Source already declares a sequence-local layout: {control}")
            runs.append({"source": str(control), "target": str(target), "version_key": control.name,
                         "reports": reports[control]})
    extra_moves = config["supplemental"]["moves"]
    for entry in extra_moves:
        source, target = Path(entry["source"]), Path(entry["target"])
        if not source.exists() or not any(_inside(source, root) for root in selected_roots):
            raise MigrationError(f"Supplemental move source is not in an explicit source root: {source}")
        if not (_inside(target, control_root) or (_inside(target, data_root) and is_generated(target, data_root))):
            raise MigrationError(f"Supplemental target is not generated sequence/control storage: {target}")
        if _inside(target, source) or _inside(source, target):
            raise MigrationError(f"Supplemental move overlaps itself: {source} -> {target}")
    extra_sources = [Path(entry["source"]) for entry in extra_moves]
    for source in extra_sources:
        if not any(_inside(source, root) for root in roots) and not any(
                other != source and _inside(source, other) for other in extra_sources):
            roots.append(source)
    adopted = {row["root"] for row in diagnostics if any(
        _inside(Path(row["root"]), root) for root in roots)}
    diagnostics = [row for row in diagnostics if row["root"] not in adopted]
    warnings = [warning for warning in warnings if not any(
        warning.endswith(": " + root) and "left untouched" in warning for root in adopted)]
    for entry in config["supplemental"]["controls"]:
        source, target = Path(entry["source"]), Path(entry["target"])
        if not any(_inside(source, root) for root in selected_roots) or not _inside(target, control_root):
            raise MigrationError(f"Supplemental control is outside declared roots: {source} -> {target}")
        existing = next((run for run in runs if _key(run["source"]) == _key(source)), None)
        if existing is not None:
            if existing["target"] != str(target) or existing["version_key"] != entry["version_key"]:
                raise MigrationError(f"Conflicting supplemental control: {source}")
            continue
        runs.append({**entry, "reports": [], "supplemental": True})
    versions = {}
    for run in runs:
        if not _safe_version(run["version_key"]):
            raise MigrationError(f"Unsafe version key: {run['version_key']!r}")
        if run["version_key"] != Path(run["target"]).name:
            raise MigrationError(f"Version key must equal new control basename: {run['target']}")
        if Path(run["target"]).exists():
            raise MigrationError(f"Refusing existing new control: {run['target']}")
        key = run["version_key"].casefold()
        if key in versions:
            raise MigrationError(f"Version basename collision: {versions[key]} and {run['source']}")
        versions[key] = run["source"]
    mappings, operations, canonical, artifacts = Mappings(), [], [], []
    identities = {}
    deferred = []

    def add_mapping(source: Path, target: Path, move=True):
        _safe(source)
        _safe(target)
        previous = next((entry for entry in mappings if _key(entry["source"]) == _key(source)), None)
        if previous is not None:
            if _key(previous["target"]) != _key(target):
                raise MigrationError(f"Conflicting source mappings: {source}")
            return
        mappings.append({"source": str(source), "target": str(target)})
        if move and source.exists():
            operations.append({"source": str(source), "target": str(target),
                               "identity": _fingerprint(source)})

    take_root_index: list[tuple[str, str]] = []


    def paths_for(run, relative):
        identity = (run["source"], relative.as_posix())
        if identity not in identities:
            paths = take_paths(Path(run["target"]), relative, data_root=data_root,
                               version_key=run["version_key"])
            identities[identity] = paths
            take_root_index.append((str(paths["take_root"]), relative.as_posix()))
            source = Path(run["source"])
            add_mapping(source / "takes" / relative, paths["take_root"])
            add_mapping(source / "_needs_review" / relative, paths["review_destination"].parent)
            add_mapping(source / "logs" / relative, paths["log_dir"])
        return identities[identity]

    for root in sorted(metadata_roots, key=_key):
        add_mapping(root, control_root / root.name, move=False)
    for entry in extra_moves:
        add_mapping(Path(entry["source"]), Path(entry["target"]))
    for run in runs:
        if run.get("supplemental"):
            continue
        source = Path(run["source"])
        if source not in roots:
            add_mapping(source, Path(run["target"]), move=False)
        jobs = {}
        for report_path in run["reports"]:
            document = _metadata(report_path)
            legacy_failed = document.get("status") == "FAILED" and "relative_path" not in document
            try:
                relative = _identity(document, data_root, require_both=not legacy_failed)
            except MigrationError as error:
                # A user-deleted historical raw file is not another take. Preserve
                # it under unassigned, while still refusing escapes/mismatches.
                # Do not let a second, missing-field error hide the report path
                # or the original identity failure.
                message = f"Invalid canonical report identity: {report_path}: {error}"
                try:
                    relative = _relative(document.get("relative_path"))
                    raw_value = document.get("input_path")
                    raw = _path(raw_value) if _absolute(raw_value) else None
                except MigrationError:
                    raise MigrationError(message) from error
                if (raw is None or raw != data_root / relative or raw.exists()
                        or is_generated(raw, data_root)
                        or _key(report_path.parent.relative_to(source / "takes")) != _key(relative)):
                    raise MigrationError(message) from error
                for bucket in ("takes", "_needs_review", "logs"):
                    old = source / bucket / relative
                    target = Path(run["target"]) / "unassigned" / bucket / relative
                    add_mapping(old, target, move=False)
                    if old.exists():
                        deferred.append({"source": str(old), "target": str(target),
                                         "runtime_source": str(Path(run["target"]) / bucket / relative),
                                         "identity": _fingerprint(old)})
                warnings.append(f"Deleted raw identity; preserved under new control/unassigned, never attributed to a sibling: {report_path}")
                continue
            if relative is None or _key(report_path.parent.relative_to(source / "takes")) != _key(relative):
                raise MigrationError(f"Canonical report location/identity mismatch: {report_path}")
            paths = paths_for(run, relative)
            job_id = document.get("job_id")
            if job_id:
                if job_id in jobs and jobs[job_id] != relative:
                    raise MigrationError(f"Ambiguous explicit job identity: {job_id}")
                jobs[job_id] = relative
            canonical.append({"source": str(report_path), "target": str(paths["report_path"]),
                              "relative_path": relative.as_posix(), "control": run["target"],
                              "status": document.get("status"), "quality": document.get("quality")})
            if legacy_failed:
                canonical[-1]["identity_evidence"] = "failed_input_path_and_take_location"
                warnings.append(f"Legacy FAILED report has no relative_path; retained canonical using exact raw input_path and takes location: {report_path}")
            products = document.get("artifacts", {})
            if not isinstance(products, dict):
                raise MigrationError(f"Invalid canonical artifacts: {report_path}")
            for value in products.values():
                if not _absolute(value):
                    raise MigrationError(f"Canonical artifact must be absolute: {value!r}")
                path = _path(value)
                if not path.is_file() or not any(_inside(path, root) for root in roots):
                    raise MigrationError(f"Missing/outside canonical artifact: {path}")
                artifacts.append({"source": str(path), "take_root": str(paths["take_root"]),
                                  "control": run["target"], "source_control": run["source"],
                                  "relative": relative.as_posix()})

        def explicit_identity(directory: Path) -> Path | None:
            found = set()
            evidence = []
            for file in _walk(directory) if directory.is_dir() else [directory]:
                if (file.name in ("attempt.json", "session.json", "session-ready.json", "result.json", "job.json", "control.json")
                        or (file.name == "report.json" and "_resume_history" in file.parts)):
                    evidence.append(file)
            for file in evidence:
                try:
                    rows = list(_identity_documents(file))
                except MigrationError as error:
                    warnings.append(f"Unusable attempt identity; retained unless other explicit evidence resolves it: {error}")
                    continue
                for row in rows:
                    # Never accept a raw identity inferred from directory nicknames.
                    if row.get("input_path") or row.get("relative_path"):
                        try:
                            identity = _identity(row, data_root)
                        except MigrationError:
                            value = row.get("input_path")
                            relative_value = row.get("relative_path")
                            relative = _relative(relative_value) if relative_value else None
                            raw = _path(value) if _absolute(value) else (data_root / relative if relative else None)
                            if (raw is None or not _inside(raw, data_root) or raw.exists()
                                    or is_generated(raw, data_root) or raw.suffix.casefold() != ".mcp"):
                                raise
                            if relative is not None and raw != data_root / relative:
                                raise
                            warnings.append(f"Deleted raw attempt identity; retained batch-level: {file}")
                            continue
                        if identity is not None:
                            found.add(identity)
                    if row.get("job_id") in jobs:
                        found.add(jobs[row["job_id"]])
                    for field in ("report", "report_path"):
                        if _absolute(row.get(field)):
                            for record in canonical:
                                if _key(row[field]) == _key(record["source"]) and record["control"] == run["target"]:
                                    found.add(Path(record["relative_path"]))
            if len(found) > 1:
                raise MigrationError(f"Conflicting explicit attempt identities: {directory}: {sorted(map(str, found))}")
            return next(iter(found)) if found else None

        for bucket, destination in BUCKETS.items():
            directory = source / bucket
            if not directory.exists():
                continue
            for entry in sorted(directory.iterdir()):
                relative = explicit_identity(entry)
                if relative is None:
                    warnings.append(f"Unknown job identity; retained batch-level under new control: {entry}")
                    continue
                paths = paths_for(run, relative)
                add_mapping(entry, paths[destination] / entry.name)
        history = source / "_resume_history"
        if history.exists():
            moved_history = set()
            for file in _walk(history):
                if file.name != "report.json":
                    continue
                relative = explicit_identity(file)
                if relative is None:
                    continue
                directory = file.parent
                if any(_inside(directory, prior) or _inside(prior, directory) for prior in moved_history):
                    raise MigrationError(f"Overlapping history ownership: {directory}")
                moved_history.add(directory)
                paths = paths_for(run, relative)
                add_mapping(directory, paths["history_root"] / directory.relative_to(history))
            leftovers = [file for file in _walk(history)
                         if not any(_inside(file, directory) for directory in moved_history)]
            if leftovers:
                warnings.append(f"Unassigned history retained batch-level: {history} ({len(leftovers)} files)")
    mappings.sort(key=lambda entry: len(_key(entry["source"])), reverse=True)
    for artifact in artifacts:
        target = _mapped(Path(artifact["source"]), mappings)
        if not _inside(target, Path(artifact["take_root"])):
            old = Path(artifact["source"])
            old_control = Path(artifact["source_control"])
            unknown_attempt = any(_inside(old, old_control / bucket) for bucket in BUCKETS)
            same_take = next((rel for root, rel in take_root_index
                              if _inside(target, Path(root))
                              and _key(rel) == _key(artifact["relative"])), None)
            if same_take is not None:
                warnings.append(f"Canonical artifact references another version of the same take; remapped through that version's mapping: {old}")
            elif not unknown_attempt or not _inside(target, Path(artifact["control"])):
                if os.environ.get("MIG_DEBUG_METADATA"):
                    for entry in mappings:
                        if _key(entry["source"]) in _key(artifact["source"]):
                            print("[anchor-debug] mapping", entry, file=sys.stderr, flush=True)
                    print("[anchor-debug] take_root", artifact["take_root"],
                          "control", artifact["control"],
                          "mapped", str(target), file=sys.stderr, flush=True)
                raise MigrationError(f"Canonical artifact has no explicit same-take mapping: {artifact['source']}")
            warnings.append(f"Canonical artifact belongs to an unidentified attempt; retained batch-level without guessed association: {old}")
        artifact["target"] = str(target)
    # Prefer child extraction, subject to destination-ancestor prerequisites.
    operations.sort(key=lambda entry: len(Path(entry["source"]).parts), reverse=True)
    for root in sorted(metadata_roots, key=_key):
        operations.append({"source": str(root), "target": str(control_root / root.name),
                           "identity": _fingerprint(root)})
    operations.extend(deferred)
    targets = set()
    for operation in operations:
        source, target = Path(operation["source"]), Path(operation["target"])
        if _key(target) in targets or target.exists():
            raise MigrationError(f"Refusing target collision: {target}")
        targets.add(_key(target))
        ancestor = target.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        if source.stat().st_dev != ancestor.stat().st_dev:
            raise MigrationError(f"Cross-volume move is forbidden: {source} -> {target}")
        for other in operations:
            parent_target = Path(other["target"])
            if other is operation or not _inside(target, parent_target):
                continue
            projected = Path(other["source"]) / target.relative_to(parent_target)
            if projected.exists():
                raise MigrationError(f"Projected target collision: {projected} -> {target}")
    operations = _order_moves(operations)
    files, json_files, projected_files = [], [], set()
    for root in roots:
        for file in _walk(root):
            target = _mapped(file, mappings)
            if _key(target) in projected_files:
                raise MigrationError(f"Two source files map to one destination: {target}")
            projected_files.add(_key(target))
            record = {"source": str(file), "target": str(target), "identity": _fingerprint(file)}
            if file.suffix.casefold() == ".json" or file.name == ".pipeline.lock":
                record["sha256"], references = _json_identity_cached(file)
                record["hash_references"] = sorted(references)
                json_files.append(record)
            else:
                files.append(record)
    by_hash = {}
    for record in json_files:
        by_hash.setdefault(record["sha256"], []).append(record)
    finished, visiting, ordered = {}, set(), []

    def finish(record):
        source = record["source"]
        if source in finished:
            return
        if source in visiting:
            raise MigrationError(f"Cyclic JSON hash dependencies: {source}")
        visiting.add(source)
        for reference in record["hash_references"]:
            for dependency in by_hash.get(reference, []):
                finish(dependency)
        extra = hashlib.sha1(json.dumps(
            {"maps": mappings,
             "deps": sorted((r, finished_hashes.get(r)) for r in record["hash_references"])},
            sort_keys=True, default=str).encode("utf-8", "surrogateescape")).hexdigest()
        new_hash = None
        try:
            key2 = _cache_key_extra(Path(source), "j2", extra)
            if key2.is_file():
                new_hash = json.loads(key2.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            new_hash = None
        if new_hash is None:
            new_hash, _ = _json_stream_big(Path(source), lambda value: _rewrite(value, mappings, finished_hashes))
            try:
                key2.parent.mkdir(parents=True, exist_ok=True)
                tmp = key2.with_suffix(".tmp")
                tmp.write_text(json.dumps(new_hash), encoding="utf-8")
                os.replace(tmp, key2)
            except OSError:
                pass
        previous = finished_hashes.get(record["sha256"])
        if previous is not None and previous != new_hash:
            raise MigrationError(f"Identical original JSON has inconsistent migrated hashes: {source}")
        finished_hashes[record["sha256"]] = new_hash
        record["migrated_sha256"] = new_hash
        finished[source] = new_hash
        visiting.remove(source)
        ordered.append(record)
        if os.environ.get("MIG_DEBUG_METADATA") and len(ordered) % 256 == 0:
            print(f"[finish] {len(ordered)}/{len(json_files)}", file=sys.stderr, flush=True)

    finished_hashes = {}
    for record in json_files:
        finish(record)
    new_json = []
    for payload in config["supplemental"]["mesh_reports"]:
        target = _path(payload["report_path"])
        document = {key: value for key, value in payload.items() if key != "report_path"}
        document = _transform_object(document, mappings, finished_hashes)
        document.setdefault("status", "MIGRATED_MESH")
        relative = _identity(document, data_root, require_both=True)
        allowed = [take_paths(Path(run["target"]), relative, data_root=data_root,
                              version_key=run["version_key"])["report_path"]
                   for run in runs]
        if target not in allowed or target.exists() or _key(target) in projected_files:
            raise MigrationError(f"Generated mesh report target is invalid or occupied: {target}")
        projected_files.add(_key(target))
        new_json.append({"target": str(target), "document": document,
                         "sha256": hashlib.sha256(_json_bytes(document)).hexdigest()})
    for record in ordered + new_json:
        target = Path(record["target"])
        temporary = target.with_name("." + target.name + ".migration-" + journal.parent.name + ".tmp")
        _safe(temporary)
        if temporary.exists() or _key(temporary) in projected_files:
            raise MigrationError(f"Reserved migration temporary collides with existing data: {temporary}")
    return {"schema_version": SCHEMA_VERSION, "configuration": config, "journal": str(journal),
            "phase": "planned", "warnings": warnings, "untouched_diagnostics": diagnostics,
            "control_mapping": [{"source": run["source"], "target": run["target"],
                                 "version_key": run["version_key"]} for run in runs],
            "path_mapping": mappings, "operations": operations,
            "canonical_reports": canonical, "artifacts": artifacts,
            "unchanged_files": files, "json_files": ordered, "new_json_files": new_json,
            "raw_baseline": _raw_baseline(data_root)}


def _verify_raw(plan: dict) -> None:
    current = _raw_baseline(Path(plan["configuration"]["data_root"]))
    if current != plan["raw_baseline"]:
        changed = sorted(key for key in current.keys() | plan["raw_baseline"].keys()
                         if current.get(key) != plan["raw_baseline"].get(key))
        raise MigrationError(f"Raw file baseline changed: {changed[:20]}")


def _archive_path(journal: Path, index: int) -> Path:
    return journal.parent / "original_json" / f"{index:08d}.json"


def _verify_results(plan: dict) -> None:
    _verify_raw(plan)
    for record in plan["unchanged_files"]:
        path = Path(record["target"])
        _safe(path)
        if not path.is_file() or _fingerprint(path) != record["identity"]:
            raise MigrationError(f"Moved file identity/size/mtime changed: {path}")
    for record in plan["json_files"]:
        path = Path(record["target"])
        _safe(path)
        if not path.is_file() or _digest(path) != record["migrated_sha256"]:
            raise MigrationError(f"Migrated JSON digest mismatch: {path}")
    for record in plan.get("new_json_files", []):
        path = Path(record["target"])
        _safe(path)
        if not path.is_file() or _digest(path) != record["sha256"]:
            raise MigrationError(f"Generated mesh report digest mismatch: {path}")
        for artifact in record["document"].get("artifacts", {}).values():
            if not Path(artifact).is_file():
                raise MigrationError(f"Generated mesh artifact is missing: {artifact}")
    for artifact in plan["artifacts"]:
        path = Path(artifact["target"])
        _safe(path)
        if not path.is_file():
            raise MigrationError(f"Mapped canonical artifact is missing: {path}")
    data_root = Path(plan["configuration"]["data_root"])
    for record in plan["canonical_reports"]:
        path = Path(record["target"])
        document = _metadata(path)
        legacy_failed = record.get("identity_evidence") == "failed_input_path_and_take_location"
        try:
            relative = _identity(document, data_root, require_both=not legacy_failed)
        except MigrationError as error:
            raise MigrationError(f"Invalid migrated canonical identity: {path}: {error}") from error
        if (relative is None or _key(relative) != _key(record["relative_path"])
                or (legacy_failed and (document.get("status") != "FAILED" or "relative_path" in document))):
            raise MigrationError(f"Migrated canonical identity evidence changed: {path}")
        expected = take_paths(Path(record["control"]), relative, data_root=data_root,
                              version_key=next(entry["version_key"] for entry in plan["control_mapping"]
                                               if entry["target"] == record["control"]))["report_path"]
        if path != expected or document.get("status") != record["status"] or document.get("quality") != record["quality"]:
            raise MigrationError(f"Canonical identity/location/status/quality changed: {path}")
        for value in document.get("artifacts", {}).values():
            if not Path(value).is_file():
                raise MigrationError(f"Canonical artifact reference is invalid: {path}: {value}")


def apply_migration(plan: dict) -> dict:
    """Journal first; archive JSON; rename; rewrite dependencies; validate; resume."""
    journal = _path(plan["journal"])
    if journal.exists():
        saved = _small_journal(journal)
        if saved["configuration"] != plan["configuration"]:
            raise MigrationError(f"Journal configuration mismatch: {journal}")
        plan = saved
    else:
        _verify_raw(plan)
        # Recheck the snapshot before first mutation, not only the raw baseline.
        for record in plan["unchanged_files"] + plan["json_files"]:
            path = Path(record["source"])
            _safe(path)
            if _fingerprint(path) != record["identity"]:
                raise MigrationError(f"Source changed since planning: {path}")
            if "sha256" in record and _digest(path) != record["sha256"]:
                raise MigrationError(f"Source JSON changed since planning: {path}")
        for operation in plan["operations"]:
            _safe(Path(operation["target"]))
            if Path(operation["target"]).exists():
                raise MigrationError(f"Refusing target collision: {operation['target']}")
        for record in plan.get("new_json_files", []):
            target = Path(record["target"])
            _safe(target)
            if target.exists():
                raise MigrationError(f"Refusing generated report collision: {target}")
        journal.parent.mkdir(parents=True, exist_ok=False)
        _sync_directory(journal.parent.parent)
        _save(journal, plan)  # Durable before the first data move.
    _verify_raw(plan)
    if plan["phase"] == "complete":
        _verify_results(plan)
        return plan
    archive = journal.parent / "original_json"
    _safe(archive)
    archive.mkdir(exist_ok=True)
    for index, record in enumerate(plan["json_files"]):
        original = _archive_path(journal, index)
        _safe(original)
        if original.exists():
            if _digest(original) != record["sha256"]:
                raise MigrationError(f"Original JSON history is corrupt: {original}")
            continue
        if plan["phase"] != "planned":
            raise MigrationError(f"Original JSON history is missing after moves: {original}")
        source = Path(record["source"])
        _safe(source)
        if _digest(source) != record["sha256"]:
            raise MigrationError(f"Source JSON changed before archival: {source}")
        temporary = original.with_suffix(".tmp")
        _safe(temporary)
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, CHUNK)
            writer.flush()
            os.fsync(writer.fileno())
        if _digest(temporary) != record["sha256"]:
            raise MigrationError(f"Original JSON archive copy mismatch: {source}")
        _rename_no_replace(temporary, original)
        _sync_directory(archive)
    if plan["phase"] == "planned":
        plan["phase"] = "moving"
        _save(journal, plan)
    for operation in plan["operations"]:
        source = Path(operation.get("runtime_source", operation["source"]))
        target = Path(operation["target"])
        _safe(source)
        _safe(target)
        if operation.get("done"):
            continue
        if source.exists():
            if target.exists():
                raise MigrationError(f"Refusing target collision during apply: {target}")
            # Directory mtimes change as children move; identity must not.
            current = _fingerprint(source)
            if any(current[key] != operation["identity"][key] for key in ("device", "inode")):
                raise MigrationError(f"Move source identity changed: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            _safe(target.parent)
            if source.stat().st_dev != target.parent.stat().st_dev:
                raise MigrationError(f"Cross-volume move is forbidden: {source} -> {target}")
            _rename_no_replace(source, target)
            _sync_directory(source.parent)
            _sync_directory(target.parent)
        elif target.exists():
            current = _fingerprint(target)
            if any(current[key] != operation["identity"][key] for key in ("device", "inode")):
                raise MigrationError(f"Interrupted move target has wrong identity: {target}")
        else:
            raise MigrationError(f"Both move endpoints are missing: {source} -> {target}")
        operation["done"] = True
    plan["phase"] = "rewriting"
    _save(journal, plan)
    digest_map = {record["sha256"]: record["migrated_sha256"] for record in plan["json_files"]}
    path_mapping = Mappings(plan["path_mapping"])
    for index, record in enumerate(plan["json_files"]):
        target = Path(record["target"])
        _safe(target)
        current = _digest(target)
        if current == record["migrated_sha256"]:
            continue
        if current != record["sha256"]:
            raise MigrationError(f"JSON changed outside migration: {target}")
        temporary = target.with_name("." + target.name + ".migration-" + journal.parent.name + ".tmp")
        _safe(temporary)
        with temporary.open("wb") as stream:
            digest, _ = _json_stream_big(_archive_path(journal, index),
                                         lambda value: _rewrite(value, path_mapping, digest_map), stream)
            stream.flush()
            os.fsync(stream.fileno())
        if digest != record["migrated_sha256"]:
            raise MigrationError(f"Planned JSON rewrite hash mismatch: {target}")
        if _digest(target) != record["sha256"]:
            raise MigrationError(f"JSON changed while rewriting: {target}")
        os.replace(temporary, target)  # Only the journaled, intentionally rewritten JSON.
        _sync_directory(target.parent)
    for record in plan.get("new_json_files", []):
        target = Path(record["target"])
        _safe(target)
        if target.exists():
            if _digest(target) != record["sha256"]:
                raise MigrationError(f"Generated JSON collision: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name("." + target.name + ".migration-" + journal.parent.name + ".tmp")
        _safe(temporary)
        with temporary.open("wb") as stream:
            stream.write(_json_bytes(record["document"]))
            stream.flush()
            os.fsync(stream.fileno())
        _rename_no_replace(temporary, target)
        _sync_directory(target.parent)
    plan["phase"] = "validating"
    _save(journal, plan)
    for control in plan["control_mapping"]:
        initialize_layout(Path(control["target"]), Path(plan["configuration"]["data_root"]), control["version_key"])
    _verify_results(plan)
    plan["phase"] = "complete"
    _save(journal, plan)
    return plan


def _small_journal(path: Path) -> dict:
    # The journal contains file records, never the (potentially 480 MB) JSON bodies.
    _safe(path)
    with path.open(encoding="utf-8") as stream:
        plan = json.load(stream)
    if plan.get("schema_version") != SCHEMA_VERSION or str(path) != plan.get("journal"):
        raise MigrationError(f"Invalid migration journal: {path}")
    return plan


def migrate(data_root, source_outputs, control_root, *, optional_source_outputs=(), supplemental=None, apply=False) -> dict:
    config = _configuration(data_root, source_outputs, control_root, optional_source_outputs, supplemental)
    journal = _journal_path(config)
    if journal.exists():
        plan = _small_journal(journal)
        if config != plan["configuration"]:
            raise MigrationError(f"Journal configuration mismatch: {journal}")
    else:
        plan = plan_migration(data_root, source_outputs, control_root, optional_source_outputs, supplemental=supplemental)
    return apply_migration(plan) if apply else plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("H:/MocapData"))
    parser.add_argument("--source-output", action="append", type=Path, default=[], help="Required centralized run root (repeatable)")
    parser.add_argument("--optional-source-output", action="append", type=Path, default=[], help="User-selected root; absence is only a warning")
    parser.add_argument("--control-root", type=Path, default=Path("H:/MocapData/_processing"))
    parser.add_argument("--supplemental-plan", type=Path, help="Evidence-checked explicit moves/controls JSON, bound into the journal")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Read-only exact plan (the default)")
    mode.add_argument("--apply", action="store_true", help="Apply or resume the durable journal")
    args = parser.parse_args(argv)
    try:
        supplemental = _small_json(_path(args.supplemental_plan)) if args.supplemental_plan else None
        plan = migrate(args.data_root, args.source_output, args.control_root,
                       optional_source_outputs=args.optional_source_output, supplemental=supplemental, apply=args.apply)
    except (MigrationError, OSError, UnicodeError, RecursionError) as error:
        print(f"Migration refused: {error}", file=sys.stderr)
        return 2
    # Exact moves, rewrites, inventories, baselines, warnings and registry handoff.
    json.dump(plan, sys.stdout, ensure_ascii=True, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
