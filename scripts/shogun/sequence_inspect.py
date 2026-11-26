"""Cheap per-sequence roster analysis for the workbench inspect endpoint.

Three sources, in order of trust:
  * run report JSON (subjects/props recorded by the authoritative pipeline);
  * direct binary FBX scan via the shared mesh_fbx parser (7700 only);
  * MCP binary subject-record scan (name record followed by an 8-byte stamp
    and the first bone "Hips"). VDF carries no roster of its own.

Nothing here writes; all functions are read-only and bounded.
"""
from __future__ import annotations

from datetime import datetime, timezone
import mmap
import re
from pathlib import Path
import struct

_TIME_UNITS = 46_186_158_000  # FBX KTime ticks per second
# FBX eTimeMode → Hz (common subset); unknown modes fall through to None.
_TIME_MODES = {1: 120.0, 2: 100.0, 3: 60.0, 4: 50.0, 5: 40.0, 6: 30.0,
               7: 29.97, 8: 29.966, 9: 24.0, 10: 25.0, 11: 20.0, 12: 15.0,
               13: 12.0, 14: 10.0, 21: 48.0}
_MAX_FBX_BYTES = 700 * 1024 * 1024
_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{2,31}")

_parser = None

_HUMAN_ONLY_FAMILIES = ("MP", "BODY")


def dataset_family(path) -> str | None:
    """Classify the dataset family from the real ancestor layout."""
    path = Path(path)
    for ancestor in path.parents:
        if ancestor.name.upper() in {"HOI", "MP"} and re.fullmatch(r"group[1-9]\d*", ancestor.parent.name, re.IGNORECASE):
            return ancestor.name.upper()
    for ancestor in path.parents:
        if ancestor.name.lower() == "body":
            return "BODY"
    return None

_P_SCALARS = {ord("i"): "<i", ord("I"): "<i", ord("l"): "<q", ord("L"): "<q",
              ord("K"): "<q", ord("f"): "<f", ord("F"): "<f", ord("d"): "<d",
              ord("D"): "<d", ord("b"): "<b", ord("B"): "<b", ord("Y"): "<h"}


def _p_props(node, buf: bytes) -> list[tuple[str, object]]:
    """Lenient property decode for P/GlobalSettings nodes.

    The shared extract_groups parser rejects FBX 7.7 type chars (K/R). This
    walk covers the scalar/string/array types Property70 entries use and
    stops at the first unknown char instead of raising.
    """
    props = buf[node.cstart - node.proplen:node.cstart]
    values: list[tuple[str, object]] = []
    index = 0
    while index < len(props):
        kind = props[index]
        index += 1
        if kind in (ord("S"), ord("R")):
            if index + 4 > len(props):
                break
            size = struct.unpack_from("<I", props, index)[0]
            index += 4
            raw = props[index:index + size]
            index += size
            if index > len(props):
                break
            values.append((chr(kind), raw.decode("utf-8", "replace") if kind == ord("S") else raw))
        elif kind in _P_SCALARS:
            fmt = _P_SCALARS[kind]
            if index + struct.calcsize(fmt) > len(props):
                break
            values.append((chr(kind), struct.unpack_from(fmt, props, index)[0]))
            index += struct.calcsize(fmt)
        else:
            break
    return values


def _load_parser():
    global _parser
    if _parser is None:
        try:
            from . import mesh_fbx as fbx
        except ImportError:
            import mesh_fbx as fbx
        _parser = (fbx, fbx.load_parser())
    return _parser


def parse_fbx_roster(path: Path) -> dict:
    """Structural scan: models per top-level branch, markers, animation spans."""
    fbx, parser = _load_parser()
    if path.stat().st_size > _MAX_FBX_BYTES:
        raise ValueError(f"FBX larger than {_MAX_FBX_BYTES} bytes is not parsed in the browser path")
    buf = path.read_bytes()
    tops = fbx.parse_checked(parser, buf, allow_unterminated_children=True)
    top = {node.name.decode("ascii"): node for node in tops}
    if "Objects" not in top or "Connections" not in top:
        raise ValueError("FBX is missing the Objects/Connections sections")

    models: dict[int, dict] = {}
    object_counts: dict[str, int] = {}
    for node in top["Objects"].children:
        name = node.name.decode("ascii", "replace")
        object_counts[name] = object_counts.get(name, 0) + 1
        values = [value for _, value in parser.decode_props(buf, node)]
        if not values or not isinstance(values[0], int) or values[0] == 0:
            continue
        if node.name == b"Model" and len(values) == 3:
            models[values[0]] = {"name": str(values[1]).split("\x00", 1)[0], "kind": str(values[2])}

    children: dict[int, list[int]] = {}
    parents: dict[int, int] = {}
    for link in top["Connections"].children:
        values = [value for _, value in parser.decode_props(buf, link)]
        if len(values) >= 3 and values[0] == "OO" and isinstance(values[1], int) and isinstance(values[2], int):
            if values[1] in models and values[2] in models and values[1] != values[2]:
                children.setdefault(values[2], []).append(values[1])
                parents.setdefault(values[1], values[2])

    def branch(uid: int) -> dict:
        counts = {"bones": 0, "meshes": 0, "markers": 0, "cameras": 0, "other": 0}
        mesh_names: list[str] = []
        stack, seen = [uid], set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            model = models[current]
            kind = model["kind"]
            if kind == "LimbNode":
                counts["bones"] += 1
            elif kind == "Mesh":
                counts["meshes"] += 1
                if len(mesh_names) < 12:
                    mesh_names.append(model["name"].rsplit(":", 1)[-1])
            elif kind == "Marker":
                counts["markers"] += 1
            elif kind == "Camera":
                counts["cameras"] += 1
            else:
                counts["other"] += 1
            stack.extend(children.get(current, []))
        return {"counts": counts, "mesh_names": mesh_names, "nodes": len(seen)}

    subjects, objects, unlisted = [], [], []
    for uid, model in sorted(models.items(), key=lambda item: item[1]["name"]):
        if uid in parents:
            continue
        name = model["name"]
        segments = name.split(":")
        leaf = segments[-1]
        branch_info = branch(uid)
        counts = branch_info["counts"]
        entry = {"name": leaf, "path": name, "kind": model["kind"], **counts,
                 "nodes": branch_info["nodes"], "mesh_names": branch_info["mesh_names"]}
        if leaf == "Unlabeled_Markers" or "Unlabeled_Markers" in segments or counts["cameras"] > 0:
            unlisted.append(entry)
        elif "Labeling" not in segments[1:] and counts["bones"] >= 4:
            subjects.append(entry)
        elif counts["bones"] > 0 or counts["meshes"] > 0 or "Labeling" in segments[1:]:
            objects.append(entry)
        elif branch_info["nodes"] > 1:
            unlisted.append(entry)

    # Animation spans: Properties70 "P" entries carry the property name as the
    # first string; KTime values arrive as int64 (L/K) payloads.
    time_keys = ("LocalStart", "LocalStop", "ReferenceStart", "ReferenceStop")

    def property_times(node, wanted, depth=0) -> None:
        if depth > 4:
            return
        for child in node.children:
            if child.name == b"P":
                values = _p_props(child, buf)
                if values and isinstance(values[0][1], str) and values[0][1] in wanted:
                    for _, value in values[1:]:
                        if isinstance(value, bool):
                            continue
                        if isinstance(value, int) and 0 <= value < _TIME_UNITS * 100_000:
                            wanted[values[0][1]].append(value)
                        elif isinstance(value, float) and 0 < value < 1_000_000:
                            wanted[values[0][1]].append(value)
            property_times(child, wanted, depth + 1)

    animation = []
    for node in top["Objects"].children:
        if node.name != b"AnimationStack":
            continue
        header = parser.decode_props(buf, node)
        raw_name = next((value for _, value in header if isinstance(value, str)), "")
        wanted = {key: [] for key in time_keys}
        property_times(node, wanted)
        span = {"name": raw_name.split("\x00", 1)[0] or "(unnamed)"}
        start = min(wanted["LocalStart"], default=None)
        stop = max(wanted["LocalStop"], default=None)
        if start is None or stop is None or stop <= start:
            start, stop = min(wanted["ReferenceStart"], default=None), max(wanted["ReferenceStop"], default=None)
        if start is not None and stop is not None and stop > start:
            span["start_seconds"] = round(start / _TIME_UNITS, 4)
            span["duration_seconds"] = round((stop - start) / _TIME_UNITS, 4)
        animation.append(span)

    frame_rate, time_mode = None, None
    if "GlobalSettings" in top:
        wanted = {"TimeMode": [], "CustomFrameRate": []}
        property_times(top["GlobalSettings"], wanted, depth=1)
        custom = [value for value in wanted["CustomFrameRate"] if isinstance(value, float) and value > 0]
        if wanted["TimeMode"]:
            time_mode = wanted["TimeMode"][-1]
        frame_rate = round(custom[-1], 6) if custom else _TIME_MODES.get(time_mode)
        if animation and "duration_seconds" not in animation[0]:
            times = {"TimeSpanStart": [], "TimeSpanStop": []}
            property_times(top["GlobalSettings"], times, depth=1)
            start, stop = min(times["TimeSpanStart"], default=None), max(times["TimeSpanStop"], default=None)
            if start is not None and stop is not None and stop > start:
                animation[0]["start_seconds"] = round(start / _TIME_UNITS, 4)
                animation[0]["duration_seconds"] = round((stop - start) / _TIME_UNITS, 4)
        if frame_rate:
            for span in animation:
                if "duration_seconds" in span:
                    span["approx_frames"] = round(span["duration_seconds"] * frame_rate)
    model_kinds: dict[str, int] = {}
    for model in models.values():
        model_kinds[model["kind"]] = model_kinds.get(model["kind"], 0) + 1
    unlabeled_markers = sum(row["markers"] for row in unlisted
                            if row["name"] == "Unlabeled_Markers" or "Unlabeled_Markers" in row["path"].split(":"))
    return {"subjects": subjects, "objects": objects, "unlisted_branches": unlisted,
            "animation": animation, "frame_rate_hz": frame_rate, "frame_rate_mode": time_mode,
            "stats": {"models": len(models), "model_kinds": model_kinds, "object_counts": object_counts,
                      "unlabeled_markers": unlabeled_markers,
                      "labeled_markers": model_kinds.get("Marker", 0) - unlabeled_markers}}


def _meta_value(view, token: bytes) -> str | None:
    at = view.find(token)
    if at < 0:
        return None
    try:
        size = struct.unpack_from("<I", view, at + len(token))[0]
        if 1 <= size <= 64:
            return view[at + len(token) + 4:at + len(token) + 4 + size].decode("ascii")
    except (struct.error, UnicodeDecodeError):
        return None
    return None


def scan_mcp_capture_window(path: Path, frame_count: int | None = None, fps: float | None = None) -> dict | None:
    """Absolute wall-clock capture window of the take.

    MCP metadata: `Date` = ms epoch when Shogun Live started (its clock is what
    the octopus timecode generator shares with the cameras), `LiveGlobalStartTick`
    = 135MHz generator tick at that moment. The frame stream carries per-record
    generator ticks; the take occupies the frame window selected in Live
    (`StartOffset` frames dropped from the session start). Returned ISO times
    are local, so they can be matched against camera recordings directly.
    """
    with open(path, "rb") as handle:
        view = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            raw_date = _meta_value(view, b"Date")
            raw_tick = _meta_value(view, b"LiveGlobalStartTick")
            if not raw_date or not raw_tick:
                return None
            try:
                date_ms = int(raw_date)
                base_tick = int(raw_tick)
            except ValueError:
                return None
            offset_frames = 0
            raw_offset = _meta_value(view, b"StartOffset")
            if raw_offset and raw_offset.strip().isdigit():
                offset_frames = int(raw_offset)
            try:
                import numpy as np
            except ImportError:
                return None
            total = view.size()
            lo = np.uint64(base_tick)
            hi = np.uint64(base_tick + 13 * 3600 * 135_000_000)
            collected = []
            chunk = 1 << 20
            for align in range(8):
                off = align
                while off + 8 <= total:
                    n = min(chunk, (total - off) // 8)
                    block = np.frombuffer(view, dtype=np.uint64, count=n, offset=off)
                    sel = block[(block >= lo) & (block <= hi)]
                    if sel.size:
                        collected.append(sel.astype(np.int64))
                    off += n * 8
        finally:
            try:
                view.close()
            except BufferError:
                pass  # numpy exports may still reference the map; GC releases it
    if not collected:
        return None
    ticks = np.unique(np.concatenate(collected))
    if not ticks.size:
        return None
    start_seconds = (int(ticks[0]) - base_tick) / 135e6
    # Tick storage is sparse (per packet, not per frame): derive the take end
    # from the exact start plus the frame budget at the capture clock rate.
    with open(path, "rb") as handle:
        view = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            clock = scan_mcp_clock(view) or {}
            raw_offset = _meta_value(view, b"StartOffset")
        finally:
            try:
                view.close()
            except BufferError:
                pass
    fps = clock.get("frequency_hz") or 120.0
    frames = int(frame_count) if frame_count else max(int(ticks.size), 0)
    offset_frames = int(raw_offset) if (raw_offset and raw_offset.strip().isdigit()) else 0
    start_seconds += offset_frames / fps
    end_seconds = start_seconds + frames / fps
    start_wall = datetime.fromtimestamp(date_ms / 1000 + start_seconds)
    end_wall = datetime.fromtimestamp(date_ms / 1000 + end_seconds)
    return {"live_opened_utc": datetime.fromtimestamp(date_ms / 1000, timezone.utc).isoformat(),
            "capture_start": start_wall.isoformat(),
            "capture_end": end_wall.isoformat(),
            "seconds_from_live_open": [round(start_seconds, 3), round(end_seconds, 3)],
            "timecode_mhz": 135.0, "fps": fps,
            "tick_window": [int(ticks[0]), int(ticks[-1])],
            "start_offset_frames": offset_frames}


def scan_mcp_clock(view) -> dict | None:
    """Capture clock: FlowParameters `Input.ClockFrequency` (Hz) and
    `Input.ClockUseTimestamp`. Values are stored as length-prefixed ASCII
    strings right after the parameter name."""
    result: dict = {}
    token = b"Input.ClockFrequency"
    at = view.find(token)
    if at >= 0:
        try:
            size = struct.unpack_from("<I", view, at + len(token))[0]
            if 1 <= size <= 16:
                result["frequency_hz"] = float(view[at + len(token) + 4:at + len(token) + 4 + size])
        except (struct.error, ValueError):
            pass
    token = b"Input.ClockUseTimestamp"
    at = view.find(token)
    if at >= 0:
        try:
            size = struct.unpack_from("<I", view, at + len(token))[0]
            if 1 <= size <= 16:
                result["timestamped"] = view[at + len(token) + 4:at + len(token) + 4 + size].decode("ascii").strip().casefold() == "true"
        except (struct.error, ValueError, UnicodeDecodeError):
            pass
    return result or None


def scan_mcp_roster(path: Path, cap: int = 40) -> dict:
    """Subject/rigid records: [u32 len][name][8-byte stamp][u32 len]['Hips'|'root'].

    'Hips' roots a human skeleton; 'root' is a labeled rigid body's root bone.
    Both are per-capture definitions, not per-frame observations. The result
    also carries the capture clock (frequency + timestamp mode) when present.
    """
    found: dict[str, dict[str, int]] = {"subject": {}, "rigid": {}}
    with open(path, "rb") as handle:
        view = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for bone, group in ((b"Hips", "subject"), (b"root", "rigid")):
                for match in re.finditer(re.escape(struct.pack("<I", len(bone)) + bone), view):
                    for name_len in range(3, 33):
                        start = match.start() - 8 - name_len
                        if start < 4:
                            break
                        if struct.unpack_from("<I", view, start - 4)[0] != name_len:
                            continue
                        try:
                            name = view[start:start + name_len].decode("ascii")
                        except UnicodeDecodeError:
                            continue
                        if _NAME_RE.fullmatch(name):
                            found[group][name] = found[group].get(name, 0) + 1
            clock = scan_mcp_clock(view)
        finally:
            view.close()
    out = {group: [{"name": name, "occurrences": count} for name, count in sorted(rows.items())][:cap]
           for group, rows in found.items()}
    out["clock"] = clock
    return out


def scan_mcp_subjects(path: Path, cap: int = 40) -> list[dict]:
    return scan_mcp_roster(path, cap)["subject"]


def parse_step(path) -> dict:
    """Stage-0 read-only roster parse used inside a processing run.

    Never mutates anything; the pipeline wraps this so degraded scans land in
    the stage result instead of failing the run.
    """
    path = Path(path)
    stat = path.stat()
    kind = path.suffix[1:].lower()
    family = dataset_family(path)
    out = {"kind": kind, "size_bytes": stat.st_size, "dataset_family": family,
           "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
           "x2d_exists": path.with_suffix(".x2d").is_file() or path.with_suffix(".X2D").is_file()}
    if family in _HUMAN_ONLY_FAMILIES:
        out["objects_policy"] = "human_only_family: 处理时会在 configure 阶段删除场景内全部刚体物体,只保留人体"
    if kind == "mcp":
        roster = scan_mcp_roster(path)
        out["subjects"] = [dict(row) for row in roster["subject"]]
        out["rigid_bodies"] = [dict(row) for row in roster["rigid"]]
        if roster.get("clock"):
            out["capture_clock"] = dict(roster["clock"])
        out["note"] = ("MCP 名单为捕获级定义扫描(人物=骨架记录,物体=已标记刚体根); "
                       "逐帧有无与质量由后续原生阶段报告给出。")
    elif kind == "fbx":
        parsed = parse_fbx_roster(path)
        out["subjects"] = [{"name": row["name"], "bones": row["bones"], "markers": row["markers"],
                            "meshes": row["meshes"]} for row in parsed["subjects"]]
        out["objects"] = [{"name": row["name"], "bones": row["bones"], "markers": row["markers"],
                           "meshes": row["meshes"]} for row in parsed["objects"]]
        out["unlisted_branches"] = [{"name": row["name"], "kind": row["kind"], "nodes": row["nodes"],
                                     "markers": row["markers"]} for row in parsed["unlisted_branches"][:12]]
        out["stats"] = parsed["stats"]
        out["animation"] = parsed["animation"]
        out["frame_rate_hz"] = parsed["frame_rate_hz"]
    else:
        out["note"] = ("VDF 不内嵌名单" + (";存在同名 X2D(离线重建源)。" if out["x2d_exists"] else "。"))
    return out


def compact(value, *, depth: int = 3, list_cap: int = 12, dict_cap: int = 24, string_cap: int = 200):
    """Bounded projection: reports keep their shape without megabyte payloads."""
    if isinstance(value, str):
        return value if len(value) <= string_cap else value[:string_cap] + "…"
    if depth <= 0:
        if isinstance(value, (list, tuple)):
            return f"list[{len(value)}]"
        if isinstance(value, dict):
            return f"dict[{len(value)}]"
        if isinstance(value, (int, float, bool, type(None))):
            return value
        return str(value)
    if isinstance(value, dict):
        items = list(value.items())
        projected = {str(key): compact(item, depth=depth - 1, list_cap=list_cap, dict_cap=dict_cap, string_cap=string_cap)
                     for key, item in items[:dict_cap]}
        if len(items) > dict_cap:
            projected["…keys_omitted"] = len(items) - dict_cap
        return projected
    if isinstance(value, (list, tuple)):
        projected = [compact(item, depth=depth - 1, list_cap=list_cap, dict_cap=dict_cap, string_cap=string_cap)
                     for item in value[:list_cap]]
        if len(value) > list_cap:
            projected.append(f"…{len(value) - list_cap} more")
        return projected
    if isinstance(value, (int, float, bool, type(None))):
        return value
    return str(value)


def summarize_report(report) -> dict | None:
    if not isinstance(report, dict):
        return None
    summary = {}
    for key in ("status", "pipeline_status_pending", "partial"):
        if key in report:
            summary[key] = report[key]
    for key in ("review_reasons", "quality", "source_quality"):
        if key in report:
            summary[key] = compact(report[key], depth=2)
    for key, out in (("subjects", "subjects"), ("props", "props")):
        rows = report.get(key)
        if isinstance(rows, list):
            summary[out] = [compact(row, depth=3) for row in rows[:40] if isinstance(row, dict)]
    return summary or None
