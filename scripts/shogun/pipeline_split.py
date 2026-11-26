"""Byte-exact staged FBX subject splits; no Shogun or FBX SDK imports."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import struct


# Load the repository helper by path, not a machine-specific PYTHONPATH or the
# historical export_one grouping configuration.
_SPEC = importlib.util.spec_from_file_location(
    "_shogun_extract_groups", Path(__file__).resolve().parents[1] / "extract_groups.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("cannot load scripts/extract_groups.py")
fbx = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fbx)

_FORBIDDEN = {"System", "Unlabeled_Markers"}


def _subject_folder(subject: str) -> str:
    # Percent-encode UTF-8, including dots, separators and Windows punctuation.
    # Always suffix the original spelling's digest: A and a must remain distinct
    # on Windows even when exported separately or in a different roster order.
    pieces = []
    length = 0
    for byte in subject.encode("utf-8"):
        token = chr(byte) if (65 <= byte <= 90 or 97 <= byte <= 122
                              or 48 <= byte <= 57 or byte in (45, 95)) else f"%{byte:02X}"
        if length + len(token) > 48:
            break
        pieces.append(token)
        length += len(token)
    return "".join(pieces) + "--" + hashlib.sha256(subject.encode("utf-8")).hexdigest()[:16]


def _payload_equal(source_buf, source_node, output_buf, output_node):
    """Compare every nested raw property, ignoring only rebased end offsets."""
    if source_buf[source_node.nstart + 8:source_node.cstart] != \
            output_buf[output_node.nstart + 8:output_node.cstart]:
        return False
    return (len(source_node.children) == len(output_node.children)
            and all(_payload_equal(source_buf, left, output_buf, right)
                    for left, right in zip(source_node.children, output_node.children)))


def _validate_graph(graph):
    if len(graph.uid2node) != len(graph.obj_children):
        raise ValueError("FBX contains duplicate object IDs")
    root_ids = [uid for uid in graph.child2.get(0, []) if uid in graph.model_names]
    if len(root_ids) != len(graph.roots):
        raise ValueError("FBX contains ambiguous duplicate root names or connections")


def _validate_names(names, label):
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{label} must contain nonempty exact FBX root names")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate {label} names")
    if set(names) & _FORBIDDEN:
        raise ValueError("System and Unlabeled_Markers cannot be split members")


def _check_destination(path: Path, root: Path, source: Path):
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("split destination must remain below the staging root")
    if str(resolved).casefold() == str(source).casefold():
        raise ValueError("split destination would overwrite the source")
    # Also enforce Windows case-insensitive non-overwrite on other platforms.
    current = root
    for part in path.relative_to(root).parts:
        if current.is_dir():
            matches = [child for child in current.iterdir()
                       if child.name.casefold() == part.casefold()]
            if matches and matches[0].name != part:
                raise FileExistsError(f"case-colliding split destination: {matches[0]}")
        current /= part
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"split destination already exists: {path}")


def split_subject_fbx(source: Path, humans: list[str], props: list[str],
                      output_dir: Path, include_props: bool) -> list[dict]:
    """Write one exact human branch plus, for HOI, all requested prop branches.

    Paths in the returned report entries are relative to the source's staging
    directory. Any missing/ambiguous branch or failed integrity check is fatal;
    no guessed ownership, name matching or placeholder export is performed.
    """
    source = Path(source).resolve(strict=True)
    root = source.parent
    output_dir = Path(output_dir).absolute()
    resolved_output = output_dir.resolve()
    if resolved_output == root or not resolved_output.is_relative_to(root):
        raise ValueError("output_dir must lie below source.parent")
    # Resolve once so .. segments cannot affect containment or report paths.
    output_dir = resolved_output
    _validate_names(humans, "human")
    _validate_names(props, "prop")
    if not humans:
        raise ValueError("at least one human is required")
    if set(humans) & set(props):
        raise ValueError("human and prop rosters overlap")
    with source.open("rb") as stream:
        header = stream.read(27)
    if len(header) != 27 or header[:23] != fbx.MAGIC:
        raise ValueError(f"not a binary FBX: {source}")
    if struct.unpack_from("<I", header, 23)[0] < 7500:
        raise ValueError("subject splitting requires binary FBX version >= 7500")
    buf, version, tops = fbx.parse_file(source)
    graph = fbx.build_graph(buf, tops)
    _validate_graph(graph)
    missing = (set(humans) | set(props)) - set(graph.roots)
    if missing:
        raise ValueError(f"requested FBX roots not found: {sorted(missing)!r}")

    plans = []
    folders = set()
    for subject in humans:
        folder = _subject_folder(subject)
        if folder.casefold() in folders:
            raise ValueError("encoded subject folders collide")
        folders.add(folder.casefold())
        destination = output_dir / folder / "sequence.fbx"
        _check_destination(destination, root, source)
        members = [subject] + (list(props) if include_props else [])
        keep = fbx.keep_set_for_group(graph, members)
        # Closure must never pull another root/model branch through OP links.
        excluded_models = set()
        for name, uid in graph.roots.items():
            if name not in members:
                excluded_models.update(fbx.branch_uids(graph, uid) & set(graph.model_names))
        if keep & excluded_models:
            raise ValueError(f"split for {subject!r} includes excluded model branches")
        plans.append((subject, destination, members, keep))

    entries = []
    created = []
    try:
        for subject, destination, members, keep in plans:
            data, object_count, connection_count = fbx.splice(buf, tops, keep)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _check_destination(destination, root, source)
            with destination.open("xb") as stream:
                created.append(destination)
                stream.write(data)
            # verify returns False on failure, including with verbose=False.
            if not fbx.verify(source, destination, verbose=False):
                raise ValueError(f"FBX subset verification failed for {subject!r}")
            out_buf, out_version, out_tops = fbx.parse_file(destination)
            out_graph = fbx.build_graph(out_buf, out_tops)
            _validate_graph(out_graph)
            if out_version != version or set(out_graph.roots) != set(members):
                raise ValueError(f"FBX root membership/version mismatch for {subject!r}")
            if set(out_graph.uid2node) != keep:
                raise ValueError(f"FBX kept-object membership mismatch for {subject!r}")
            # Existing verify checks object headers and samples curves. Check ALL
            # nested payloads too: geometry arrays, curve keys, settings and time.
            if any(not _payload_equal(buf, graph.uid2node[uid], out_buf, node)
                   for uid, node in out_graph.uid2node.items()):
                raise ValueError(f"FBX nested object payload mismatch for {subject!r}")
            source_globals = [node for node in tops
                              if node.name not in (b"Objects", b"Connections")]
            output_globals = [node for node in out_tops
                              if node.name not in (b"Objects", b"Connections")]
            if len(source_globals) != len(output_globals) or any(
                    not _payload_equal(buf, left, out_buf, right)
                    for left, right in zip(source_globals, output_globals)):
                raise ValueError(f"FBX global/time settings mismatch for {subject!r}")
            entries.append({
                "subject": subject,
                "relative_path": destination.relative_to(root).as_posix(),
                "members": members,
                "verification": {"passed": True, "exact_roots": True,
                                 "byte_exact_payloads": True,
                                 "global_settings_preserved": True,
                                 "dangling_links": 0, "fbx_version": version,
                                 "objects": object_count, "connections": connection_count},
            })
    except BaseException:
        # Remove only files created by this call, never preexisting stage data.
        for destination in created:
            destination.unlink(missing_ok=True)
        raise
    return entries
