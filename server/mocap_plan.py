"""Read-only catalog of the p2_hoi mocap plan (导演台本 + 拍摄备注).

Parses <plan-dir>/groupNN/director_sheet.md rehearsals and <plan-dir>/notes.json
shooting notes into lookups keyed by dataset group ("group9") and the sequence
code used by the dataset directories ("INFO1_X_TEA1_SM02"). Sheets sometimes
carry the wrong 组 number in their headers, so the group is always taken from
the directory name. Missing or malformed files degrade to missing plan info;
the dashboard must keep working without it.
"""
from __future__ import annotations

from pathlib import Path
import re

# 中文场次词 -> 数据集序列代码段（导演台本括号内的顺序即代码里的顺序）。
CN_TO_CODE = {"信息": "INFO", "茶水": "TEA", "书写": "WRITE"}

_HEADER = re.compile(r"^##\s*组\s*\d+\s*·\s*演练\s*\d+\s*：\s*(.+?)\s*$")
_PAIR = re.compile("（([^（）]+)）\\s*$")
_ACTOR = re.compile(r"^\-\s*\*\*演员(\d)（([^）]*)）道具\*\*[:：]\s*(.*)$")
# 数据目录后缀：SM/SO/DU/IN/OUT/CL + 编号（MP 的 "MP-S-WALK" 不匹配，整段保留）。
_TAIL = re.compile(r"_[A-Z]{2}\d{1,3}$")
_PROP = re.compile(r"^\s*(?P<zh>[^()（）]+?)\s*[（(](?P<en>[A-Za-z0-9_/.\- ]+)[)）]\s*$")


def _pair_to_code(pair):
    """（信息1-茶水1） -> INFO1_X_TEA1；未知词返回 None。"""
    parts = [p.strip() for p in pair.split("-")]
    pieces = []
    for part in parts:
        match = re.fullmatch(r"([\u4e00-\u9fff]+)(\d*)", part)
        if not match:
            return None
        base = CN_TO_CODE.get(match.group(1))
        if base is None:
            return None
        pieces.append(base + match.group(2))
    return "_X_".join(pieces)


def _split_props(raw):
    props = []
    for token in re.split(r"[、,，]", raw):
        token = token.strip()
        if not token:
            continue
        match = _PROP.match(token)
        if match:
            props.append({"zh": match.group("zh").strip(), "en": match.group("en").strip()})
        else:
            props.append({"zh": token, "en": None})
    return props


class PlanCatalog:
    def __init__(self, root=None, *, rehearsals=None, notes=None, errors=None):
        self.root = str(root) if root else None
        self.rehearsals = rehearsals if rehearsals is not None else {}
        self.notes = notes if notes is not None else {}
        self.errors = errors or []

    def __bool__(self):
        return bool(self.rehearsals or self.notes)

    def lookup(self, group, take_code):
        """Return display fields for one take; missing plan -> all-None/empty."""
        notes = list(self.notes.get((group, take_code), []))
        base = _TAIL.sub("", take_code) if take_code else ""
        rehearsal = self.rehearsals.get((group, base)) if base else None
        if rehearsal is None:
            return {"plan_title": None, "plan_actors": None, "plan_props": None, "plan_notes": notes or None}
        return {"plan_title": rehearsal["title"],
                "plan_actors": [{"slot": a["slot"], "label": a["label"], "props": a["props"]} for a in rehearsal["actors"]],
                "plan_props": sorted({p["en"] or p["zh"] for a in rehearsal["actors"] for p in a["props"]}),
                "plan_notes": notes or None}

    def stats(self):
        return {"root": self.root, "rehearsals": len(self.rehearsals),
                "note_entries": sum(len(v) for v in self.notes.values()), "errors": self.errors[:8]}

    def _parse_sheet(self, group, sheet):
        current = None
        for line in sheet.read_text(encoding="utf-8", errors="replace").splitlines():
            header = _HEADER.match(line.strip())
            if header:
                tail = header.group(1)
                pair = _PAIR.search(tail)
                code = _pair_to_code(pair.group(1)) if pair else None
                title = re.sub(r"（[^（）]*）\s*$", "", tail).strip()
                current = (group, code) if code else None
                if current:
                    self.rehearsals.setdefault(current, {"title": title, "actors": []})
                continue
            actor = _ACTOR.match(line.strip())
            if actor and current:
                slot = f"演员{actor.group(1)}"
                self.rehearsals[current]["actors"].append({
                    "slot": slot, "label": (actor.group(2) or "").strip() or slot,
                    "props": _split_props(actor.group(3))})

    def _parse_notes(self, path):
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            self.errors.append("notes.json 顶层不是对象")
            return
        for group_key, sequences in data.items():
            group = f"group{group_key}"
            if not isinstance(sequences, dict):
                continue
            for code, actors in sequences.items():
                entries = []
                if isinstance(actors, dict):
                    for actor_key, steps in sorted(actors.items(), key=lambda item: str(item[0])):
                        if not isinstance(steps, dict):
                            continue
                        for step, text in sorted(steps.items(),
                                                 key=lambda item: int(item[0]) if str(item[0]).isdigit() else 0):
                            entries.append(f"演员{actor_key} · 第 {step} 步：{text}")
                if entries:
                    self.notes[(group, str(code))] = entries


def load(plan_dir):
    root = Path(plan_dir) if plan_dir else None
    catalog = PlanCatalog(root)
    if not root or not root.is_dir():
        catalog.errors.append(f"台本目录不可读：{root}")
        return catalog
    for group_dir in sorted(root.iterdir()):
        group = group_dir.name
        sheet = group_dir / "director_sheet.md"
        if group_dir.is_dir() and sheet.is_file():
            try:
                catalog._parse_sheet(group, sheet)
            except OSError as exc:
                catalog.errors.append(f"{group}: {exc}")
    notes_path = root / "notes.json"
    if notes_path.is_file():
        try:
            catalog._parse_notes(notes_path)
        except (OSError, ValueError) as exc:
            catalog.errors.append(f"notes.json: {exc}")
    return catalog
