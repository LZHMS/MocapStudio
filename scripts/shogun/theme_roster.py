"""Audit six logical props per actor against the recorded scenario plan.

Scene contents never determine the expected roster. Mesh evidence is supplied by
the caller; this module does not establish actor presence or tracking quality.
"""

from pathlib import Path
import re

from prop_mapping import CN_TO_ITEM, SHOGUN_TO_ITEM


DEFAULT_ROSTER_ROOT = Path(__file__).resolve().parents[2] / 'scenario_sim' / 'output'
_THEMES = {'书写': 'WRITE', '信息': 'INFO', '茶水': 'TEA'}
_COMPOSITE_PARTS = {
    'white_handbook_base': ('white_handbook_base', 'white_handbook_cover'),
    'white_laptop_base': ('white_laptop_base', 'white_laptop_screen'),
}
_SEPARATE_PARTS = {part for parts in _COMPOSITE_PARTS.values() for part in parts}
_HEADER = re.compile(
    r'^◆\s*第(\d+)批物体演练[：:]\s*组\s*(\d+)\s*·\s*演示\s*\d+'
    r'（([^（）]+)）.*?本人道具[：:]\s*([^）\n]+)）', re.MULTILINE)


def parse_themes_from_take_name(take_name: str) -> tuple[str | None, str | None]:
    """Read the ordered actor themes, including same-theme pairs."""
    match = re.search(r'(?<![A-Z])(WRITE|INFO|TEA)(?:_X_|_)(WRITE|INFO|TEA)(?![A-Z])',
                      str(take_name).upper())
    return match.groups() if match else (None, None)


def _native_name(name):
    name = str(name).strip().split(':', 1)[0].casefold()
    # The mapping's cover/screen aliases identify a logical object, not its base.
    if name in _SEPARATE_PARTS:
        return name
    item = SHOGUN_TO_ITEM.get(name)
    return item['shogun_name'] if item else name


def _actor_roster(path, group, themes, actor_index):
    candidates = []
    for header in _HEADER.finditer(path.read_text(encoding='utf-8-sig')):
        batch, header_group, pair, prop_text = header.groups()
        pair = tuple(_THEMES.get(name.strip(), name.strip().upper())
                     for name in re.split(r'[-－—]', pair))
        if int(header_group) == group and pair == themes:
            candidates.append((int(batch), [name.strip() for name in
                                           re.split(r'[,，]', prop_text)]))
    if len(candidates) != 1:
        raise ValueError(f'{path}: expected one roster for {themes}, found {len(candidates)}')
    batch, names = candidates[0]
    props = []
    for name in names:
        item = CN_TO_ITEM.get(name)
        if item is None or item['is_global']:
            raise ValueError(f'{path}: unknown interactive prop {name!r}')
        logical = item['shogun_name']
        props.append({'logical_prop': logical, 'cn_name': name,
                      'expected_parts': list(_COMPOSITE_PARTS.get(logical, (logical,)))})
    if len(props) != 6 or len({prop['logical_prop'] for prop in props}) != 6:
        raise ValueError(f'{path}: roster must contain exactly six distinct logical props')
    return {'actor': f'actor{actor_index + 1}', 'theme': themes[actor_index],
            'batch': batch, 'roster_path': str(path), 'props': props}


def audit_take_mesh_completeness(take_name: str, scene_props_with_markers: set[str],
                                 matched_props: set[str], *, roster_root=None) -> dict:
    """Return COMPLETE only for two verified six-logical-prop actor rosters.

    The original three positional arguments remain supported. ``take_name`` must
    include a G<number>/group<number> identity (a complete source path is valid).
    ``matched_props`` must describe the current evidence stage, not planned assets
    when auditing prepared or attached output. ``roster_root`` overrides the
    scenario output directory, primarily for isolated roster tests.
    """
    themes = parse_themes_from_take_name(take_name)
    groups = {int(value) for value in re.findall(
        r'(?<![A-Za-z0-9])(?:GROUP|G)[ _-]?(\d+)(?![A-Za-z0-9])',
        str(take_name).upper())}
    result = {'schema_version': 2, 'status': 'UNVERIFIED', 'verified': False,
              'group': next(iter(groups)) if len(groups) == 1 else None,
              'theme1': themes[0], 'theme2': themes[1],
              'required_logical_count_per_actor': 6, 'missing_by_actor': {},
              'missing_by_theme': {}, 'all_missing': [], 'missing_props': [],
              'all_complete': False, 'errors': []}
    if None in themes:
        result['errors'].append('UNKNOWN_THEMES')
    if len(groups) != 1:
        result['errors'].append('UNKNOWN_OR_AMBIGUOUS_GROUP')
    if result['errors']:
        return result
    root = Path(roster_root) if roster_root is not None else DEFAULT_ROSTER_ROOT
    group_dir = root / f"group{result['group']}"
    actors = []
    for index in range(2):
        path = group_dir / f'actor{index + 1}.txt'
        try:
            actors.append(_actor_roster(path, result['group'], themes, index))
        except (OSError, UnicodeError, ValueError) as exc:
            result['errors'].append(str(exc))
    if len(actors) == 2 and actors[0]['batch'] != actors[1]['batch']:
        result['errors'].append('Actor rosters disagree on batch')
    scene = {_native_name(name) for name in scene_props_with_markers}
    matched = {_native_name(name) for name in matched_props}
    all_missing = set()
    for actor in actors:
        for prop in actor['props']:
            expected = set(prop['expected_parts'])
            prop['scene_parts'] = sorted(expected & scene)
            prop['matched_parts'] = sorted(expected & matched)
            prop['missing_scene_parts'] = sorted(expected - scene)
            prop['missing_mesh_parts'] = sorted(expected - matched)
            prop['is_complete'] = expected <= scene and expected <= matched
        props = actor['props']
        missing_scene = sorted({part for prop in props for part in prop['missing_scene_parts']})
        missing_mesh = sorted({part for prop in props for part in prop['missing_mesh_parts']})
        missing = sorted(set(missing_scene) | set(missing_mesh))
        actor.update(expected_logical_count=len(props),
                     scene_props_count=sum(not prop['missing_scene_parts'] for prop in props),
                     matched_count=sum(prop['is_complete'] for prop in props),
                     missing_scene_props=missing_scene, missing_mesh_props=missing_mesh,
                     missing_logical_props=[prop['logical_prop'] for prop in props
                                            if not prop['is_complete']],
                     missing=missing,
                     is_complete_6=len(props) == 6 and all(prop['is_complete'] for prop in props))
        result['missing_by_actor'][actor['actor']] = actor
        key = actor['theme'] if themes[0] != themes[1] else f"{actor['actor']}:{actor['theme']}"
        result['missing_by_theme'][key] = actor
        all_missing.update(missing)
    result['all_missing'] = result['missing_props'] = sorted(all_missing)
    result['verified'] = not result['errors'] and len(actors) == 2
    result['all_complete'] = result['verified'] and all(actor['is_complete_6'] for actor in actors)
    result['status'] = ('COMPLETE' if result['all_complete'] else
                        'INCOMPLETE' if result['verified'] else 'UNVERIFIED')
    return result
