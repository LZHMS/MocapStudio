"""Authoritative logical rosters and mesh-stage completeness regressions."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from theme_roster import audit_take_mesh_completeness
from step6_attach_prop_meshes import _arguments, _audit_roster, run


INFO15 = {'blue_phone', 'white_huatong', 'white_support', 'white_box',
          'white_remote_controller', 'white_clicker'}
TEA15 = {'blue_tray', 'white_teaport', 'white_cup', 'white_bottled_water',
         'white_can', 'white_trash'}
WRITE14 = {'white_handbook_base', 'white_handbook_cover', 'white_pen_box',
           'white_sign_pen', 'white_phone', 'red_box', 'white_wavy_cup'}


class LogicalRosterTests(unittest.TestCase):
    def test_group15_absent_actor2_plan_is_not_erased_by_info_only_scene(self):
        audit = audit_take_mesh_completeness('G15-INFO_X_TEA_IN01', INFO15, INFO15)
        self.assertEqual(audit['status'], 'INCOMPLETE')
        self.assertTrue(audit['verified'])
        self.assertFalse(audit['all_complete'])
        self.assertTrue(audit['missing_by_actor']['actor1']['is_complete_6'])
        actor2 = audit['missing_by_actor']['actor2']
        self.assertEqual(actor2['matched_count'], 0)
        self.assertEqual(actor2['expected_logical_count'], 6)
        self.assertEqual(set(actor2['missing_scene_props']), TEA15)
        self.assertEqual(set(audit['all_missing']), TEA15)

    def test_empty_scene_is_zero_of_six_not_zero_of_zero(self):
        audit = audit_take_mesh_completeness('G15-INFO_X_TEA_SO01', set(), set())
        self.assertFalse(audit['all_complete'])
        self.assertEqual(len(audit['all_missing']), 12)
        for actor in audit['missing_by_actor'].values():
            self.assertEqual(actor['expected_logical_count'], 6)
            self.assertEqual(actor['matched_count'], 0)
            self.assertFalse(actor['is_complete_6'])

    def test_five_logical_objects_are_incomplete_even_if_all_scene_props_match(self):
        scene = (INFO15 | TEA15) - {'white_support'}
        audit = audit_take_mesh_completeness('G15-INFO_X_TEA_IN02', scene, scene)
        self.assertFalse(audit['all_complete'])
        self.assertEqual(audit['missing_by_actor']['actor1']['matched_count'], 5)
        self.assertIn('white_support', audit['all_missing'])

    def test_handbook_two_native_parts_count_as_one_and_cover_is_required(self):
        scene = WRITE14 | INFO15
        complete = audit_take_mesh_completeness('G14-WRITE_X_INFO_IN04', scene, scene)
        self.assertTrue(complete['all_complete'])
        actor1 = complete['missing_by_actor']['actor1']
        self.assertEqual(actor1['scene_props_count'], 6)
        self.assertEqual(actor1['matched_count'], 6)
        missing_cover = audit_take_mesh_completeness(
            'G14-WRITE_X_INFO_IN04', scene, scene - {'white_handbook_cover'})
        self.assertFalse(missing_cover['all_complete'])
        self.assertEqual(missing_cover['missing_by_actor']['actor1']['matched_count'], 5)
        self.assertEqual(missing_cover['all_missing'], ['white_handbook_cover'])
        cover_only = audit_take_mesh_completeness(
            'G14-WRITE_X_INFO_IN04', scene, scene - {'white_handbook_base'})
        self.assertIn('white_handbook_base', cover_only['all_missing'])

    def test_laptop_screen_does_not_substitute_for_base(self):
        info1 = {'white_laptop_base', 'white_laptop_screen', 'white_mouse', 'blue_udrive',
                 'blue_phone', 'white_presentation_clicker', 'white_mug_cup'}
        tea1 = {'blue_tray', 'white_teaport', 'white_cup', 'white_bottled_water',
                'white_drink', 'white_trash'}
        scene = info1 | tea1
        complete = audit_take_mesh_completeness('G1-INFO_X_TEA_IN01', scene, scene)
        self.assertTrue(complete['all_complete'])
        for part in ('white_laptop_base', 'white_laptop_screen'):
            with self.subTest(part=part):
                audit = audit_take_mesh_completeness('G1-INFO_X_TEA_IN01', scene, scene - {part})
                self.assertFalse(audit['all_complete'])
                self.assertIn(part, audit['all_missing'])
                self.assertEqual(audit['missing_by_actor']['actor1']['matched_count'], 5)

    def test_mapping_aliases_resolve_without_inflating_logical_count(self):
        scene = INFO15 | TEA15 | {'white_presentation_clicker', 'clicker'}
        audit = audit_take_mesh_completeness('G15-INFO_X_TEA_IN03', scene, scene)
        self.assertTrue(audit['all_complete'])
        self.assertEqual(audit['missing_by_actor']['actor1']['matched_count'], 6)

    def test_unknown_group_or_roster_is_unverified_never_complete(self):
        for take in ('INFO_X_TEA_IN01', 'G999-INFO_X_TEA_IN01',
                     'G15-WRITE_X_WRITE_IN01', 'G15-UNKNOWN_IN01',
                     'group14/G15-INFO_X_TEA_IN01', 'G15wrong-INFO_X_TEA_IN01',
                     'G15-INFO_X_TEACUP_IN01'):
            with self.subTest(take=take):
                audit = audit_take_mesh_completeness(take, INFO15 | TEA15, INFO15 | TEA15)
                self.assertEqual(audit['status'], 'UNVERIFIED')
                self.assertFalse(audit['verified'])
                self.assertFalse(audit['all_complete'])
                self.assertTrue(audit['errors'])

    def test_same_theme_actors_have_independent_rosters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            group = root / 'group7'
            group.mkdir()
            names1 = '话筒, 话筒支架, 白色收纳盒, 设备遥控器, 翻页笔, 蓝色手机'
            names2 = '笔记本电脑, 鼠标, U盘, 蓝色手机, 翻页笔, 马克杯'
            for index, names in enumerate((names1, names2), 1):
                (group / f'actor{index}.txt').write_text(
                    f'◆ 第1批物体演练：组 07 · 演示 1（信息-信息）'
                    f'（全局编号: 演员00{index} | 本人道具: {names}）\n', encoding='utf-8')
            audit = audit_take_mesh_completeness(
                'G7-INFO_X_INFO_IN01', INFO15, INFO15, roster_root=root)
            self.assertTrue(audit['verified'])
            self.assertEqual(set(audit['missing_by_actor']), {'actor1', 'actor2'})
            self.assertEqual(set(audit['missing_by_theme']), {'actor1:INFO', 'actor2:INFO'})
            self.assertTrue(audit['missing_by_actor']['actor1']['is_complete_6'])
            self.assertFalse(audit['missing_by_actor']['actor2']['is_complete_6'])
            self.assertIn('white_laptop_screen', audit['all_missing'])

    def test_five_item_or_unknown_roster_cannot_be_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            group = root / 'group7'
            group.mkdir()
            for names in ('话筒, 话筒支架, 白色收纳盒, 设备遥控器, 翻页笔',
                          '话筒, 话筒支架, 白色收纳盒, 设备遥控器, 翻页笔, 未知道具',
                          '话筒, 话筒支架, 白色收纳盒, 设备遥控器, 翻页笔, 翻页笔'):
                for index in (1, 2):
                    (group / f'actor{index}.txt').write_text(
                        '◆ 第1批物体演练：组 07 · 演示 1（信息-信息）'
                        f'（全局编号: 演员001 | 本人道具: {names}）\n', encoding='utf-8')
                audit = audit_take_mesh_completeness(
                    'G7-INFO_X_INFO_IN01', INFO15, INFO15, roster_root=root)
                self.assertEqual(audit['status'], 'UNVERIFIED')
                self.assertFalse(audit['all_complete'])

    def test_prepared_stage_reaudits_instead_of_reusing_matched_stage(self):
        scene = INFO15 | TEA15
        plan = {'matched_props': [{'prop': prop} for prop in scene], 'skipped_props': []}
        self.assertTrue(_audit_roster('G15-INFO_X_TEA_IN01', plan, scene)['all_complete'])
        prepared = _audit_roster('G15-INFO_X_TEA_IN01', plan, scene - {'white_support'})
        self.assertFalse(prepared['all_complete'])
        self.assertEqual(prepared['missing_by_actor']['actor1']['missing_scene_props'], [])
        self.assertEqual(prepared['all_missing'], ['white_support'])


class MeshRosterPersistenceTests(unittest.TestCase):
    def run_attachment(self, *, fail_assets=False, fail_publish=False, info_only=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / 'input' / 'group15' / 'INFO_X_TEA_IN01.fbx'
        source.parent.mkdir(parents=True)
        original = b'original animation and markers'
        source.write_bytes(original)
        assets = root / 'assets'
        assets.mkdir()
        mapping = root / 'mapping.json'
        mapping.write_text('{}', encoding='utf-8')
        output = root / 'mesh'
        args = _arguments(['--input', str(source), '--output-dir', str(output),
                           '--assets-root', str(assets), '--mapping', str(mapping),
                           '--no-strip-unlabeled'])
        scene = INFO15 if info_only else INFO15 | TEA15
        records = [{'prop': prop, 'asset_id': prop} for prop in sorted(scene)]
        plan = {'matched_props': records, 'skipped_props': [], 'native_unit_mm': 1.,
                'detected_prop_count': len(scene),
                'source_sha256': hashlib.sha256(original).hexdigest()}
        attached = [{'prop': prop, 'mesh_parts': 1} for prop in sorted(scene)]
        if fail_assets:
            attached = [record for record in attached if record['prop'] != 'white_support']
        rejected = ([{'prop': 'white_support', 'reason': 'ASSET_PREPARATION_FAILED'}]
                    if fail_assets else [])

        def publish(src, destination, meshes, *, verification_path, overwrite):
            if fail_publish:
                raise ValueError('verification rejected publication')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b'verified output fixture')
            return {'passed': True, 'checks': {}}

        with patch('mesh_alignment.load_catalog', return_value={prop: {'id': prop} for prop in scene}), \
                patch('mesh_alignment.load_mapping', return_value={}), \
                patch('mesh_alignment.analyze_fbx', return_value=plan) as analyze, \
                patch('step6_attach_prop_meshes._find_blender', return_value='unused'), \
                patch('step6_attach_prop_meshes._prepare_assets', return_value={}), \
                patch('step6_attach_prop_meshes._build_meshes',
                      return_value=([{'prop': record['prop']} for record in attached], attached, rejected)), \
                patch('mesh_fbx.graft_file', side_effect=publish), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = run(args)
        self.assertEqual(analyze.call_count, 1)
        self.assertEqual(analyze.call_args.kwargs['max_rms_mm'], args.max_rms_mm)
        self.assertEqual(source.read_bytes(), original)
        batch = json.loads((output / 'mesh_batch_report.json').read_text(encoding='utf-8'))
        entry = batch['files'][0]
        report = json.loads(Path(entry['report']).read_text(encoding='utf-8'))
        self.assertEqual(entry['roster_audit'], report['roster_audit'])
        return code, entry

    def test_asset_failure_keeps_matched_complete_but_attached_incomplete(self):
        code, entry = self.run_attachment(fail_assets=True)
        self.assertEqual(code, 2)
        self.assertEqual(entry['status'], 'PARTIAL')
        self.assertTrue(entry['roster_audits']['matched']['all_complete'])
        self.assertFalse(entry['roster_audits']['prepared']['all_complete'])
        self.assertFalse(entry['roster_audit']['all_complete'])
        self.assertEqual(entry['roster_audit']['all_missing'], ['white_support'])

    def test_info_only_publication_persists_missing_tea_without_repair(self):
        code, entry = self.run_attachment(info_only=True)
        self.assertEqual(code, 2)
        self.assertEqual(entry['status'], 'PARTIAL')
        self.assertEqual(set(entry['roster_audit']['all_missing']), TEA15)
        self.assertEqual({record['prop'] for record in entry['attached_props']}, INFO15)

    def test_failed_publication_never_claims_attached_roster_complete(self):
        code, entry = self.run_attachment(fail_publish=True)
        self.assertEqual(code, 1)
        self.assertEqual(entry['status'], 'FAILED')
        self.assertTrue(entry['roster_audits']['prepared']['all_complete'])
        self.assertFalse(entry['roster_audit']['all_complete'])
        self.assertEqual(entry['attached_props'], [])

    def test_complete_verified_roster_can_succeed(self):
        code, entry = self.run_attachment()
        self.assertEqual(code, 0)
        self.assertEqual(entry['status'], 'SUCCESS')
        self.assertTrue(entry['roster_audit']['all_complete'])


if __name__ == '__main__':
    unittest.main()
