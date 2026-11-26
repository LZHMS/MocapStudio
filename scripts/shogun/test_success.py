"""Success-view eligibility and safe refresh regressions; native-free fixtures."""
import csv
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest

from step5_index_success import collect_successes, refresh, OWNER
from pipeline_layout import initialize_layout, take_paths
from step7_export_overview import scan_all_takes


class SuccessViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.lock = self.root / '.pipeline.lock'
        self.lock.write_bytes(b'active scheduler lock remains untouched')

    def report(self, name, status='SUCCESS', job_id=None):
        path = self.root / 'takes' / 'group1' / 'MP' / name / 'report.json'
        products = {}
        for role in ('vdf', 'fbx', 'vsk'):
            target = path.parent / 'artifacts' / f'scene.{role}'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'original solved capture data')
            products[role] = str(target)
        data = {'job_id': job_id or name, 'status': status, 'input_kind': 'mcp',
                'input_path': f'H:/raw/{name}.mcp', 'artifacts': products,
                'quality': {'basis': 'post_processing', 'usable': status == 'SUCCESS'},
                'subjects': [{'name': 'actor', 'processing_target': True,
                              'rig_class': 'HUMAN_RIG',
                              'quality': {'basis': 'post_processing', 'usable': True}}]}
        self.save(path, data)
        return path, data

    @staticmethod
    def save(path, data):
        path.write_text(json.dumps(data), encoding='utf-8')

    @staticmethod
    def fixture_shortcut_writer(entries):
        # Only the Windows Shell boundary is replaced; eligibility and cleanup run normally.
        for row in entries:
            Path(row['shortcut_path']).write_bytes(b'shortcut fixture')

    def enable_local(self):
        raw = self.root / "raw"
        raw.mkdir()
        self.root = raw / "_processing" / "control"
        initialize_layout(self.root, raw, "control")
        return raw

    def local_report(self, relative, status="SUCCESS"):
        old, data = self.report(Path(relative).name, status=status, job_id=relative)
        paths = take_paths(self.root, relative)
        paths["take_root"].parent.mkdir(parents=True, exist_ok=True)
        old.parent.rename(paths["take_root"])
        data["relative_path"] = relative
        data["artifacts"] = {role: str(paths["destination"] / Path(value).name)
                             for role, value in data["artifacts"].items()}
        self.save(paths["report_path"], data)
        return paths, data

    def test_local_success_shortcuts_target_each_actual_sequence_artifact_root(self):
        self.enable_local()
        first, _ = self.local_report("group1/HOI/sequence/first.mcp")
        second, _ = self.local_report("group1/HOI/sequence/second.mcp")
        result = refresh(self.root, self.fixture_shortcut_writer)
        self.assertEqual(result["success_count"], 2)
        entries = json.loads((self.root / "_success" / "_index.json").read_text())["entries"]
        self.assertEqual({row["artifacts_directory"] for row in entries},
                         {str(first["destination"]), str(second["destination"])})
        self.assertEqual({row["relative_path"] for row in entries},
                         {"group1/HOI/sequence/first.mcp", "group1/HOI/sequence/second.mcp"})
        self.assertTrue(all(Path(row["shortcut_path"]).is_file() for row in entries))

    def test_local_success_rejects_raw_neighbor_version_and_take_references(self):
        raw = self.enable_local()
        paths, data = self.local_report("group1/HOI/sequence/take.mcp")
        for target in (raw / "raw.fbx",
                       paths["take_root"].parent / "neighbor.mcp" / "artifacts" / "scene.fbx",
                       paths["take_root"].parent.parent / "other" / "take.mcp" / "artifacts" / "scene.fbx"):
            with self.subTest(target=target):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"not this take's product")
                data["artifacts"]["fbx"] = str(target)
                self.save(paths["report_path"], data)
                rows, issues = collect_successes(self.root)
                self.assertEqual(rows, [])
                self.assertEqual(len(issues), 1)

    def test_local_overview_promotes_repair_with_report_relative_artifacts(self):
        raw = self.enable_local()
        relative = "group1/HOI/sequence/take.mcp"
        original, data = self.local_report(relative, status="NEEDS_REVIEW")
        repair = self.root / "_repair_runs" / "fixed"
        initialize_layout(repair, raw, "fixed")
        paths = take_paths(repair, relative)
        paths["destination"].mkdir(parents=True)
        (paths["destination"] / "scene.fbx").write_bytes(b"repaired product")
        data.update(status="SUCCESS", artifacts={"fbx": "artifacts/scene.fbx"})
        self.save(paths["report_path"], data)
        rows, stats = scan_all_takes(self.root, raw / "mesh-not-present")
        row = rows[relative]
        self.assertEqual(row["report_path"], str(paths["report_path"]))
        self.assertEqual(row["fbx_path"], str(paths["destination"] / "scene.fbx"))
        self.assertEqual(row["artifacts_dir"], str(paths["destination"]))
        self.assertEqual(row["source_status"], "NEEDS_REVIEW")
        self.assertTrue(row["repaired"])
        self.assertEqual(stats["repaired_success"], 1)
        self.assertTrue(original["report_path"].is_file())

    def test_overview_discovers_only_mesh_controls_and_uses_exact_take_identity(self):
        raw = self.enable_local()
        first = "group1/HOI/sequence-one/same.mcp"
        second = "group1/HOI/sequence-two/same.mcp"
        self.local_report(first)
        self.local_report(second)
        mesh = raw / "_processing" / "mesh-version"
        initialize_layout(mesh, raw, "mesh-version")
        paths = take_paths(mesh, second)
        paths["destination"].mkdir(parents=True)
        artifact = paths["destination"] / "scene_with_mesh.fbx"
        artifact.write_bytes(b"mesh product")
        subject = paths["destination"] / "subjects" / "actor_with_mesh.fbx"
        subject.parent.mkdir()
        subject.write_bytes(b"subject-only mesh product")
        self.save(mesh / "mesh_batch_report.json", {"files": []})
        report = {"relative_path": second, "artifacts": {
            "subject_actor_fbx": "artifacts/subjects/actor_with_mesh.fbx",
            "fbx": "artifacts/scene_with_mesh.fbx"}}
        self.save(paths["report_path"], report)
        rows, stats = scan_all_takes(self.root, raw / "_processing")
        self.assertEqual(rows[first]["mesh_status"], "NONE")
        self.assertEqual(rows[second]["mesh_fbx_path"], str(artifact))
        self.assertEqual(stats["with_mesh_count"], 1)
        explicit, _ = scan_all_takes(self.root, mesh)
        self.assertEqual(explicit[second]["mesh_fbx_path"], str(artifact))
        unowned = raw / "private_with_mesh.fbx"
        unowned.write_bytes(b"not a mesh product")
        report["artifacts"] = {"fbx": str(unowned)}
        self.save(paths["report_path"], report)
        _, unsafe_stats = scan_all_takes(self.root, raw / "_processing")
        self.assertEqual(unsafe_stats["with_mesh_count"], 0)

    def test_generic_overview_resolves_fbx_relative_to_its_report(self):
        path, data = self.report("standalone.mcp")
        data["artifacts"]["fbx"] = "artifacts/scene.fbx"
        self.save(path, data)
        rows, stats = scan_all_takes(self.root, self.root / "no-mesh")
        row = rows["group1/MP/standalone.mcp"]
        self.assertEqual(row["fbx_path"], str(path.parent / "artifacts" / "scene.fbx"))
        self.assertEqual(row["artifacts_dir"], str(path.parent / "artifacts"))
        self.assertEqual(stats["success_total"], 1)

    def test_only_final_success_is_listed_even_when_review_has_old_artifacts(self):
        self.report('good.mcp')
        self.report('review.mcp', 'NEEDS_REVIEW')
        self.report('failed.mcp', 'FAILED')
        path, data = self.report('wrong-basis.mcp')
        data['quality']['basis'] = 'before_processing'
        self.save(path, data)
        rows, issues = collect_successes(self.root)
        self.assertEqual([row['job_id'] for row in rows], ['good.mcp'])
        self.assertEqual(len(issues), 1)

    def test_missing_current_artifact_is_not_rescued_by_an_old_file(self):
        path, data = self.report('missing.mcp')
        data['artifacts']['fbx'] = str(path.parent / 'artifacts' / 'missing.fbx')
        self.save(path, data)
        rows, issues = collect_successes(self.root)
        self.assertEqual(rows, [])
        self.assertEqual(len(issues), 1)
        self.assertTrue((path.parent / 'artifacts' / 'scene.fbx').is_file())

    def test_duplicate_identity_is_excluded_not_selected_by_directory_order(self):
        self.report('one.mcp', job_id='same')
        self.report('two.mcp', status='FAILED', job_id='same')
        rows, issues = collect_successes(self.root)
        self.assertEqual(rows, [])
        self.assertEqual(len(issues), 1)

    def test_refresh_removes_only_stale_shortcuts_not_original_artifacts(self):
        path, data = self.report('take.mcp')
        original = {name: Path(value).read_bytes() for name, value in data['artifacts'].items()}
        refresh(self.root, self.fixture_shortcut_writer)
        view = self.root / '_success'
        manifest = json.loads((view / '_index.json').read_text(encoding='utf-8'))
        shortcut = view / manifest['entries'][0]['shortcut']
        self.assertTrue(shortcut.is_file())
        data['status'] = 'NEEDS_REVIEW'
        data['quality']['usable'] = False
        self.save(path, data)
        result = refresh(self.root, self.fixture_shortcut_writer)
        self.assertEqual(result['success_count'], 0)
        self.assertFalse(shortcut.exists())
        for name, value in data['artifacts'].items():
            self.assertEqual(Path(value).read_bytes(), original[name])
        self.assertEqual(self.lock.read_bytes(), b'active scheduler lock remains untouched')
        with Path(result['csv']).open(encoding='utf-8-sig', newline='') as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])

    def test_unmanaged_nonempty_view_is_preserved(self):
        self.report('good.mcp')
        view = self.root / '_success'
        view.mkdir()
        foreign = view / 'my-notes.txt'
        foreign.write_text('user content', encoding='utf-8')
        with self.assertRaises(ValueError):
            refresh(self.root, self.fixture_shortcut_writer)
        self.assertEqual(foreign.read_text(encoding='utf-8'), 'user content')

    def test_manifest_cannot_delete_outside_managed_view(self):
        path, _ = self.report('good.mcp')
        view = self.root / '_success'
        view.mkdir()
        payload = {'owner': OWNER, 'output_dir': str(self.root),
                   'entries': [{'shortcut': '../takes/group1/MP/good.mcp/report.json'}]}
        self.save(view / '_index.json', payload)
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            refresh(self.root, self.fixture_shortcut_writer)
        self.assertEqual(path.read_bytes(), before)

    @unittest.skipUnless(os.name == 'nt', 'Windows Explorer shortcuts')
    def test_real_shell_batch_links_open_their_distinct_artifact_directories(self):
        self.report('first scene.mcp')
        self.report('第二条.mcp')
        result = refresh(self.root)
        view = Path(result['view'])
        entries = json.loads((view / '_index.json').read_text(encoding='utf-8'))['entries']
        script = r"""
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$shell = New-Object -ComObject WScript.Shell
$paths = ConvertFrom-Json -InputObject ([Console]::In.ReadToEnd())
@($paths | ForEach-Object { $shell.CreateShortcut($_).TargetPath }) | ConvertTo-Json -Compress
"""
        observed = subprocess.run(
            ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
            input=json.dumps([entry['shortcut_path'] for entry in entries], ensure_ascii=False),
            capture_output=True, text=True, encoding='utf-8',
            creationflags=subprocess.CREATE_NO_WINDOW, check=True, timeout=30,
        )
        targets = {Path(value).resolve() for value in json.loads(observed.stdout)}
        self.assertEqual(targets, {Path(entry['artifacts_directory']) for entry in entries})
        self.assertEqual(len(targets), 2)


if __name__ == '__main__':
    unittest.main()
