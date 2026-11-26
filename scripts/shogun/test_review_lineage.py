"""Completion proof for already-mutated native review checkpoints."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import unittest

from pipeline_layout import initialize_layout, take_paths
from pipeline_review_plan import checkpoint_completion
import test_review_plan as fixtures


class ReviewLineageTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ReviewPlanTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.make_lineage()

    def make_lineage(self):
        self.original_report, self.original = self.fixture.make_report()
        self.original_vdf = Path(self.original['artifacts']['vdf'])
        self.source = self.fixture.output
        self.checkpoint = take_paths(self.source, self.original['relative_path'])['review_destination'] / 'take.vdf'
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint.write_bytes(b'mutated-review-native-vdf')
        self.report = deepcopy(self.original)
        stat = self.original_vdf.stat()
        self.report.update(
            review_repair=True, source_output=str(self.fixture.source),
            source_report=str(self.original_report),
            source_report_sha256=hashlib.sha256(self.original_report.read_bytes()).hexdigest(),
            artifacts={'vdf': str(self.checkpoint)},
            stages=[{'name': 'save_native', 'status': 'SUCCESS'}],
            processing_source={
                'path': str(self.original_vdf), 'checkpoint_resume_without_fill': True,
                'completed_steps_reused': ['fix_occlusion', 'solve_labeling', 'solve_solving'],
                'review_mutations_enabled': True,
                'checkpoint_fingerprint': {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns},
            },
        )

    def proof(self):
        return checkpoint_completion(self.report, self.source, self.checkpoint)

    def test_mutated_review_inherits_hash_verified_completed_source(self):
        proof = self.proof()
        self.assertEqual(proof['basis'], 'validated_review_lineage')
        self.assertEqual(proof['source_report_sha256'], hashlib.sha256(self.original_report.read_bytes()).hexdigest())
        self.assertEqual(proof['source_completion']['basis'], 'successful_native_stages')
        self.assertEqual(set(proof['completed_steps']), {'fix_occlusion', 'solve_labeling', 'solve_solving'})

    def test_source_report_modification_invalidates_lineage(self):
        self.fixture.save(self.original_report, {**self.original, 'status': 'SUCCESS'})
        with self.assertRaisesRegex(ValueError, 'checksum changed'):
            self.proof()

    def test_different_input_cannot_inherit_completed_stages(self):
        self.report['input_path'] = str(self.fixture.raw / 'other.mcp')
        with self.assertRaisesRegex(ValueError, 'different input'):
            self.proof()

    def test_cropped_review_cannot_hide_unchecked_original_frames(self):
        self.report['range'].update(play_end=990, frame_count=990)
        with self.assertRaisesRegex(ValueError, 'full source checkpoint'):
            self.proof()

    def test_changed_source_native_invalidates_completion(self):
        self.original_vdf.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
            self.proof()

    def test_failed_save_cannot_inherit_completion(self):
        self.report['stages'][0]['status'] = 'FAILED'
        with self.assertRaisesRegex(ValueError, 'successfully saved'):
            self.proof()

    def test_lineage_cannot_cycle_to_current_output(self):
        self.report['source_output'] = str(self.source)
        with self.assertRaisesRegex(ValueError, 'ancestor repair root'):
            self.proof()

    def test_declared_completion_without_actual_ancestor_solves_is_rejected(self):
        original = {**self.original, 'stages': []}
        self.fixture.save(self.original_report, original)
        self.report['source_report_sha256'] = hashlib.sha256(self.original_report.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'Missing successful fill'):
            self.proof()

    def local_lineage(self):
        self.fixture.use_local_layout()
        initialize_layout(self.fixture.output, self.fixture.raw)
        self.make_lineage()

    def test_local_lineage_follows_nested_controls_to_sequence_local_source_report(self):
        self.local_lineage()
        proof = self.proof()
        self.assertEqual(proof['basis'], 'validated_review_lineage')
        self.assertEqual(proof['source_report'], str(self.original_report))
        self.assertEqual(proof['source_report_sha256'], hashlib.sha256(self.original_report.read_bytes()).hexdigest())
        self.assertFalse(self.original_report.is_relative_to(self.fixture.source))
        self.assertFalse(self.checkpoint.is_relative_to(self.source))

    def test_local_lineage_retains_source_report_sha_validation(self):
        self.local_lineage()
        self.fixture.save(self.original_report, {**self.original, 'review_reasons': ['changed evidence']})
        with self.assertRaisesRegex(ValueError, 'checksum changed'):
            self.proof()

    def test_local_lineage_rejects_raw_or_another_take_source_report(self):
        self.local_lineage()
        other_path, _ = self.fixture.make_report('other')
        raw_report = self.fixture.raw / self.original['relative_path']
        for path in (other_path, raw_report):
            with self.subTest(path=path):
                self.report['source_report'] = str(path)
                self.report['source_report_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    self.proof()

    def test_local_lineage_cannot_inherit_neighbor_take_checkpoint(self):
        self.local_lineage()
        _, other = self.fixture.make_report('other')
        foreign = Path(other['artifacts']['vdf'])
        self.original['artifacts']['vdf'] = str(foreign)
        self.fixture.save(self.original_report, self.original)
        self.report['source_report_sha256'] = hashlib.sha256(self.original_report.read_bytes()).hexdigest()
        stat = foreign.stat()
        self.report['processing_source'].update(
            path=str(foreign), checkpoint_fingerprint={'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
        with self.assertRaises(ValueError):
            self.proof()

    def test_local_lineage_rejects_checkpoint_from_different_run_version(self):
        self.local_lineage()
        other_output = self.fixture.source / '_repair_runs/other-run'
        initialize_layout(other_output, self.fixture.raw)
        foreign = take_paths(other_output, self.original['relative_path'])['review_destination'] / 'take.vdf'
        foreign.parent.mkdir(parents=True)
        foreign.write_bytes(b'not this repair checkpoint')
        self.report['artifacts']['vdf'] = str(foreign)
        with self.assertRaises(ValueError):
            checkpoint_completion(self.report, self.source, foreign)



if __name__ == '__main__':
    unittest.main()
