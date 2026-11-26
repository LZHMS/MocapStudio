"""Repair-entry safety boundaries without a licensed native runtime."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_review_plan as fixtures

from step3_repair_reviews import main


class ReviewEntrySafetyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        (self.source / 'takes').mkdir(parents=True)
        self.output = self.root / 'repaired'
        self.sentinel = self.source / 'original.vdf'
        self.sentinel.write_bytes(b'unchanged source')
        self.arguments = ['-SourceOutput', str(self.source), '-OutputDir', str(self.output), '-DryRun']

    def test_camera_waiver_cannot_expand_to_furniture_or_similarly_named_objects(self):
        for names in ('desk', 'camera_10', 'camera_1,orange_chair_1', 'camera_1,camera_1'):
            with self.subTest(names=names), self.assertRaises(ValueError):
                main([*self.arguments, '-CameraExemptions', names])
        self.assertEqual(self.sentinel.read_bytes(), b'unchanged source')
        self.assertFalse(self.output.exists())

    def test_automatic_crop_and_correction_cannot_expand_beyond_authorized_bounds(self):
        for option, value in (('-TrimMaxLossFraction', '.100001'), ('-TrimMinFrames', '119'),
                              ('-PropJitterMaxCorrection', '5.0001'), ('-PropJitterMaxGap', '6')):
            with self.subTest(option=option), self.assertRaises(ValueError):
                main([*self.arguments, option, value])
        self.assertFalse(self.output.exists())

    def test_source_or_target_lock_prevents_a_second_repair_batch(self):
        for parent in (self.source, self.output):
            parent.mkdir(exist_ok=True)
            lock = parent / '.pipeline.lock'
            lock.write_text('{"pid":123}', encoding='utf-8')
            try:
                with self.subTest(parent=parent), self.assertRaises(RuntimeError):
                    main(self.arguments)
                self.assertEqual(lock.read_text(encoding='utf-8'), '{"pid":123}')
                self.assertEqual(self.sentinel.read_bytes(), b'unchanged source')
            finally:
                lock.unlink()

    def local_source(self):
        fixture = fixtures.ReviewPlanTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.use_local_layout()
        fixture.make_report()
        return fixture

    def test_local_source_without_takes_directory_can_dry_plan_without_native_or_manifest_writes(self):
        fixture = self.local_source()
        before = {path: path.read_bytes() for path in fixture.root.rglob('*') if path.is_file()}
        text = io.StringIO()
        with patch('step3_repair_reviews.discover_shogun', side_effect=AssertionError('native discovery forbidden')), \
                patch('step3_repair_reviews.run_jobs', side_effect=AssertionError('native workers forbidden')), \
                redirect_stdout(text):
            result = main(['-SourceOutput', str(fixture.source), '-OutputDir', str(fixture.output), '-DryRun'])
        self.assertEqual(result, 0)
        self.assertIn('"selected": 1', text.getvalue())
        self.assertFalse((fixture.source / 'takes').exists())
        self.assertFalse(fixture.output.exists())
        self.assertEqual(before, {path: path.read_bytes() for path in fixture.root.rglob('*') if path.is_file()})

    def test_local_dry_run_default_nested_control_is_not_created(self):
        fixture = self.local_source()
        text = io.StringIO()
        with patch('step3_repair_reviews.discover_shogun', side_effect=AssertionError('native discovery forbidden')), \
                redirect_stdout(text):
            result = main(['-SourceOutput', str(fixture.source), '-DryRun'])
        self.assertEqual(result, 0)
        self.assertIn(str(fixture.source / '_repair_runs'), text.getvalue())
        self.assertFalse((fixture.source / '_repair_runs').exists())

    def test_local_source_lock_still_blocks_dry_plan(self):
        fixture = self.local_source()
        lock = fixture.source / '.pipeline.lock'
        lock.write_text('{"pid":123}', encoding='utf-8')
        with self.assertRaises(RuntimeError):
            main(['-SourceOutput', str(fixture.source), '-OutputDir', str(fixture.output), '-DryRun'])
        self.assertEqual(lock.read_text(encoding='utf-8'), '{"pid":123}')
        self.assertFalse(fixture.output.exists())


if __name__ == '__main__':
    unittest.main()
