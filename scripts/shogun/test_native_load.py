"""Byte-preserving native load selection and owned snapshot lifetime; no SDK."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline_session import ShogunSession, ShogunSessionError, load_native_file


class NativeLoadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.session = ShogunSession("unused.exe", self.root / "session", self.root / "native.log", 12345)
        self.calls = []
        self.session.api = SimpleNamespace(LoadFile=self.calls.append)

    def tearDown(self):
        try:
            self.session._close()
        finally:
            self.temporary.cleanup()

    def source(self, suffix=".vdf"):
        path = self.root / ("original" + suffix)
        path.write_bytes(b"exact original native scene bytes\x00\xff")
        return path

    def test_vdf_load_uses_identical_owned_bytes_until_session_closes(self):
        source = self.source()
        original = source.read_bytes()
        observed = []

        def load(path):
            snapshot = Path(path)
            self.assertNotEqual(snapshot, source)
            self.assertEqual(snapshot.suffix, ".vdf")
            self.assertEqual(snapshot.read_bytes(), original)
            observed.append(snapshot)

        self.session.api.LoadFile = load
        audit = []
        first = load_native_file(self.session, source, audit=audit)
        second = load_native_file(self.session, source, audit=audit)
        self.assertEqual(len(audit), 2)
        self.assertEqual(len(self.session.native_load_audit), 2)
        self.assertNotEqual(observed[0], observed[1])
        self.assertTrue(all(path.is_file() for path in observed))
        self.assertEqual(first["source_path"], str(source.resolve()))
        self.assertEqual(first["source_before"]["sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(first["source_before"], first["source_after"])
        self.assertEqual(second["snapshot_before"], second["snapshot_after"])
        self.assertEqual(source.read_bytes(), original)
        self.session._close()
        self.assertTrue(all(not path.exists() for path in observed))
        self.assertEqual(source.read_bytes(), original)

    def test_mcp_stays_at_original_location_for_relative_x2d_dependencies(self):
        source = self.source(".mcp")
        result = load_native_file(self.session, source)
        self.assertEqual(self.calls, [source.as_posix()])
        self.assertEqual(result["method"], "original_mcp")
        self.assertIsNone(self.session._native_load_directory)

    def test_native_failure_is_not_retried_or_disguised_and_copy_is_owned(self):
        source = self.source()
        failure = RuntimeError("Invalid Parameter Value")

        def fail(path):
            self.calls.append(path)
            raise failure

        self.session.api.LoadFile = fail
        with self.assertRaises(RuntimeError) as raised:
            load_native_file(self.session, source)
        self.assertIs(raised.exception, failure)
        self.assertEqual(len(self.calls), 1)
        entry = self.session.native_load_audit[0]
        self.assertEqual(entry["status"], "FAILED")
        self.assertEqual(entry["source_before"], entry["source_after"])
        self.assertEqual(entry["snapshot_before"], entry["snapshot_after"])
        snapshot = Path(entry["snapshot_path"])
        self.assertTrue(snapshot.is_file())
        self.session._close()
        self.assertFalse(snapshot.exists())
        self.assertTrue(source.is_file())

    def test_concurrent_source_change_during_copy_is_rejected_before_sdk(self):
        source = self.source()
        import shutil
        copy = shutil.copyfile

        def changed_copy(original, snapshot):
            copy(original, snapshot)
            original.write_bytes(b"concurrent external source replacement")

        with patch("pipeline_session.shutil.copyfile", side_effect=changed_copy):
            with self.assertRaisesRegex(ShogunSessionError, "source changed or snapshot differs"):
                load_native_file(self.session, source)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.session.native_load_audit[0]["status"], "FAILED")

    def test_source_change_during_sdk_read_is_rejected(self):
        source = self.source()
        self.session.api.LoadFile = lambda path: source.write_bytes(b"external source change")
        with self.assertRaisesRegex(ShogunSessionError, "source or snapshot changed during load"):
            load_native_file(self.session, source)
        entry = self.session.native_load_audit[0]
        self.assertNotEqual(entry["source_before"], entry["source_after"])
        self.assertEqual(entry["snapshot_before"], entry["snapshot_after"])

    def test_snapshot_mutation_during_sdk_read_is_rejected(self):
        source = self.source()
        before = source.read_bytes()
        self.session.api.LoadFile = lambda path: Path(path).write_bytes(b"changed snapshot")
        with self.assertRaisesRegex(ShogunSessionError, "source or snapshot changed during load"):
            load_native_file(self.session, source)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(self.session.native_load_audit[0]["status"], "FAILED")

    def test_inexact_copy_never_reaches_sdk(self):
        source = self.source()
        with patch("pipeline_session.shutil.copyfile", side_effect=lambda source, snapshot: snapshot.write_bytes(b"wrong bytes")):
            with self.assertRaisesRegex(ShogunSessionError, "source changed or snapshot differs"):
                load_native_file(self.session, source)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
