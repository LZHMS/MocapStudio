"""Unit regressions for pipeline_denoise: static windows get smoothed,
moving segments stay byte-identical."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from pipeline_denoise import denoise_bone


def native_call(_name, operation, *args):
    return operation(*args)


class FakeApi:
    def __init__(self, trans, rot, exists, marker=None, exists_t=None, exists_r=None):
        self.trans = trans
        self.rot = rot
        self.exists = exists
        self.exists_t = exists_t if exists_t is not None else exists
        self.exists_r = exists_r if exists_r is not None else exists
        self.marker = marker if marker is not None else trans
        self.written = {}

    def GetBoneTranslationAnimation(self, subject, bone, skeleton, space):
        return (self.trans[:, 0], self.trans[:, 1], self.trans[:, 2], self.exists_t)

    def GetBoneRotationAnimation(self, subject, bone, skeleton, space):
        return (self.rot[:, 0], self.rot[:, 1], self.rot[:, 2], self.exists_r)

    def SetBoneTranslationAnimation(self, subject, bone, skeleton, space, x, y, z, e):
        self.written["trans"] = np.column_stack([x, y, z])
        self.written["trans_lists"] = all(isinstance(channel, list) for channel in (x, y, z)) and isinstance(e, list)

    def SetBoneRotationAnimation(self, subject, bone, skeleton, space, x, y, z, e):
        self.written["rot"] = np.column_stack([x, y, z])

    def GetTrajectory(self, subject, marker):
        return (self.marker[:, 0], self.marker[:, 1], self.marker[:, 2], self.exists)

    def SetTrajectory(self, subject, marker, x, y, z, e):
        self.written["marker"] = np.column_stack([x, y, z])


class DenoiseTests(unittest.TestCase):
    def build(self):
        n = 120
        rng = np.random.default_rng(7)
        trans = np.zeros((n, 3))
        # Moving head/tail: 0.5 cm/frame. Static-but-jittering middle 40..79:
        # violent +-0.4 cm oscillation whose net drift across any window is ~0.
        trans[:, 1] = np.where(np.arange(n) < 40, np.arange(n) * 0.5,
                               np.where(np.arange(n) < 80, 20.0, 20.0 + (np.arange(n) - 79) * 0.5))
        trans[40:80] += rng.normal(0, 0.4, size=(40, 3))
        rot = np.zeros((n, 3))
        rot[40:80, 0] = rng.normal(0, 3.0, size=40)
        exists = np.ones(n, dtype=bool)
        return n, trans, rot, exists

    def test_static_window_smoothed_motion_untouched(self):
        n, trans, rot, exists = self.build()
        api = FakeApi(trans, rot, exists)
        ranges = {"slice_start": 0, "slice_stop": n}
        result = denoise_bone(native_call, api, "actor", "LFRM", "Solving", ranges)
        self.assertGreater(result["smoothed_frames"], 0)
        self.assertIn("trans", api.written)
        self.assertTrue(api.written["trans_lists"])
        before = np.std(trans[45:75, 0])
        after = np.std(api.written["trans"][45:75, 0])
        self.assertLess(after, before * 0.6)
        # moving regions untouched
        self.assertTrue(np.allclose(api.written["trans"][:35], trans[:35], atol=1e-9))
        self.assertTrue(np.allclose(api.written["trans"][90:], trans[90:], atol=1e-9))

    def test_pure_motion_noop(self):
        n = 120
        trans = np.column_stack([np.arange(n) * 0.5, np.zeros(n), np.zeros(n)])
        rot = np.zeros((n, 3))
        exists = np.ones(n, dtype=bool)
        api = FakeApi(trans, rot, exists)
        result = denoise_bone(native_call, api, "actor", "LFRM", "Solving", {"slice_start": 0, "slice_stop": n})
        self.assertEqual(result["smoothed_frames"], 0)
        self.assertEqual(api.written, {})

    def test_finger_rotation_only(self):
        n = 120
        rng = np.random.default_rng(5)
        trans = np.zeros((n, 3))
        rot = np.zeros((n, 3))
        rot[40:80] = rng.normal(0, 3.0, size=(40, 3))
        exists_t = np.zeros(n, dtype=bool)              # finger local translation never animated
        exists_r = np.ones(n, dtype=bool)
        api = FakeApi(trans, rot, exists_r, exists_t=exists_t, exists_r=exists_r)
        result = denoise_bone(native_call, api, "actor", "LeftHandIndex1", "Solving", {"slice_start": 0, "slice_stop": n})
        self.assertGreater(result["smoothed_frames"], 0)
        self.assertIn("rot", api.written)
        self.assertNotIn("trans", api.written)          # zero-translation channel left untouched

    def test_marker_jitter_smoothed_motion_untouched(self):
        from pipeline_denoise import denoise_marker
        n = 120
        rng = np.random.default_rng(11)
        marker = np.zeros((n, 3))
        marker[:, 1] = np.where(np.arange(n) < 40, np.arange(n) * 0.5,
                                np.where(np.arange(n) < 80, 20.0, 20.0 + (np.arange(n) - 79) * 0.5))
        marker[40:80] += rng.normal(0, 0.4, size=(40, 3))
        exists = np.ones(n, dtype=bool)
        api = FakeApi(marker, np.zeros((n, 3)), exists, marker=marker)
        result = denoise_marker(native_call, api, "actor", "actor:LFRM", {"slice_start": 0, "slice_stop": n})
        self.assertGreater(result["smoothed_frames"], 0)
        self.assertIn("marker", api.written)
        before = np.std(marker[45:75, 0])
        after = np.std(api.written["marker"][45:75, 0])
        self.assertLess(after, before * 0.6)
        self.assertTrue(np.allclose(api.written["marker"][:35], marker[:35], atol=1e-9))
        self.assertTrue(np.allclose(api.written["marker"][90:], marker[90:], atol=1e-9))


if __name__ == "__main__":
    unittest.main()
