import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from view_evoke import Recording
import export_evoke_npy as exporter


class ExportTests(unittest.TestCase):
    def recording(self):
        # Deliberately scrambled source insertion order, with distinct positions.
        poses = {}
        for name, value in [("Head", 4), ("LeftHand", 5), ("RightFoot", 1),
                            ("Spine", 0), ("LeftFoot", 3), ("RightHand", 2)]:
            poses["SmartObject:Character1_" + name] = np.array([
                [value, value + .1, value + .2, 0, 0, 0, 1], [np.nan]*7])
        poses["SmartObject:Character1"] = np.full((2, 7), 999.)
        return Recording(np.array([100, 103]), np.array([0., .03]), poses)

    def test_exact_order_shape_units_and_missing(self):
        rec = self.recording()
        data = exporter.to_array(rec)
        self.assertEqual(data.shape, (2, 6, 4))
        self.assertEqual(data.dtype, np.float32)
        np.testing.assert_allclose(data[0, :, 0], [0, 1000, 2000, 3000, 4000, 5000])
        np.testing.assert_allclose(data[0, :, 3], 0)
        np.testing.assert_allclose(data[1, :, :3], 0)
        np.testing.assert_allclose(data[1, :, 3], 1)
        self.assertTrue(np.isfinite(data).all())

    def test_partial_xyz_marks_whole_tracker_occluded(self):
        rec = self.recording()
        rec.entities["SmartObject:Character1_Head"][0, 1] = np.nan
        np.testing.assert_array_equal(exporter.to_array(rec)[0, 4], [0, 0, 0, 1])

    def test_missing_spine_never_substitutes_root(self):
        rec = self.recording()
        del rec.entities["SmartObject:Character1_Spine"]
        with self.assertRaisesRegex(ValueError, "Character1_Spine"):
            exporter.to_array(rec)

    def test_save_load_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "motion.npy"
            with patch.object(exporter, "load_recording", return_value=self.recording()), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                args = ["unused.csv", "--output", str(output)]
                self.assertEqual(exporter.main(args), 0)
                saved = output.read_bytes()
                self.assertEqual(exporter.main(args), 1)
                self.assertEqual(output.read_bytes(), saved)
            np.testing.assert_allclose(np.load(output, allow_pickle=False), exporter.to_array(self.recording()))


if __name__ == "__main__":
    unittest.main()
