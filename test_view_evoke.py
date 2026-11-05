"""Offline parser and headless playback checks."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from view_evoke import load_recording, rotation_matrix, Viewer


class ViewerTests(unittest.TestCase):
    def test_notebook_npy_skeleton_and_occlusion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.npy"
            data = np.zeros((2, 6, 4), dtype=np.float32)
            data[0, :, :3] = [[0, 0, 1000], [-200, 0, 100], [-500, 0, 900],
                              [200, 0, 100], [0, 0, 1600], [500, 0, 900]]
            data[1, :, 3] = 1
            np.save(path, data)
            with self.assertRaisesRegex(ValueError, "--fps"):
                load_recording(path)
            rec = load_recording(path, fps=100)
            np.testing.assert_allclose(rec.times, [0, .01])
            np.testing.assert_allclose(rec.entities["SmartObject:Character1_Spine"][0, :3], [0, 0, 1])
            viewer = Viewer(rec)
            try:
                self.assertEqual(len(viewer.bones), 5)
                for start, end, line in viewer.bones:
                    self.assertEqual(len(line.get_data_3d()[0]), 2)
                viewer.fig.canvas.draw()
                viewer.slider.set_val(1)
                for start, end, line in viewer.bones:
                    self.assertEqual(len(line.get_data_3d()[0]), 0)
                viewer.fig.canvas.draw()
            finally:
                plt.close(viewer.fig)
            np.save(path, np.zeros((2, 6, 3)))
            with self.assertRaisesRegex(ValueError, "shape"):
                load_recording(path, fps=100)

    def capture(self, directory, times=(0, .01, .04)):
        path = Path(directory) / "all_objects.csv"
        with path.open("w", newline="") as output:
            csv.writer(output).writerows([
                ["", "", "", "SmartObject:Head"] + [""]*6 + ["Trajectory:7", "", ""],
                ["Frame", "Sub Frame", "Time", "X", "Y", "Z", "QX", "QY", "QZ", "QW", "X", "Y", "Z"],
                ["", "", "s", "mm", "mm", "mm", "", "", "", "", "mm", "mm", "mm"],
                [100, 0, times[0]] + [""]*10,
                [101, 0, times[1], 1000, 2000, 3000, 0, 0, 0, 1, 500, 600, 700],
                [104, 0, times[2], 2000, 2000, 3000, 0, 0, 0, 0, "", "", ""],
            ])
        return path

    def test_units_missing_data_and_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            rec = load_recording(self.capture(directory))
        np.testing.assert_allclose(rec.times, [0, .01, .04])
        np.testing.assert_allclose(rec.entities["SmartObject:Head"][1], [1, 2, 3, 0, 0, 0, 1])
        self.assertTrue(np.isnan(rec.entities["SmartObject:Head"][0]).all())
        self.assertTrue(np.isnan(rec.entities["Trajectory:7"][2]).all())

    def test_missing_and_bad_times(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.capture(directory, ("", "", ""))
            with self.assertRaisesRegex(ValueError, "--fps"):
                load_recording(path)
            np.testing.assert_allclose(load_recording(path, fps=100).times, [0, .01, .04])
            path = self.capture(directory, (0, .1, .05))
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                load_recording(path)

    def test_quaternion_order(self):
        np.testing.assert_allclose(rotation_matrix([0, 0, 0, 2]), np.eye(3))
        np.testing.assert_allclose(rotation_matrix([0, 0, 1, 1]) @ [1, 0, 0], [0, 1, 0], atol=1e-15)
        self.assertIsNone(rotation_matrix([0, 0, 0, 0]))
        self.assertIsNone(rotation_matrix([0, 0, np.nan, 1]))

    def test_render_seek_and_timed_playback(self):
        with tempfile.TemporaryDirectory() as directory:
            viewer = Viewer(load_recording(self.capture(directory)))
            try:
                self.assertFalse(viewer.artists["SmartObject:Head"][1].get_visible())
                viewer.slider.set_val(1)
                viewer.fig.canvas.draw()
                self.assertTrue(viewer.artists["SmartObject:Head"][1].get_visible())
                viewer.slider.set_val(0)
                viewer.toggle()
                with patch("view_evoke.time.perf_counter", return_value=viewer.anchor + .025):
                    viewer.tick()
                self.assertEqual(viewer.index, 1)
                with patch("view_evoke.time.perf_counter", return_value=viewer.anchor + .05):
                    viewer.tick()
                self.assertEqual(viewer.index, 2)
                self.assertFalse(viewer.playing)
                viewer.fig.savefig(Path(directory) / "preview.png")
            finally:
                plt.close(viewer.fig)


if __name__ == "__main__":
    unittest.main()
