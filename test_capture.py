"""Offline end-to-end capture/export tests; no cameras or Vicon SDK required."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from export_evoke_npy import class_array, save_class
from record_evoke import AllObjectsCsv, finalize_csv
from utils import LiveFrames
from view_evoke import Recording, TRACKER_SUFFIXES, load_recording


def sample(frame, missing=()):
    return dict(frame=frame, received_unix_ns=time.time_ns(), markers=[((1, 2, 3), 9)],
                objects=[dict(name="Character1_"+suffix, root="root", present=True,
                              position=None if suffix in missing else (frame, i*100, 1000),
                              quaternion=(0, 0, 0, 1))
                         for i, suffix in enumerate(TRACKER_SUFFIXES)])


class CaptureTests(unittest.TestCase):
    def recording(self, frames=(10, 11, 12, 14)):
        return Recording(np.array(frames), (np.array(frames)-frames[0])/100,
                         {"SmartObject:Character1_"+suffix:
                          np.array([[f/100, i, 1, 0, 0, 0, 1] for f in frames])
                          for i, suffix in enumerate(TRACKER_SUFFIXES)})

    def test_resampling_does_not_compress_gaps(self):
        data, meta = class_array(self.recording())
        self.assertEqual(len(data), 5)  # 0, 8.33, 16.67, 25, 33.33 ms
        self.assertEqual(meta["source_frame_rate_hz"], 100)
        np.testing.assert_allclose(data[:3, 0, 0], [100, 108.33333, 116.66667])
        self.assertTrue((data[3:, :, 3] == 1).all())
        self.assertTrue((data[3:, :, :3] == 0).all())

    def test_occlusion_and_leading_trim(self):
        rec = self.recording((10, 11, 12, 13, 14))
        rec.entities["SmartObject:Character1_Head"][0, :3] = np.nan
        rec.entities["SmartObject:Character1_LeftHand"][2, :3] = np.nan
        data, meta = class_array(rec)
        self.assertEqual(meta["trimmed_leading_samples"], 1)
        self.assertEqual(meta["source_start_frame"], 11)
        self.assertFalse(data[0, :, 3].any())
        self.assertEqual(data[1, 5, 3], 1)
        self.assertEqual(data[2, 5, 3], 1)
        self.assertEqual(data[3, 5, 3], 0)
        rec.entities["SmartObject:Character1_Head"][:, :3] = np.nan
        with self.assertRaisesRegex(ValueError, "all six"):
            class_array(rec)

    def test_120hz_rounded_csv_timestamps_preserve_exact_samples(self):
        rec = self.recording((10, 11, 12, 14))
        rec.times = np.round((rec.frames-10)/120, 9)
        rec.entities["SmartObject:Character1_Head"][2, :3] = np.nan
        data, _ = class_array(rec)
        self.assertEqual(len(data), 5)
        self.assertEqual(data[1, 4, 3], 0)
        self.assertEqual(data[2, 4, 3], 1)
        self.assertTrue(data[3, :, 3].all())
        self.assertFalse(data[4, :, 3].any())

    def test_capture_two_outputs_and_playback(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(temp)/"take"
            with AllObjectsCsv(folder, "take", 10, .01) as export:
                export.write(sample(10, ("Head",)))
                export.write(sample(11))
                export.write(sample(12))
                export.write(sample(14))
            local = load_recording(folder)
            np.testing.assert_allclose(local.times, [0, .01, .02, .04])
            self.assertEqual(len(local.frames), 4)
            self.assertEqual(local.entities["SmartObject:Character1_Head"][1, 6], 1)
            self.assertIn("Trajectory:9", local.entities)
            info = json.loads((folder/"session.json").read_text())
            self.assertEqual(info["missing_source_frames"], 1)
            self.assertEqual(info["class_status"], "ready")
            course = load_recording(folder/"class/motion.npy")
            np.testing.assert_allclose(np.diff(course.times), 1/120)
            self.assertFalse((folder/"frames.jsonl").exists())

    def test_invalid_capture_keeps_local_data(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            folder = Path(temp)/"take"
            with AllObjectsCsv(folder, "take", 10, .01) as export:
                export.write(sample(10, ("Spine", "Head")))
            self.assertTrue((folder/"all_objects.csv").exists())
            self.assertFalse((folder/"class/motion.npy").exists())
            self.assertEqual(json.loads((folder/"session.json").read_text())["class_status"], "unavailable")

    def test_capture_error_finalizes_received_frames(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(temp)/"take"
            with self.assertRaisesRegex(RuntimeError, "disconnect"):
                with AllObjectsCsv(folder, "take", 10, .01) as export:
                    export.write(sample(10))
                    raise RuntimeError("disconnect")
            self.assertTrue(json.loads((folder/"session.json").read_text())["capture_error"])
            self.assertEqual(len(load_recording(folder).frames), 1)

    def test_recovery_ignores_only_incomplete_tail(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()):
            journal = Path(temp)/"frames.jsonl"
            journal.write_text(json.dumps(dict(frame=1, time_s=0, objects={}, markers={}))+'\n{"frame":')
            path = finalize_csv(journal)
            self.assertEqual(len(load_recording(path).frames), 1)

    def test_sequence_duplicate_gap_reset_and_rate_change(self):
        sequence = LiveFrames(10)
        self.assertFalse(sequence.accept(10, 100))
        self.assertTrue(sequence.accept(11, 100))
        self.assertFalse(sequence.accept(11, 100))
        self.assertTrue(sequence.accept(14, 100))
        self.assertEqual(sequence.missing, 2)
        with self.assertRaisesRegex(RuntimeError, "backwards"):
            sequence.accept(1, 100)
        with self.assertRaisesRegex(RuntimeError, "rate changed"):
            sequence.accept(15, 120)

    def test_export_protects_existing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"motion.npy"
            save_class(self.recording(), path)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_class(self.recording(), path)
            self.assertEqual(before, path.read_bytes())

    def test_live_transport_and_course_order(self):
        from evoke_stream import FramePublisher
        from receive_evoke import EvokeReceiver, course_frame
        with contextlib.redirect_stdout(io.StringIO()), FramePublisher("tcp://127.0.0.1:*") as sender:
            with EvokeReceiver(sender.endpoint) as receiver:
                # PUB/SUB subscription handshake is asynchronous; retry until delivery.
                deadline = time.monotonic() + 3
                while True:
                    sender.send(sample(10, ("Head",)), 10, .01)
                    try:
                        packet = receiver.receive(.05)
                        break
                    except TimeoutError:
                        if time.monotonic() >= deadline:
                            raise
                pose = course_frame(packet)
                self.assertEqual(pose.shape, (6, 4))
                self.assertEqual(pose.dtype, np.float32)
                np.testing.assert_array_equal(pose[4], [0, 0, 0, 1])
                np.testing.assert_array_equal(pose[1], [10, 100, 1000, 0])
                self.assertEqual(packet["frame_rate_hz"], 100)


if __name__ == "__main__":
    unittest.main()
