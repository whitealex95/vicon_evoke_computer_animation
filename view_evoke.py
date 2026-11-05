"""Offline 3D playback of record_evoke.py's grouped all_objects.csv files.

    python view_evoke.py PATH_TO_CSV_OR_CAPTURE_FOLDER
    python view_evoke.py --latest

Requires numpy and matplotlib, but no Evoke connection or Vicon SDK.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np

TRACKER_SUFFIXES = ("Spine", "RightFoot", "RightHand", "LeftFoot", "Head", "LeftHand")


def load_npy(path, fps):
    """Read the P5_Twister (frame, tracker, XYZ-mm + occlusion) format."""
    data = np.load(path, allow_pickle=False)
    if data.ndim == 3 and data.shape[1:] == (6, 3):
        raise ValueError("The XYZ-only visualization format (shape frames, 6, 3; meters) is deprecated. "
                         "Use all_objects.csv for local playback or class/motion.npy for the course notebook. "
                         "Regenerate from the CSV with export_evoke_npy.py --class-ready.")
    if data.ndim != 3 or data.shape[1:] != (6, 4) or not len(data):
        raise ValueError("Expected nonempty NPY with shape (frames, 6, 4): XYZ mm, occluded_flag")
    sidecar = Path(path).with_suffix(".json")
    if fps is None and sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("schema") == "evoke.course.v1":
            fps = metadata["frame_rate_hz"]
    if fps is None or not np.isfinite(fps) or fps <= 0:
        raise ValueError("NPY has no timestamps; supply --fps with the capture frame rate")
    if data.dtype.kind not in "fiu" or not np.isin(data[:, :, 3], [0, 1]).all():
        raise ValueError("Expected numeric NPY with occlusion flags 0 or 1")
    entities = {}
    for index, suffix in enumerate(TRACKER_SUFFIXES):
        pose = np.full((len(data), 7), np.nan)
        present = (data[:, index, 3] == 0) & np.isfinite(data[:, index, :3]).all(axis=1)
        pose[present, :3] = data[present, index, :3] / 1000.0
        entities[f"SmartObject:Character1_{suffix}"] = pose
    frames = np.arange(len(data))
    return Recording(frames, frames / fps, entities)


@dataclass
class Recording:
    frames: np.ndarray
    times: np.ndarray
    entities: dict


def load_recording(path, fps=None):
    """Read named column groups; blanks/nonfinite measurements stay missing."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return load_npy(path, fps)
    if path.is_dir():
        path = path / "all_objects.csv"
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        groups, columns, units = (next(reader, []) for _ in range(3))
        if not columns or columns[:2] != ["Frame", "Sub Frame"]:
            raise ValueError("Expected the three-header all_objects.csv format from record_evoke.py")
        if len(groups) != len(columns) or len(units) != len(columns):
            raise ValueError("CSV header lengths do not match")
        specs = []
        for start, name in enumerate(groups):
            if not name:
                continue
            if name.startswith("SmartObject:"):
                fields = ["X", "Y", "Z", "QX", "QY", "QZ", "QW"]
            elif name.startswith("Trajectory:"):
                fields = ["X", "Y", "Z"]
            else:
                raise ValueError(f"Unsupported entity group: {name}")
            if columns[start:start + len(fields)] != fields:
                raise ValueError(f"Unexpected coordinates for {name}")
            if units[start:start + 3] != ["mm"] * 3:
                raise ValueError(f"Expected XYZ in millimeters for {name}")
            if any(existing[0] == name for existing in specs):
                raise ValueError(f"Duplicate entity: {name}")
            specs.append((name, start, len(fields)))
        rows = []
        for number, row in enumerate(reader, 4):
            if not row:
                continue
            if len(row) != len(columns):
                raise ValueError(f"Row {number}: expected {len(columns)} columns, got {len(row)}")
            try:
                rows.append([float(value) if value.strip() else np.nan for value in row])
            except ValueError as error:
                raise ValueError(f"Row {number}: invalid numeric value") from error
    if not rows:
        raise ValueError("Recording has no frames")
    data = np.asarray(rows, dtype=float)
    frames = data[:, 0]
    if not np.isfinite(frames).all() or (frames != np.floor(frames)).any() or (np.diff(frames) <= 0).any():
        raise ValueError("Frame numbers must be finite, integral, and strictly increasing")
    if "Time" in columns and np.isfinite(data[:, columns.index("Time")]).all():
        times = data[:, columns.index("Time")].copy()
        if (np.diff(times) <= 0).any():
            raise ValueError("Time values must be strictly increasing")
        times -= times[0]
    else:
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Recording has no complete Time column; supply --fps with the capture frame rate")
        times = (frames - frames[0]) / fps
    entities = {name: data[:, start:start + width].copy() for name, start, width in specs}
    for values in entities.values():
        values[~np.isfinite(values)] = np.nan
        values[:, :3] /= 1000.0
    return Recording(frames.astype(np.int64), times, entities)


def rotation_matrix(quaternion):
    """Return local axes in world coordinates for an XYZW quaternion."""
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    x, y, z, w = q / norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


class Viewer:
    def __init__(self, recording, title="Evoke recording", axis_length=0.12):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        self.recording = recording
        self.axis_length = axis_length
        self.index = 0
        self.playing = False
        self.anchor = time.perf_counter()
        self.position = 0.0
        self.fig = plt.figure(figsize=(11, 8))
        self.fig.suptitle(title)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(bottom=0.24)
        self.ax.set(xlabel="X forward (m)", ylabel="Y left (m)", zlabel="Z up (m)")
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.view_init(elev=20, azim=-60)
        bounds = []
        self.artists = {}
        colors = plt.get_cmap("tab10")
        for name, values in recording.entities.items():
            valid = values[np.isfinite(values[:, :3]).all(axis=1), :3]
            if len(valid):
                bounds.extend([valid.min(axis=0), valid.max(axis=0)])
            if name.startswith("SmartObject:"):
                color = colors(len(self.artists) % 10)
                point, = self.ax.plot([], [], [], "o", color=color, markersize=7)
                label = self.ax.text(0, 0, 0, name.split(":", 1)[1], fontsize=8, color=color)
                axes = [self.ax.plot([], [], [], color=c, linewidth=1.5)[0] for c in ("r", "g", "b")]
                self.artists[name] = (point, label, axes)
        self.markers, = self.ax.plot([], [], [], ".", color="gray", markersize=5, linestyle="None")
        self.bones = []
        for name in self.artists:
            if not name.endswith("_Spine"):
                continue
            prefix = name[:-len("Spine")]
            for suffix in TRACKER_SUFFIXES[1:]:
                end = prefix + suffix
                if end in self.artists:
                    line, = self.ax.plot([], [], [], color="red" if suffix == "Head" else "black", linewidth=2)
                    self.bones.append((name, end, line))
        if bounds:
            bounds = np.asarray(bounds)
            center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
            radius = max(float(np.ptp(bounds, axis=0).max()) / 2 + axis_length, 0.5)
        else:
            center, radius = np.zeros(3), 1.0
        for setter, value in zip((self.ax.set_xlim, self.ax.set_ylim, self.ax.set_zlim), center):
            setter(value-radius, value+radius)
        self.status = self.fig.text(0.12, 0.19, "")
        self.fig.text(0.12, 0.025, "Drag to rotate • Scroll/toolbar to zoom • Space: play/pause • Left/right: step\n"
                      "Skeleton: spine to head (red), hands/feet (black). RGB: local axes. Gray: markers.", fontsize=9)
        self.slider = Slider(self.fig.add_axes((0.16, 0.13, 0.69, 0.025)), "Sample", 0,
                             max(1, len(recording.frames)-1), valinit=0, valstep=1)
        self.slider.on_changed(self.seek)
        self.button = Button(self.fig.add_axes((0.12, 0.075, 0.15, 0.035)), "Play")
        self.button.on_clicked(self.toggle)
        self.speed = Slider(self.fig.add_axes((0.43, 0.08, 0.4, 0.025)), "Speed", 0.1, 2.0,
                            valinit=1.0, valstep=0.1, valfmt="%1.1fx")
        self.speed.on_changed(self.change_speed)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.timer = self.fig.canvas.new_timer(interval=30)
        self.timer.add_callback(self.tick)
        self.timer.start()
        self.fig.canvas.mpl_connect("close_event", lambda event: self.timer.stop())
        self.draw_frame(0)

    def draw_frame(self, index):
        self.index = index
        visible = 0
        for name, (point, label, axes) in self.artists.items():
            pose = self.recording.entities[name][index]
            present = np.isfinite(pose[:3]).all()
            point.set_data_3d(*([ [v] for v in pose[:3]] if present else [[], [], []]))
            label.set_visible(present)
            if present:
                visible += 1
                label.set_position((pose[0], pose[1]))
                label.set_3d_properties(pose[2], zdir=None)
            rotation = rotation_matrix(pose[3:]) if present else None
            for axis, artist in enumerate(axes):
                if rotation is None:
                    artist.set_data_3d([], [], [])
                else:
                    end = pose[:3] + rotation[:, axis] * self.axis_length
                    artist.set_data_3d(*np.stack((pose[:3], end), axis=1))
        markers = [values[index, :3] for name, values in self.recording.entities.items()
                   if name.startswith("Trajectory:") and np.isfinite(values[index, :3]).all()]
        self.markers.set_data_3d(*(np.asarray(markers).T if markers else [[], [], []]))
        visible_bones = 0
        for start, end, line in self.bones:
            points = np.stack([self.recording.entities[name][index, :3] for name in (start, end)])
            valid = np.isfinite(points).all()
            line.set_data_3d(*(points.T if valid else [[], [], []]))
            visible_bones += int(valid)
        self.status.set_text(f"Frame {self.recording.frames[index]}   |   {self.recording.times[index]:.3f} / "
                             f"{self.recording.times[-1]:.3f} s   |   {visible} objects, {len(markers)} markers, {visible_bones} bones")
        self.fig.canvas.draw_idle()

    def seek(self, value):
        index = min(int(value), len(self.recording.frames)-1)
        self.position = float(self.recording.times[index])
        self.anchor = time.perf_counter()
        self.draw_frame(index)

    def change_speed(self, value):
        self.position = float(self.recording.times[self.index])
        self.anchor = time.perf_counter()

    def toggle(self, event=None):
        if not self.playing and self.index == len(self.recording.frames)-1:
            self.slider.set_val(0)
        self.playing = not self.playing
        self.change_speed(self.speed.val)
        self.button.label.set_text("Pause" if self.playing else "Play")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == " ":
            self.toggle()
        elif event.key in ("left", "right"):
            if self.playing:
                self.toggle()
            self.slider.set_val(max(0, min(len(self.recording.frames)-1,
                                          self.index + (1 if event.key == "right" else -1))))

    def tick(self):
        if not self.playing:
            return
        target = self.position + (time.perf_counter()-self.anchor) * self.speed.val
        index = min(len(self.recording.frames)-1, int(np.searchsorted(self.recording.times, target, side="right"))-1)
        if index != self.index:
            # Moving the playback cursor must not reset the wall-clock anchor.
            self.slider.eventson = False
            self.slider.set_val(index)
            self.slider.eventson = True
            self.draw_frame(index)
        if target >= self.recording.times[-1]:
            self.toggle()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", nargs="?", type=Path, help="CSV, notebook-format NPY, or capture directory")
    parser.add_argument("--latest", action="store_true", help="open newest capture in --record-dir")
    parser.add_argument("--record-dir", type=Path, default=Path(__file__).resolve().parent / "record_exports")
    parser.add_argument("--fps", type=float, help="frame rate for NPY without a class sidecar or older CSVs without Time")
    parser.add_argument("--snapshot", type=Path, help="save first visible frame as an image without opening a window")
    args = parser.parse_args(argv)
    if bool(args.recording) == args.latest:
        parser.error("provide a recording path or --latest")
    try:
        path = args.recording
        if args.latest:
            paths = sorted(args.record_dir.glob("*/all_objects.csv"))
            if not paths:
                raise ValueError(f"No recordings found in {args.record_dir}")
            path = paths[-1]
        recording = load_recording(path, args.fps)
        import matplotlib
        if args.snapshot:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        viewer = Viewer(recording, str(path))
        print(f"Loaded {len(recording.frames)} frames, {len(recording.entities)} entities, "
              f"{recording.times[-1]:.3f} seconds")
        if args.snapshot:
            visible = np.zeros(len(recording.frames), dtype=bool)
            for values in recording.entities.values():
                visible |= np.isfinite(values[:, :3]).all(axis=1)
            viewer.slider.set_val(int(np.argmax(visible)))
            viewer.fig.savefig(args.snapshot, dpi=140)
            plt.close(viewer.fig)
            print(f"Snapshot: {args.snapshot.resolve()}")
        else:
            plt.show()
        return 0
    except (OSError, ValueError, ImportError) as error:
        print(f"Viewer error: {error}", file=sys.stderr)
        if isinstance(error, ImportError):
            print("Install viewer dependencies: python -m pip install numpy matplotlib", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
