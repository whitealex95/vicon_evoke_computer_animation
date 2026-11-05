"""Convert a recorded CSV to the course notebook's six-tracker order.

    python export_evoke_npy.py record_exports/TIMESTAMP/all_objects.csv

Notebook format: (frames, 6, 4), XYZ in millimeters and occluded_flag.
Missing XYZ is stored as zero with flag 1; visible XYZ has flag 0.
Keep view_evoke.py beside this script; no Matplotlib or Vicon SDK is needed.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from view_evoke import load_recording


IDX = dict(
    spine=0,
    r_foot=1,
    r_hand=2,
    l_foot=3,
    head=4,
    l_hand=5,
)

SUFFIX = dict(
    spine="Spine",
    r_foot="RightFoot",
    r_hand="RightHand",
    l_foot="LeftFoot",
    head="Head",
    l_hand="LeftHand",
)


def to_array(recording, character="Character1"):
    """Map names explicitly; keep every received frame and preserve missing data.

    Output axes are (frame, tracker, component). Positions are global XYZ
    (+X forward, +Y left, +Z up), in mm; component 3 is the occlusion flag.
    Character1 (the root object) is deliberately not substituted for Spine.
    """
    names = {key: f"SmartObject:{character}_{suffix}" for key, suffix in SUFFIX.items()}
    missing = [name for name in names.values() if name not in recording.entities]
    if missing:
        raise ValueError("Missing required tracker columns: " + ", ".join(missing))
    result = np.zeros((len(recording.frames), len(IDX), 4), dtype=np.float32)
    for key, index in IDX.items():
        xyz = recording.entities[names[key]][:, :3]
        present = np.isfinite(xyz).all(axis=1)
        result[present, index, :3] = xyz[present] * 1000
        result[:, index, 3] = ~present
    return result


def class_array(recording, character="Character1", source_fps=None):
    """120 Hz course data; trim only the lead-in, never interpolate across loss.

    Start at the first simultaneously visible six-tracker pose so the notebook's
    forward fill has a valid seed. Interpolate only adjacent source frames with
    visible endpoints; preserve occlusion during dropped frames/tracking loss.
    """
    raw = to_array(recording, character)
    complete = np.flatnonzero(~raw[:, :, 3].any(axis=1))
    if not len(complete):
        raise ValueError("No frame has all six trackers visible together; recapture Spine, Head, hands and feet")
    start = int(complete[0])
    times = recording.times[start:] - recording.times[start]
    raw = raw[start:]
    grid = np.arange(int(np.floor(times[-1] * 120 + 1e-7)) + 1) / 120.0
    result = np.zeros((len(grid), 6, 4), dtype=np.float32)
    result[:, :, 3] = 1
    right = np.searchsorted(times, grid, side="left").clip(0, len(times)-1)
    left = np.maximum(right - 1, 0)
    exact_left = np.isclose(times[left], grid, rtol=0, atol=1e-8)
    exact = exact_left | np.isclose(times[right], grid, rtol=0, atol=1e-8)
    nearest = np.where(exact_left, left, right)
    result[exact] = raw[nearest[exact]]
    between = ~exact
    if between.any():
        lo, hi = left[between], right[between]
        weights = ((grid[between] - times[lo]) / (times[hi] - times[lo]))[:, None, None]
        positions = raw[lo, :, :3] * (1-weights) + raw[hi, :, :3] * weights
        consecutive = (recording.frames[start:][hi] - recording.frames[start:][lo]) == 1
        valid = (raw[lo, :, 3] == 0) & (raw[hi, :, 3] == 0) & consecutive[:, None]
        result[between, :, :3] = np.where(valid[:, :, None], positions, 0)
        result[between, :, 3] = ~valid
    if source_fps is None and len(recording.frames) > 1:
        source_fps = float(np.median(np.diff(recording.frames) / np.diff(recording.times)))
    metadata = {
        "schema": "evoke.course.v1", "frame_rate_hz": 120,
        "source_frame_rate_hz": source_fps, "character": character,
        "tracker_order": list(IDX), "position_units": "mm",
        "axes": ["forward", "left", "up"], "components": ["x", "y", "z", "occluded_flag"],
        "source_start_frame": int(recording.frames[start]),
        "trimmed_leading_samples": start, "trimmed_leading_seconds": float(recording.times[start]),
        "frames": len(result), "duration_s": float(grid[-1]),
        "occluded_samples_per_tracker": result[:, :, 3].sum(axis=0).astype(int).tolist(),
        "resampling": "120 Hz linear interpolation only between consecutive visible source frames; gaps are occluded",
    }
    return result, metadata


def save_class(recording, destination, character="Character1", source_fps=None):
    data, metadata = class_array(recording, character, source_fps)
    destination = Path(destination)
    sidecar = destination.with_suffix(".json")
    if destination.exists() or sidecar.exists():
        raise FileExistsError(f"Class export already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        np.save(output, data, allow_pickle=False)
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path, help="all_objects.csv or capture directory")
    parser.add_argument("--output", type=Path, help="destination .npy (default: motion.npy beside CSV)")
    parser.add_argument("--character", default="Character1", help="recorded character name prefix")
    parser.add_argument("--fps", type=float, help="capture rate for legacy CSVs without Time")
    parser.add_argument("--class-ready", action="store_true", help="trim to a complete pose and resample to 120 Hz; default output: class/motion.npy")
    args = parser.parse_args(argv)
    try:
        source = args.recording / "all_objects.csv" if args.recording.is_dir() else args.recording
        destination = args.output if args.output else (source.parent / "class/motion.npy" if args.class_ready else source.with_name("motion.npy"))
        if destination.suffix.lower() != ".npy":
            raise ValueError("Output filename must end in .npy")
        recording = load_recording(source, args.fps)
        if args.class_ready:
            metadata = save_class(recording, destination, args.character, args.fps)
            print(f"Saved {destination.resolve()} ({metadata['frames']} frames at 120 Hz)")
            return 0
        data = to_array(recording, args.character)
        # Exclusive creation protects recordings and previously exported arrays.
        with destination.open("xb") as output:
            np.save(output, data, allow_pickle=False)
        print(f"Saved {destination.resolve()}")
        print(f"shape={data.shape}, dtype={data.dtype}, XYZ units=mm, last component=occluded_flag")
        print("Tracker order: " + ", ".join(IDX))
        print(f"Frames with any occluded tracker: {data[:, :, 3].any(axis=1).sum()} / {len(data)}")
        for key, index in IDX.items():
            if data[:, index, 3].all():
                print(f"Warning: {key} is occluded throughout the recording; recapture before using it for IK.")
        return 0
    except (OSError, ValueError) as error:
        print(f"Export failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
