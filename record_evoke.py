"""Record live Evoke motion locally and export the course's six-tracker array.

    python record_evoke.py
    python record_evoke.py --duration 30

Open Evoke in Live mode and enable Include Unlabeled Reconstructions.
No recording or fixed object/marker count is needed. Press Ctrl+C to stop.
Output: record_exports/<timestamp>/all_objects.csv
Also saves session.json and, when all six trackers are usable, class/motion.npy
at 120 Hz with class/motion.json metadata. Use --character for another prefix.

Each received frame becomes one data row. Three header rows contain entity
names/trajectory IDs, coordinate names, and units, like Vicon's grouped CSVs.
Smart objects have X,Y,Z,QX,QY,QZ,QW; unlabeled trajectories have X,Y,Z.
Missing/occluded measurements are blank. New entities add column groups.
Unlabeled trajectory IDs may change after tracking loss.
XYZ is global millimeters (+X forward, +Y left, +Z up); quaternion is XYZW.
Object poses use the root-segment origin. Sub Frame is always 0.
Time is seconds from the first captured frame, using the source frame rate.

Frames are saved to a recovery journal during capture; the CSV is assembled
on Ctrl+C, duration completion, or a handled capture error, using all IDs seen.
If the process is forcibly killed, recover the CSV with:
    python record_evoke.py --recover PATH_TO_frames.jsonl

Keep the project Python modules together.
Use Python 3.11: conda activate vicon
"""

import argparse
import csv
from functools import partial
import json
from pathlib import Path
import sys

from utils import record_live, load_evoke_api, positive_float, positive_int


class AllObjectsCsv:
    """Spool frames to disk, then write a rectangular CSV with all entity groups."""

    def __init__(self, directory, capture, first_frame, period_s, *, character="Character1"):
        self.directory = Path(directory)
        self.capture = capture
        self.first_frame = first_frame
        self.period_s = period_s
        self.frame_count = 0
        self.character = character

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "all_objects.csv"
        self.journal = self.directory / "frames.jsonl"
        self.file = self.journal.open("x", encoding="utf-8")
        return self

    def write(self, sample):
        row = {"frame": sample["frame"],
               "time_s": f"{(sample['frame'] - self.first_frame) * self.period_s:.9f}",
               "objects": {obj["name"]: (
                   list(obj["position"]) if obj["present"] and obj["position"] is not None else [""] * 3
               ) + (
                   list(obj["quaternion"]) if obj["present"] and obj["quaternion"] is not None else [""] * 4
               ) for obj in sample["objects"]},
               "markers": {str(trajectory_id): list(xyz) for xyz, trajectory_id in sample["markers"]}}
        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.file.flush()
        self.frame_count += 1

    def __exit__(self, *exception):
        self.file.close()
        print("Writing CSV with all object and trajectory columns...", flush=True)
        try:
            finalize_csv(self.journal)
        except BaseException:
            print(f"CSV finalization did not finish. Recover saved frames using: "
                  f'python record_evoke.py --recover "{self.journal}"', file=sys.stderr)
            raise
        finalize_capture(self.path, self.character, 1.0 / self.period_s,
                         interrupted=exception[0] is not None and exception[0] is not KeyboardInterrupt)


def journal_rows(path):
    """Read complete records, ignoring only an interrupted final write."""
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.endswith("\n"):
                print("Ignoring an incomplete final recovery record.", file=sys.stderr)
                break
            yield json.loads(line)


def finalize_csv(journal):
    """Two streaming passes keep memory bounded by entity count, not duration."""
    journal = Path(journal)
    destination = journal.parent / "all_objects.csv"
    if destination.exists():
        raise FileExistsError(f"CSV already exists: {destination}")
    objects, markers = set(), set()
    for row in journal_rows(journal):
        objects.update(row["objects"])
        markers.update(row["markers"])
    objects = sorted(objects)
    markers = sorted(markers, key=int)
    groups = ["", "", ""]
    coordinates = ["Frame", "Sub Frame", "Time"]
    units = ["", "", "s"]
    for name in objects:
        groups.extend([f"SmartObject:{name}"] + [""] * 6)
        coordinates.extend(["X", "Y", "Z", "QX", "QY", "QZ", "QW"])
        units.extend(["mm"] * 3 + [""] * 4)
    for marker in markers:
        groups.extend([f"Trajectory:{marker}", "", ""])
        coordinates.extend(["X", "Y", "Z"])
        units.extend(["mm"] * 3)
    # Build beside the destination so a failed conversion never leaves a partial CSV.
    temporary = journal.parent / "all_objects.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerows([groups, coordinates, units])
        for row in journal_rows(journal):
            # Older recovery journals did not store time; leave it unknown.
            values = [row["frame"], 0, row.get("time_s", "")]
            for name in objects:
                values.extend(row["objects"].get(name, [""] * 7))
            for marker in markers:
                values.extend(row["markers"].get(marker, [""] * 3))
            writer.writerow(values)
    temporary.rename(destination)
    journal.unlink()  # The complete CSV now contains every saved record.
    return destination


def display_counts(sample, character="Character1"):
    from view_evoke import TRACKER_SUFFIXES
    names = ", ".join(obj["name"] for obj in sample["objects"]) or "none"
    visible = {obj["name"] for obj in sample["objects"] if obj["present"] and obj["position"] is not None}
    missing = [suffix for suffix in TRACKER_SUFFIXES if f"{character}_{suffix}" not in visible]
    return (f"Frame {sample['frame']} | Smart objects: {len(sample['objects'])} ({names})"
            f" | Unlabeled markers: {len(sample['markers'])}"
            f" | Course trackers visible: {6-len(missing)}/6"
            + (f"; missing: {', '.join(missing)}" if missing else ""))


def finalize_capture(path, character="Character1", source_fps=None, *, interrupted=False):
    """Keep the full CSV even when tracking quality prevents a course export."""
    from export_evoke_npy import save_class
    from view_evoke import load_recording
    recording = load_recording(path, source_fps)
    if source_fps is None and len(recording.frames) > 1:
        source_fps = float((recording.frames[-1] - recording.frames[0]) / recording.times[-1])
    info = {
        "schema": "evoke.session.v1", "character": character,
        "frame_rate_hz": source_fps, "samples": len(recording.frames),
        "first_frame": int(recording.frames[0]), "last_frame": int(recording.frames[-1]),
        "missing_source_frames": int(recording.frames[-1] - recording.frames[0] + 1 - len(recording.frames)),
        "duration_s": float(recording.times[-1]), "capture_error": interrupted,
        "position_units": "mm", "quaternion_order": "xyzw", "axes": ["forward", "left", "up"],
        "local_recording": "all_objects.csv",
    }
    try:
        destination = Path(path).parent / "class" / "motion.npy"
        metadata = save_class(recording, destination, character, source_fps)
        info.update(class_export="class/motion.npy", class_status="ready", class_metadata=metadata)
        print(f"Class motion: {destination.resolve()} ({metadata['frames']} frames at 120 Hz)", flush=True)
        if any(metadata["occluded_samples_per_tracker"]):
            print("Warning: class motion includes occlusion; the notebook holds the previous pose during these gaps.", flush=True)
    except (ValueError, OSError) as error:
        info.update(class_export=None, class_status="unavailable", class_error=str(error))
        print(f"Class export unavailable: {error}. Full local CSV is preserved.", file=sys.stderr)
    (Path(path).parent / "session.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def report_csv(export):
    if export.path.exists():
        print(f"CSV: {export.path.resolve()}", flush=True)
    else:
        print(f"Recovery data: {export.journal.resolve()}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost:801", help="Evoke computer:DataStream port")
    parser.add_argument("--api-port", type=positive_int, default=52800)
    parser.add_argument("--duration", type=positive_float, help="seconds to capture; default: until Ctrl+C")
    parser.add_argument("--hz", type=positive_float, default=5.0, help="console updates/s; CSV saves every received frame")
    parser.add_argument("--connect-timeout", type=positive_float, default=10.0)
    parser.add_argument("--idle-timeout", type=positive_float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "record_exports")
    parser.add_argument("--recover", type=Path, help="convert a frames.jsonl recovery journal to CSV without connecting to Evoke")
    parser.add_argument("--character", default="Character1", help="six-tracker character prefix for the class export")
    parser.add_argument("--publish", help="publish live JSON frames, e.g. tcp://0.0.0.0:5556")
    parser.add_argument("--stream-only", action="store_true", help="publish without saving CSV (requires --publish)")
    args = parser.parse_args(argv)
    if args.stream_only and not args.publish:
        parser.error("--stream-only requires --publish")
    if args.recover and args.publish:
        parser.error("--recover cannot be combined with --publish")
    if args.recover is not None:
        try:
            path = finalize_csv(args.recover)
            print(f"CSV: {path.resolve()}", flush=True)
            finalize_capture(path, args.character)
            return 0
        except Exception as error:
            print(f"Recovery failed: {error}", file=sys.stderr)
            return 1
    if sys.version_info >= (3, 14):
        print("Use the tested Python 3.11 environment: conda activate vicon", file=sys.stderr)
        return 1
    try:
        from vicon_dssdk import ViconDataStream
        core_client, playback_services = load_evoke_api()
        csv_factory = partial(AllObjectsCsv, character=args.character)
        display = partial(display_counts, character=args.character)
        if args.publish:
            from evoke_stream import FramePublisher, StreamingExport
            with FramePublisher(args.publish) as publisher:
                factory = partial(StreamingExport, publisher=publisher,
                                  csv_factory=None if args.stream_only else csv_factory)
                return record_live(args, ViconDataStream, core_client, playback_services,
                                   export_factory=factory, display=display,
                                   summarize=lambda export: export.report(report_csv))
        return record_live(args, ViconDataStream, core_client, playback_services,
                           export_factory=csv_factory, display=display, summarize=report_csv)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
