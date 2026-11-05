"""Vicon SDK utilities and live capture with quaternion XYZW orientation."""

import argparse
from contextlib import ExitStack
from datetime import datetime
import math
import json
import os
from pathlib import Path
import sys
import time


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be an integer greater than zero")
    return number


def load_evoke_api():
    """Use installed packages or the official pure-Python API shipped with Evoke."""
    try:
        from evoke_api import PlaybackServices
        from vicon_core_api import Client
    except ImportError:
        vicon_dir = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Vicon"
        for sdk_dir in sorted(vicon_dir.glob("Evoke*/SDK/Python"), reverse=True):
            if (sdk_dir / "evoke_api/evoke_api").is_dir():
                sys.path.extend([str(sdk_dir / "evoke_api"), str(sdk_dir / "vicon_core_api")])
                break
        from evoke_api import PlaybackServices
        from vicon_core_api import Client
    return Client, PlaybackServices


def api_value(reply):
    """Check Vicon API result codes, including calls without a return value."""
    if isinstance(reply, tuple):
        result, *values = reply
    else:
        result, values = reply, []
    if not result:
        raise RuntimeError(f"Evoke API failed: {result}")
    return values[0] if len(values) == 1 else values


def read_positions(client, subject=None):
    """Read position and orientation from the same successful GetFrame() result."""
    subjects = client.GetSubjectNames()
    selected = subjects if subject is None else [subject]
    objects = []
    for name in selected:
        obj = {"name": name, "root": "", "position": None,
               "quaternion": None, "present": name in subjects}
        if obj["present"]:
            root = client.GetSubjectRootSegmentName(name)
            xyz, position_occluded = client.GetSegmentGlobalTranslation(name, root)
            quaternion, quaternion_occluded = client.GetSegmentGlobalRotationQuaternion(name, root)
            obj.update(root=root,
                       position=None if position_occluded else tuple(xyz),
                       quaternion=None if quaternion_occluded else tuple(quaternion))
        objects.append(obj)
    return {
        "frame": client.GetFrameNumber(),
        "subjects": subjects,
        "objects": objects,
        # SDK format: [((x, y, z), trajectory_id), ...]. Preserve every marker.
        "markers": sorted(client.GetUnlabeledMarkers(), key=lambda marker: marker[1]),
    }


def connect_datastream(client, host, timeout, sdk):
    client.SetConnectionTimeout(1000)
    deadline = time.monotonic() + timeout
    while not client.IsConnected():
        try:
            client.Connect(host)
        except sdk.DataStreamException as error:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Cannot connect to DataStream at {host}: {error}") from error
            time.sleep(0.1)


def require_live(state, original=None):
    if state.mode.name != "ELive":
        raise RuntimeError(
            "Evoke is not in Live mode. Close the recording/review view in Evoke "
            "to return to Live, with cameras connected and tracking enabled."
        )
    if original is not None and state.session_id != original.session_id:
        raise RuntimeError("The live synchronization session changed. Saved data is preserved; start a new export.")


class LiveFrames:
    """Keep source frame numbers and expose gaps instead of inventing live data."""

    def __init__(self, previous):
        self.previous = previous
        self.first = None
        self.rate = None
        self.missing = 0

    def accept(self, frame, rate):
        if not math.isfinite(rate) or rate <= 0:
            raise RuntimeError(f"Invalid live frame rate: {rate}")
        if self.first is None and frame <= self.previous:
            return False  # Discard any cached packets from before this capture.
        if frame < self.previous:
            raise RuntimeError(f"Live frame numbers reset or moved backwards ({self.previous} -> {frame}). Start a new export.")
        if self.rate is not None and not math.isclose(rate, self.rate, rel_tol=1e-6):
            raise RuntimeError("The live frame rate changed. Start a new export to keep CSV times accurate.")
        if frame == self.previous:
            return False
        if self.first is None:
            self.first, self.rate = frame, rate
        else:
            self.missing += frame - self.previous - 1
        self.previous = frame
        return True


def record_live(args, sdk, core_client, playback_services, *,
                export_factory, display, summarize):
    """Receive live frames without sending commands that change Evoke's state."""
    api = client = export = None
    sequence = None
    timing = {'process_entry_utc_ns': time.time_ns(), 'blank_frames_filtered': False,
              'time_note': 'UTC fields are remote host receipt times, not exposure times. CSV time remains relative to its first saved source frame.'}
    try:
        api_host = args.host.rsplit(":", 1)[0] if ":" in args.host else args.host
        api = core_client(api_host, args.api_port, connect_timeout_seconds=args.connect_timeout,
                          send_timeout_seconds=5)
        if not api.connected:
            raise RuntimeError(f"Cannot connect to Evoke at {api_host}:{args.api_port}. Open Evoke first.")
        status = playback_services(api)
        original = api_value(status.state())
        require_live(original)

        client = sdk.Client()
        connect_datastream(client, args.host, args.connect_timeout, sdk)
        client.EnableSegmentData()
        client.EnableUnlabeledMarkerData()
        axes = sdk.Client.AxisMapping
        client.SetAxisMapping(axes.EForward, axes.ELeft, axes.EUp)

        # Establish data readiness independently of marker visibility.
        client.SetBufferSize(1)
        client.SetStreamMode(sdk.Client.StreamMode.EClientPull)
        deadline = time.monotonic() + args.idle_timeout
        print("Connected to live Evoke. Waiting for camera data...", flush=True)
        while not client.GetFrame():
            if time.monotonic() >= deadline:
                raise RuntimeError("No live frames arrived. Check camera connections and Live mode in Evoke.")
            time.sleep(0.002)
        first_received_ns = time.time_ns()
        require_live(api_value(status.state()), original)
        prefetched = True
        timing['ready_utc_ns'] = first_received_ns
        timing['start_instruction_observed_utc_ns'] = timing['process_entry_utc_ns']
        # Preserve the very first received frame, even when tracking is blank.
        sequence = LiveFrames(client.GetFrameNumber()-1)
        client.SetBufferSize(1000)
        client.SetStreamMode(sdk.Client.StreamMode.EServerPush)

        started = None
        last_new_frame = time.monotonic()
        next_print = next_state_check = 0.0
        print("Press Ctrl+C to finish. Data will be saved as frames arrive.", flush=True)
        with ExitStack() as files:
            while True:
                now = time.monotonic()
                if started is not None and args.duration is not None and now - started >= args.duration:
                    print(f"Finished the requested {args.duration:g}-second live capture.", flush=True)
                    return 0
                if now >= next_state_check:
                    require_live(api_value(status.state()), original)
                    next_state_check = now + 0.25
                if now - last_new_frame >= args.idle_timeout:
                    raise RuntimeError("Live frames stopped arriving. Existing CSV rows have been saved.")
                if not prefetched and not client.GetFrame():
                    time.sleep(0.001)
                    continue
                received_ns = first_received_ns if prefetched else time.time_ns()
                prefetched = False
                frame = client.GetFrameNumber()
                rate = client.GetFrameRate()
                missing_before = sequence.missing
                if not sequence.accept(frame, rate):
                    time.sleep(0.001)
                    continue
                sample = read_positions(client)
                sample["received_unix_ns"] = received_ns
                last_new_frame = time.monotonic()
                if export is None:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    folder = args.output_dir / stamp
                    suffix = 2
                    while folder.exists():
                        folder = args.output_dir / f"{stamp}_{suffix}"
                        suffix += 1
                    export = files.enter_context(export_factory(folder, "LIVE_" + folder.name,
                                                          sequence.first, 1.0 / sequence.rate))
                    started = last_new_frame
                    timing.update(frame_rate_hz=rate, first_saved_frame=frame, first_saved_received_utc_ns=received_ns,
                        start_to_first_receive_s=(received_ns-timing['start_instruction_observed_utc_ns'])*1e-9)
                    (folder/'capture_timing.json').write_text(json.dumps(timing,indent=2),encoding='utf-8')
                    print(f"Recording at {rate:g} Hz. Saving to: {folder.resolve()}", flush=True)
                export.write(sample)
                if sequence.missing != missing_before:
                    print(f"Warning: missed {sequence.missing - missing_before} live frame(s); "
                          "source frame numbers and time gaps are preserved.", file=sys.stderr)
                if now >= next_print:
                    print(display(sample), flush=True)
                    next_print = now + 1.0 / args.hz
    except KeyboardInterrupt:
        print("\nStopped live capture.", flush=True)
        return 0
    finally:
        if client is not None:
            try:
                if client.IsConnected():
                    client.Disconnect()
            except sdk.DataStreamException:
                pass
        if api is not None:
            api.stop()
        if export is not None:
            print(f"Processed {export.frame_count} live frames; "
                  f"{sequence.missing} missing frame numbers within the captured span.", flush=True)
            summarize(export)
        else:
            print("No live frames were recorded; no CSV files were created.", flush=True)
