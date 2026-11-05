"""JSON/ZeroMQ live frames. Only needed when record_evoke.py uses --publish."""

from pathlib import Path
import time
import uuid

import zmq


class FramePublisher:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.stream_id = str(uuid.uuid4())

    def __enter__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        try:
            self.socket.bind(self.endpoint)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self.endpoint = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Publishing live frames at {self.endpoint}", flush=True)
        return self

    def send(self, sample, first_frame, period_s):
        packet = {
            "schema": "evoke.frame.v1", "stream_id": self.stream_id,
            "frame": sample["frame"], "time_s": (sample["frame"] - first_frame) * period_s,
            "frame_rate_hz": 1.0 / period_s,
            "received_unix_ns": sample["received_unix_ns"],
            "sent_unix_ns": time.time_ns(),
            "axes": ["forward", "left", "up"], "position_units": "mm",
            "quaternion_order": "xyzw",
            "objects": {obj["name"]: {
                "segment": obj["root"], "present": obj["present"],
                "position_mm": obj["position"], "quaternion_xyzw": obj["quaternion"]
            } for obj in sample["objects"]},
            "markers": {str(marker_id): xyz for xyz, marker_id in sample["markers"]},
        }
        # A single JSON message is required for CONFLATE. No pickle or multipart frames.
        self.socket.send_json(packet, flags=zmq.NOBLOCK, allow_nan=False)

    def __exit__(self, *exception):
        self.socket.close(linger=0)
        self.context.term()


class StreamingExport:
    """Publish each frame, optionally forwarding it to the existing CSV writer."""
    def __init__(self, directory, capture, first_frame, period_s, *, publisher, csv_factory=None):
        self.directory = Path(directory)
        self.first_frame, self.period_s = first_frame, period_s
        self.publisher = publisher
        self.csv = csv_factory(directory, capture, first_frame, period_s) if csv_factory else None
        self.frame_count = 0

    def __enter__(self):
        if self.csv is not None:
            self.csv.__enter__()
        else:
            # Preserve the recorder's capture_timing.json, even without CSV recording.
            self.directory.mkdir(parents=True, exist_ok=False)
        return self

    def write(self, sample):
        if self.csv is not None:
            self.csv.write(sample)
        self.publisher.send(sample, self.first_frame, self.period_s)
        self.frame_count += 1

    def __exit__(self, *exception):
        if self.csv is not None:
            return self.csv.__exit__(*exception)

    def report(self, csv_report):
        print(f"Published {self.frame_count} frames (delivery to receivers is not acknowledged).", flush=True)
        if self.csv is not None:
            csv_report(self.csv)
