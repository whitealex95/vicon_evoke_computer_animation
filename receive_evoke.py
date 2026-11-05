"""Receive live student motion on another computer. Requires Python, numpy, and pyzmq; no Vicon SDK.

    python receive_evoke.py --connect tcp://EVOKE_PC_IP:5556
    python receive_evoke.py --connect tcp://EVOKE_PC_IP:5556 --callback my_animation:on_frame

on_frame(frame) receives a dictionary containing NumPy float32 pose arrays. Use course_frame(frame) for the notebook's six-tracker ordering.
Synchronize both PCs' clocks for the default 0.5-second age check. Timestamps
measure receipt at the Evoke PC, not camera exposure time. A timeout stops the
receiver instead of repeatedly using the last observation.
"""

import argparse
import importlib
import math
import sys
import time

import numpy as np
import zmq


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


class EvokeReceiver:
    """Single-threaded subscriber; call receive() for a fresh latest observation."""
    def __init__(self, endpoint, max_age=0.5):
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("max_age must be finite and positive")
        self.endpoint, self.max_age = endpoint, max_age
        self.last_stream = None
        self.last_frame = None

    def __enter__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.MAXMSGSIZE, 16 * 1024 * 1024)
        try:
            self.socket.connect(self.endpoint)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def receive(self, timeout=2.0):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.socket.poll(max(1, math.ceil(remaining * 1000))):
                raise TimeoutError("No fresh Evoke frames. Check sender, network, and synchronized PC clocks.")
            packet = self.socket.recv_json()
            if packet.get("schema") != "evoke.frame.v1":
                raise ValueError("Unsupported Evoke message schema")
            age = (time.time_ns() - packet["received_unix_ns"]) * 1e-9
            if age < -self.max_age:
                raise RuntimeError("Evoke timestamps are in the future. Synchronize both PCs' clocks.")
            if age > self.max_age:
                continue
            stream, frame = packet["stream_id"], packet["frame"]
            if stream == self.last_stream and frame <= self.last_frame:
                continue
            packet = numpy_poses(packet)
            self.last_stream, self.last_frame = stream, frame
            return packet

    def __exit__(self, *exception):
        self.socket.close(linger=0)
        self.context.term()


def numpy_poses(frame):
    """Convert decoded wire lists into motion arrays, preserving missing values."""
    def vector(value, length):
        if value is None:
            return None
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (length,) or not np.isfinite(result).all():
            raise ValueError(f"Expected a finite pose vector of shape ({length},)")
        return result

    for obj in frame["objects"].values():
        xyz = obj["position_mm"] = vector(obj["position_mm"], 3)
        quat = obj["quaternion_xyzw"] = vector(obj["quaternion_xyzw"], 4)
        obj["pose_mm_xyzw"] = np.concatenate((xyz, quat)) if obj["present"] and xyz is not None and quat is not None else None
    frame["markers"] = {key: vector(xyz, 3) for key, xyz in frame["markers"].items()}
    return frame


def course_frame(frame, character="Character1"):
    """Return one (6, 4) course-format pose in mm, including occlusion flags.

    Live transport delivers latest frames and is not uniformly sampled. Use the
    recorder's class export for the notebook's 120 Hz offline playback.
    """
    from view_evoke import TRACKER_SUFFIXES
    result = np.zeros((6, 4), dtype=np.float32)
    result[:, 3] = 1
    for index, suffix in enumerate(TRACKER_SUFFIXES):
        obj = frame["objects"].get(f"{character}_{suffix}")
        if obj and obj["present"] and obj["position_mm"] is not None:
            result[index, :3] = obj["position_mm"]
            result[index, 3] = 0
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--connect", required=True, help="tcp://EVOKE_PC_IP:5556")
    parser.add_argument("--timeout", type=positive, default=2.0)
    parser.add_argument("--max-age", type=positive, default=0.5, help="maximum age since source-PC receipt, seconds")
    parser.add_argument("--callback", help="Python module:function called with each fresh frame")
    args = parser.parse_args(argv)
    try:
        callback = None
        if args.callback:
            module, name = args.callback.split(":", 1)
            callback = getattr(importlib.import_module(module), name)
            if not callable(callback):
                raise ValueError("Callback entry point is not callable")
        with EvokeReceiver(args.connect, args.max_age) as receiver:
            while True:
                frame = receiver.receive(args.timeout)
                if callback:
                    callback(frame)
                else:
                    print(f"Frame {frame['frame']} | objects={list(frame['objects'])} | trajectories={list(frame['markers'])}", flush=True)
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"Receiver stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
