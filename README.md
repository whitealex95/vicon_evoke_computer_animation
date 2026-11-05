# Student motion capture for Computer Animation

Record students from Vicon Evoke's live DataStream, optionally stream their poses
to another computer, and play recordings locally. Every finished capture attempts
a separate export for Section 3.3 (MoCap / Retargeting) of `P5_Twister.ipynb`.

## Setup

Use the existing Python 3.11 Vicon environment on the capture computer:

```powershell
conda activate vicon
python -m pip install -r requirements.txt
```

Capture requires the installed Vicon DataStream Python SDK (`vicon_dssdk`) and
Evoke's Python API. The recorder finds the API shipped under
`C:\Program Files\Vicon\Evoke*\SDK\Python` when needed. Open Evoke in Live mode,
with cameras tracking. Enable Include Unlabeled Reconstructions if those markers
are needed. The default DataStream address is `localhost:801`; the Evoke API port
is `52800`. Use `--host ADDRESS:PORT` / `--api-port PORT` to override.

For the course export, configure these six tracked smart objects in Evoke:

| Array index | Notebook key | Evoke object |
| --- | --- | --- |
| 0 | spine | Character1_Spine |
| 1 | r_foot | Character1_RightFoot |
| 2 | r_hand | Character1_RightHand |
| 3 | l_foot | Character1_LeftFoot |
| 4 | head | Character1_Head |
| 5 | l_hand | Character1_LeftHand |

Use `--character NAME` if the prefix differs. All six must be visible together at
least once. A generic `Character1` root is not a substitute for its Spine tracker.

## Record

```powershell
python record_evoke.py --duration 15
# Or capture until Ctrl+C:
python record_evoke.py
```

Each take creates `record_exports/TIMESTAMP/`:

- `all_objects.csv`: full local recording, every received frame, source frame
  numbers and relative times, all object positions and XYZW orientations, and
  unlabeled markers. Blank fields mean unavailable measurements.
- `capture_timing.json`: capture start receipt times and source frame rate.
- `session.json`: capture rate, missing-frame count, duration, and class-export
  status (including a reason if it could not be made).
- `class/motion.npy`: course array, float32 `(frames, 6, 4)`, with
  `[X_mm, Y_mm, Z_mm, occluded_flag]` in the order above. Flag 0 is visible;
  flag 1 is missing, with zero XYZ placeholders.
- `class/motion.json`: 120 Hz timing, source rate, trimmed lead-in, tracker order,
  and occlusion counts. Keep it beside the NPY for automatic local playback rate.

Class export trims the lead-in before the first complete six-tracker pose and
resamples to 120 Hz using linear interpolation between consecutive visible source
frames. It does not interpolate over dropped source frames or occluded endpoints.
Those samples remain flagged; the course notebook forward-fills them. This can
hold a pose through a long gap, so review tracking quality before distributing a
take. Nothing is centered or scaled: the notebook applies its own `1.2 / 1000`
scale. Axes are X forward, Y left, Z up. The full local CSV retains original timing
and the lead-in. Conversion loads the CSV into RAM after capture ends.

If the six trackers cannot form a complete pose, the local capture is still saved
and `session.json` explains why no class file was created. Record about 15 seconds
with all trackers visible to supply the notebook's usual first 10 seconds.

Frames are journaled during recording. Ctrl+C, normal completion, and handled
capture failures finalize the data. After a forced process termination, recover:

```powershell
python record_evoke.py --recover record_exports/TIMESTAMP/frames.jsonl
```

## Play recorded motion

```powershell
python view_evoke.py --latest
python view_evoke.py record_exports/TIMESTAMP
python view_evoke.py record_exports/TIMESTAMP/class/motion.npy
```

Play/Pause, sample seeking, playback speed, mouse rotation, and toolbar zoom are
available. Space toggles playback; arrow keys pause and step. CSV playback follows
source timestamps, including gaps, and displays orientations and markers. The
skeleton connects Spine to Head, both hands, and both feet; it does not infer knees
or elbows. Full character retargeting happens in the course notebook.

Playback needs NumPy and Matplotlib, without Evoke or the Vicon SDK. A standalone
NPY without its metadata requires an explicit rate, e.g. `--fps 120` for class
exports. Use `--snapshot preview.png` for a headless image.

## Use the course notebook

`reference/P5_Twister.ipynb` is byte-identical to the notebook in the supplied
Downloads ZIP. `P5_Twister_local.ipynb` is a separate local copy with only the
motion-file selection and short-capture frame limits adjusted. Course exercises
and retargeting logic remain unchanged; complete the required course setup and IK
exercises as usual. The notebook's separate course dependencies are not included
in this recorder's requirements file.

Open `P5_Twister_local.ipynb` with this project as its working directory. In
Section 3.3.1, leave `selected_mocap = "latest"` or choose a capture folder listed
by the preceding cell. If the notebook runs elsewhere, change `MOCAP_ROOT` to this
project's absolute `record_exports` path.

To distribute to the original course environment, copy the take's
`class/motion.npy` to `/voc/data/mocap_data/STUDENT_ID.npy` and set
`selected_mocap = "STUDENT_ID"`. The original notebook expects 120 Hz and up to
1200 source frames. For shorter takes, use these limits in its preprocessing and
IK setup cells:

```python
end_frame_idx = min(1200, len(mocap_frames))
# After slicing target_mocap_frames:
total_render_frames = len(target_mocap_frames)
max_frames = min(300, len(target_mocap_frames))
```

To convert an existing CSV explicitly:

```powershell
python export_evoke_npy.py record_exports/TIMESTAMP --class-ready
```

Without `--class-ready`, this exporter retains the original sample rows in
`motion.npy`; that legacy export has no timing metadata or 120 Hz resampling.
Existing output files are protected from replacement.

The preserved take `20261105_165228` is 1.41 seconds at 100 Hz and has no valid
Spine or Head data. It can demonstrate local playback but cannot supply a valid
six-tracker retargeting take. Its older arrays remain preserved as legacy data.

## Stream live motion

On the Evoke computer, record and publish together:

```powershell
python record_evoke.py --publish tcp://0.0.0.0:5556
```

On the receiving computer (NumPy and pyzmq required):

```powershell
python receive_evoke.py --connect tcp://EVOKE_PC_IP:5556
python receive_evoke.py --connect tcp://EVOKE_PC_IP:5556 --callback my_animation:on_frame
```

The callback receives a dictionary with source frame number, relative time,
source rate, named positions in mm, XYZW rotations, and markers. Call
`receive_evoke.course_frame(frame)` for a float32 `(6, 4)` pose in course order;
pass `character="NAME"` for another prefix. Live packets are latest-only and may
skip frames. Use the sender's saved class export for uniformly sampled notebook
motion. Synchronize the computers' clocks for the receiver's freshness check;
allow the chosen TCP port through the local firewall when streaming between PCs.

`--stream-only --publish ENDPOINT` disables recording; only capture-start metadata
is retained. The receiver exits on a stream timeout rather than replaying a stale
pose indefinitely.

## Verify offline

```powershell
python -m unittest discover -v
```

Tests cover capture finalization and recovery, tracker order and units, 120 Hz
resampling, tracking loss and frame gaps, class export validation, playback, and a
local ZeroMQ sender/receiver connection. Actual camera capture must be checked in
the lab with Evoke and all six trackers active.

Duplicate transfer scripts and the old synthetic preview are retained in
`.archive/`. The active scripts live only at the project root.
