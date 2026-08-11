# move_to operator — the positioner (visual servo)

Owns the leslider's carriage. The slider runs in velocity mode, so placement is
closed-loop from vision: the robot carries an ArUco marker (4x4, id 10), and a
PID loop drives `slider.vel` (raw ticks/s) until that marker sits on a target
line in the overhead image.

`pos` is 0..100 across a calibrated safe zone (two lines perpendicular to the
slider's travel); requests are capped to `[0, 100]`. Set `AXIS` in `config.py`
to `"vertical"` or `"horizontal"` to match the camera mounting.

## Detection in poor light

Detection drops out — motion blur, glare, a shadowed stretch of rail. The fix
is local to this operator: nothing here reconfigures the camera, because the
robot publishes one frame stream to every operator and exposure is rig-wide
state this operator has no business setting.

`ArucoDetector` post-processes every frame before
detection: a gamma LUT lifts the shadows, then CLAHE equalizes local contrast,
then a wider and finer adaptive-threshold sweep than OpenCV's default (which
assumes an evenly lit marker). Measured against a synthetic tag with sensor
noise, detection rate at a given brightness:

| brightness | stock | + threshold sweep | + CLAHE | + gamma 0.6 |
|---|---|---|---|---|
| 7% | 0% | 95% | 100% | 100% |
| 5% | 0% | 0% | 100% | 100% |
| 2.5% | 0% | 0% | 100% | 100% |
| 2.0% | 0% | 0% | 75% | 95% |
| 1.5% | 0% | 0% | 0% | 25% |

Note what is *not* on that list: a plain brightness/gain lift. ArUco thresholds
each pixel against its local mean, so scaling the whole frame is very nearly a
no-op for it. Gamma is nonlinear and CLAHE is local, which is why they work where
a global stretch wouldn't — and it's also why the realistic case (rail lit at one
end, marker in shadow at the other) detects at 100% down past 1% brightness: the
shadowed tile gets equalized against its own neighborhood, not the bright end.

Costs ~0.5 ms/frame for the enhancement, ~1 ms total when the marker is found and
~13 ms when it isn't (the full sweep runs before giving up) — inside the 33 ms
tick, but the knobs are in `config.py` if you need the headroom back.

## Layout

| file | responsibility |
|------|----------------|
| `config.py`      | tuning constants: PID gains, deadzone, max velocity, marker id, … |
| `vision.py`      | `ArucoDetector` (marker → pixel center) + `SafeZone` (pos 0..100 ↔ image coord) |
| `servo.py`       | `PID` + `SliderServo` — claims control, runs the ArUco→PID→`slider.vel` loop |
| `run.py`         | operator entry point: wires the `move_to` RPC |
| `calibrate.py`   | 2-click safe-zone tool → writes `safe_zone.yaml` |
| `debug_move_to.py` | live tool: click a target, tune the PID by eye |

## RPC: `move_to(payload) -> JSON`

- `payload` is a bare number (`"30"`) or `{"position": 30}`; 0..100 (capped).
- Servos the marker to the target line, then stops (`slider.vel = 0`).

```python
await room.local_participant.perform_rpc(
    destination_identity="move-to-operator", method="move_to", payload="30")
# -> {"requested": 30.0, "capped": false, "target_pos": 30.0, "reached": true,
#     "reason": "reached", "iterations": 52, "elapsed_s": 1.7, "final_pos": 30.4}
```

Errors: `1400` empty/non-numeric payload, `1409` no robot state yet.

Parking the rig is not here — the robot's own `reset_to_zero_position`
(`robot/run.py`) folds the arm and stops the slider, and doubles as the
preempt-anything cancel path.

## Setup

A robot process (`uv run robot`) must be joined to the same `LIVEKIT_ROOM`,
otherwise no camera frames or state arrive.

```bash
uv run move-to-calibrate   # once: click the pos 0 line then the pos 100 line
uv run move-to             # run the operator
```

Tune the PID by eye:

```bash
uv run move-to-debug
# click to set a target; [ ] tune kp, ; ' tune ki, , . tune kd; copy into config.py
```

## Tuning (`config.py`)

Plain constants — edit and restart:

- `AXIS` — `"vertical"` or `"horizontal"` (re-run `calibrate.py` after changing)
- `PID_KP` / `PID_KI` / `PID_KD` / `PID_D_TAU` — PID gains (default `12.0 / 0.0 / 1.0 / 0.05`)
- `DEADZONE_PX` (default 12), `MAX_VELOCITY` (raw ticks/s, default 2000)
- `INVERT` — flip if the camera sees the rail's travel reversed
- `CONVERGE_TICKS` (default 5), `TIMEOUT_S` (default 20)
- `MARKER_DICT` (`DICT_4X4_50`), `MARKER_ID` (10)

Detection in poor light:

- `GAMMA` (0.6; 1.0 disables) — shadow lift before equalizing
- `CLAHE_ENABLED` / `CLAHE_CLIP_LIMIT` / `CLAHE_TILE_GRID`
- `ADAPTIVE_THRESH_WIN_MIN` / `_MAX` / `_STEP` — narrow the range or coarsen the
  step to buy back CPU; OpenCV's defaults are `3 / 23 / 10`
- `MIN_MARKER_PERIMETER_RATE` (0.01), `CORNER_REFINE_SUBPIX`

`LIVEKIT_*` and `LIVEKIT_ROOM` come from the environment (`.env`).
