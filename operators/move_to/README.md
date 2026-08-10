# move_to operator — the positioner (visual servo)

Owns the leslider's **carriage**. The slider runs in **velocity mode**, so there
is no absolute slider position on the wire — placement is closed-loop from
vision: the robot carries an **ArUco marker** (4x4, id 10), and a PID loop drives
`slider.vel` until that marker sits on a target line in the overhead image.

```
overhead frame ─► ArUco detect ─► pixel error (target - marker) along AXIS
                                          │
                                          ▼
                                       PID loop ─► slider.vel ─► robot
```

`pos` is **0..100** across a calibrated **safe zone** — two lines in the image
perpendicular to the slider's travel. Set `AXIS` in `config.py` to `"vertical"`
or `"horizontal"` to match how the camera is mounted (marker moving up/down vs
left/right). Requests are capped to `[0, 100]`, so the servo is never asked to
drive past a bound. The candy shop works from two fixed stations, so the agent
just drives to two numbers (`POSITIONS` in `voice-agent`).

## Layout

| file | responsibility |
|------|----------------|
| `config.py`      | plain-constant tuning: PID gains, deadzone, max velocity, marker id, … |
| `vision.py`      | `ArucoDetector` (marker → pixel center) + `SafeZone` (pos 0..100 ↔ image-y) |
| `servo.py`       | `PID` + `SliderServo` — claims control, runs the ArUco→PID→`slider.vel` loop |
| `run.py`         | the operator entry point: wires the `move_to` RPC |
| `calibrate.py`   | 2-click safe-zone tool → writes `safe_zone.yaml` |
| `debug_move_to.py` | live tool: click a target, tune the PID by eye (includes its overlay) |

## RPC

### `move_to(payload) -> JSON` — the whole order path

- `payload` is a bare number (`"30"`) or `{"position": 30}`; 0..100 (capped).
- Servos the marker to the target line, then stops (`slider.vel = 0`).

```python
await room.local_participant.perform_rpc(
    destination_identity="move-to-operator", method="move_to", payload="30")
# -> {"requested": 30.0, "capped": false, "target_pos": 30.0, "reached": true,
#     "reason": "reached", "iterations": 52, "elapsed_s": 1.7, "final_pos": 30.4}
```

Errors: `1400` empty/non-numeric payload, `1409` no robot state yet.

Parking the rig is **not** here — the robot's own `reset_to_zero_position`
(`robot/run.py`) folds the arm and stops the slider, and doubles as the
preempt-anything cancel path.

## Setup

You need a robot process (`uv run robot`) joined to the same `LIVEKIT_ROOM`,
otherwise no camera frames or state arrive.

```bash
uv run python operators/move_to/calibrate.py   # once: click the pos 0 line then the pos 100 line
uv run move-to                                   # run the operator
```

Tune the PID by eye:

```bash
uv run python operators/move_to/debug_move_to.py
# click to set a target; [ ] tune kp, ; ' tune ki, , . tune kd; copy into config.py
```

## Tuning (`config.py`)

All knobs are plain constants in [`config.py`](config.py) (same style as
`voice-agent/config.py`) — edit and restart:

- `AXIS` — `"vertical"` or `"horizontal"` (re-run `calibrate.py` after changing)
- `PID_KP` / `PID_KI` / `PID_KD` / `PID_D_TAU` — PID gains (default `12.0 / 0.0 / 1.0 / 0.05`)
- `DEADZONE_PX` (default 12), `MAX_VELOCITY` (raw ticks/s, default 2000)
- `INVERT` — flip if the camera sees the rail's travel reversed
- `CONVERGE_TICKS` (default 5), `TIMEOUT_S` (default 20)
- `MARKER_DICT` (`DICT_4X4_50`), `MARKER_ID` (10)

`LIVEKIT_*` and `LIVEKIT_ROOM` still come from the environment (`.env`).
