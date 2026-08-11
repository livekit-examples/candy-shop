# move_to operator — the positioner (visual servo)

Owns the leslider's carriage. The slider runs in velocity mode, so placement is
closed-loop from vision: the robot carries an ArUco marker (4x4, id 10), and a
sqrt decel profile drives `slider.vel` (raw ticks/s) until that marker sits on a
target line in the overhead image.

`pos` is 0..100 across a calibrated safe zone (two lines perpendicular to the
slider's travel); requests are capped to `[0, 100]`. Set `AXIS` in `config.py`
to `"vertical"` or `"horizontal"` to match the camera mounting.

## Stream resolution

Everything that *locates* the tag is normalized: `safe_zone.yaml` stores `pos0`
and `pos100` as fractions of the axis, marker centers are divided by the frame
size, and the estimator's fitted gain is in axis fractions per tick/s. So a
resolution renegotiation mid-move (simulcast, bandwidth) leaves the target and
the measured position exactly where they were — provided the resize preserves
aspect ratio and field of view. A *crop* would invalidate the calibration; a
rescale doesn't.

The control law was a different story. It scaled the normalized error by the live
frame width to get `error_px`, so the loop gain and the deadzone rode the stream
resolution:

| stream width | effective `KP` (tuned 22) | deadzone |
|---|---|---|
| 960 | 33.0 | 1.39 pos |
| 640 | 22.0 | 2.08 pos |
| 320 | 11.0 | 4.17 pos |
| 160 | 5.5 | 8.33 pos |

A drop to 320 halved the gain and doubled the placement tolerance, silently. The
error is now scaled by `REFERENCE_AXIS_PX` — a fixed constant you set to the axis
length you tuned at — so "pixels" in the gains mean the same thing regardless of
what arrives. Verified: a move takes 1.90 s at 320, 1.90 s at 640, 1.97 s at 960,
and 1.90 s when the stream switches mid-flight.

One thing a resize genuinely does affect is detection, since
`ADAPTIVE_THRESH_WIN_*` are in real pixels — so a resolution change is logged.

## Losing sight of the marker

Detection drops out — motion blur, glare, a shadowed stretch of rail. Two
independent defenses, both local to this operator (nothing here reconfigures the
camera; the robot publishes one frame stream to every operator, so exposure is
rig-wide state this operator has no business setting):

**See it more often.** `ArucoDetector` post-processes every frame before
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

**Coast when you can't.** The marker is bolted to the carriage, so its image
motion along `AXIS` is `gain * slider.vel` — and `slider.vel` is on the wire as
state. `AxisEstimator` is a 1D Kalman filter using that as its control input, so
the estimate rides through dropouts on dead reckoning. `gain` is fitted online by
least squares over the intervals between accepted detections, so there is nothing
to calibrate by hand and it re-fits if the camera moves.

Two properties that matter for safety:

- Until the gain has enough evidence (`GAIN_MIN_SAMPLES` intervals of real
  motion), coasting is **disallowed** — a missing marker stops the slider, as
  before. The servo never drives on a guess it can't justify.
- Detections are gated at `EST_GATE_SIGMA`; a wild outlier is rejected rather
  than followed. Persistent disagreement (`EST_REACQUIRE_TICKS` in a row) means
  the *estimate* is wrong — someone shoved the carriage — so it hard-resets to
  the camera.

Coast error after 1 s of *fully black* frames measures under a pixel, against an
8 px deadzone. In simulation a 1-second total blackout mid-move costs 70 ms of
extra travel time and still reaches.

Only a newly arrived frame produces a measurement. This matters more than it
sounds: if the stream dies, the last frame keeps detecting perfectly, so a servo
that re-detects it would drive the rail blind while believing it had vision. That
failure showed up in testing as a 20 s runaway before the check went in.

**When coasting runs out.** `_recover` walks a ladder, cheapest and safest first:

1. *Never acquired yet* — just wait. Nothing is wrong; the marker may not be in
   frame yet. Ends at `TIMEOUT_S`.
2. *Observations stopped arriving* (`FRAME_STALE_S`) — stop and fail
   `no_frames` immediately. Blind motion with the robot or stream gone is the one
   thing we must not do.
3. *Hold still* for `REACQUIRE_HOLD_S`. Free and safe, and it fixes the most
   common cause: motion blur is self-curing once stopped, and the marker cannot
   wander off on its own.
4. *Creep blind* toward wherever the servo was last driving, hoping to carry the
   marker out of a shadow — `LOST_CREEP_ENABLED`, **off by default** because it
   is open-loop motion. Bounded by `LOST_CREEP_MAX_S` and by the dead-reckoned
   safe-zone edge: the estimate is too degraded to servo on but still good enough
   to know where the rail ends. Re-acquisition resumes the move normally.

Otherwise: stop, `reason: "lost"`, rather than burning the full `TIMEOUT_S`.

Optical flow was considered and rejected: it needs image gradients exactly as
much as ArUco needs corner contrast, so it adds nothing in the dark case, and
frame-to-frame flow drifts with no absolute reference where the encoder doesn't.

## Layout

| file | responsibility |
|------|----------------|
| `config.py`      | tuning constants: profile, deadzone, max velocity, marker id, … |
| `vision.py`      | `ArucoDetector` (marker → pixel center), `AxisEstimator` (ArUco + slider dead reckoning), `SafeZone` (pos 0..100 ↔ image coord) |
| `servo.py`       | `SliderServo` — claims control, runs the track→profile→`slider.vel` loop |
| `run.py`         | operator entry point: wires the `move_to` + `stop` RPCs |
| `calibrate.py`   | 2-click safe-zone tool → writes `safe_zone.yaml` |
| `debug_move_to.py` | live tool: click a target, tune the profile by eye |

## RPC: `move_to(payload) -> JSON`

- `payload` is a bare number (`"30"`) or `{"position": 30}`; 0..100 (capped).
- Servos the marker to the target line, then stops (`slider.vel = 0`).

```python
await room.local_participant.perform_rpc(
    destination_identity="move-to-operator", method="move_to", payload="30")
# -> {"requested": 30.0, "capped": false, "target_pos": 30.0, "reached": true,
#     "reason": "reached", "iterations": 52, "coasted_ticks": 3,
#     "creep_ticks": 0, "elapsed_s": 1.7, "final_pos": 30.4}
```

`reason`:

| value | meaning |
|---|---|
| `reached` | marker held inside the deadzone for `CONVERGE_TICKS` |
| `lost` | marker gone past the coast + re-acquire window; slider stopped |
| `no_frames` | observations stopped arriving; stopped without moving blind |
| `timeout` | `TIMEOUT_S` elapsed (also covers "never saw the marker at all") |
| `stopped` | a `stop` RPC arrived mid-move |

`coasted_ticks` counts ticks driven on dead reckoning rather than a live
detection; `creep_ticks` counts ticks driven fully blind. Persistently nonzero
means the lighting wants attention.

Errors: `1400` empty/non-numeric payload, `1409` no robot state yet.

## RPC: `stop() -> JSON`

Preempts the move in flight. The carriage is the positioner's to move, so halting
it belongs here rather than on the robot's heavier cancel path.

```python
await room.local_participant.perform_rpc(
    destination_identity="move-to-operator", method="stop", payload="")
# -> {"stopped": true}
```

Sets a flag the servo loop checks first on each tick, so the move exits through
the same path as a timeout or a converged move — carriage zeroed, control
released — and returns `reason: "stopped"` to whoever called `move_to`. Returns
immediately; it does not wait for the loop to unwind.

Safe to send when nothing is moving: the flag is cleared on entry to each
`move_to`, so a stop that lands between two moves can't abort the next one. That
matters because the voice agent issues back-to-back `move_to` calls.

Parking the rig is still not here — the robot's own `reset_to_zero_position`
(`robot/run.py`) folds the arm *and* stops the slider, and doubles as the
preempt-anything cancel path. Use `stop` to halt the travel and leave the arm
where it is; use the robot's reset to put everything back at rest.

## Setup

A robot process (`uv run robot`) must be joined to the same `LIVEKIT_ROOM`,
otherwise no camera frames or state arrive.

```bash
uv run move-to-calibrate   # once: click the pos 0 line then the pos 100 line
uv run move-to             # run the operator
```

Tune the approach by eye:

```bash
uv run move-to-debug
# click to set a target; [ ] approach px, ; ' min velocity, , . deadzone
# watch the last inch for hunting; copy what you settle on into config.py
```

## Tuning (`config.py`)

Plain constants — edit and restart:

- `AXIS` — `"vertical"` or `"horizontal"` (re-run `calibrate.py` after changing)
- `REFERENCE_AXIS_PX` (640) — the axis length the px below are expressed in; set
  it to what you calibrated at, and the tuning survives a stream resize
- `DEADZONE_PX` (default 8, in reference px) — also *is* your placement accuracy,
  since nothing closes out the residual
- `MIN_VELOCITY` (250) — breakaway floor outside the deadzone; below it the
  carriage buzzes instead of moving, which reads as sluggish
- `APPROACH_FULL_SPEED_PX` (60) — where the profile hits full speed; see below
- `MAX_VELOCITY` (raw ticks/s, default 2800 against a 3000 hardware limit)
- `INVERT` — flip if the camera sees the rail's travel reversed
- `CONVERGE_TICKS` (default 3), `TIMEOUT_S` (default 20)
- `MARKER_DICT` (`DICT_4X4_50`), `MARKER_ID` (10)

### The control law

Not a PID — there used to be one, and instrumenting the loop showed it set the
command on **0% of ticks**: the profile below dominated everywhere between the
deadzone and saturation, so `KP`/`KD` were inert. It's gone.

What runs is memoryless — velocity is a function of the current error and
nothing else:

```
|v| = clamp(MAX_VELOCITY * sqrt(|e| / APPROACH_FULL_SPEED_PX),
            MIN_VELOCITY, MAX_VELOCITY)     for |e| > DEADZONE_PX
|v| = 0                                     otherwise
```

That's the profile a motion controller uses under an acceleration limit, and it
self-damps: velocity falling with distance *is* the braking, which is why no
derivative term is missed. Plain proportional control instead tapers linearly and
crawls the last stretch — it asks for under a quarter of max speed at 30 px out
and under a tenth at 12.

Two consequences worth holding onto:

- **No integrator, so the deadzone residual is permanent.** `DEADZONE_PX` is
  your placement spec, not a convergence detail. Shrink it for tighter
  placement (subpixel corner refinement supports going well below 8).
- **Effective stiffness `dv/de` rises as the error shrinks** — 1.1x `KP`-
  equivalent at 60 px, 2.9x at the deadzone edge, unbounded as `e → 0`. High
  stiffness plus transport delay is the classic hunting recipe, so the last inch
  is where a too-low `APPROACH_FULL_SPEED_PX` bites.

Simulated against a slider with 100 ms of first-order lag, the knob trades
arrival speed against overshoot:

| `APPROACH_FULL_SPEED_PX` | long move | short move | overshoot | hunting |
|---|---|---|---|---|
| 120 | 1.34 s | 0.47 s | 0.00 | no |
| **60** (default) | **1.17 s** | **0.37 s** | 0.71 pos | no |
| 30 | 1.13 s | 0.34 s | 1.06 pos | no |
| 15 | 1.14 s | 0.34 s | 1.65 pos | no |

Below 30 buys nothing and just arrives hotter. Lower it and the slider has to
shed more speed inside `DEADZONE_PX`; past what it can absorb it will overshoot
and hunt, which no simulation here can predict for your hardware — the lag figure
above is a guess. Tune it on the rig. Values at or below 1 px degrade to
bang-bang (full speed right up to the deadzone), which is almost certainly worse.

Other ways to speed up the endgame, if the profile isn't enough: raise
`MIN_VELOCITY` (blunter — a flat floor rather than a distance-aware one), or drop
`CONVERGE_TICKS`, which is pure dwell at the end (3 ticks = 100 ms).

Detection in poor light:

- `GAMMA` (0.6; 1.0 disables) — shadow lift before equalizing
- `CLAHE_ENABLED` / `CLAHE_CLIP_LIMIT` / `CLAHE_TILE_GRID`
- `ADAPTIVE_THRESH_WIN_MIN` / `_MAX` / `_STEP` — narrow the range or coarsen the
  step to buy back CPU; OpenCV's defaults are `3 / 23 / 10`
- `MIN_MARKER_PERIMETER_RATE` (0.01), `CORNER_REFINE_SUBPIX`

Recovery once coasting expires:

- `REACQUIRE_HOLD_S` (1.0) — hold still and wait before anything else
- `LOST_CREEP_ENABLED` (off), `LOST_CREEP_VELOCITY` (400), `LOST_CREEP_MAX_S` (2.0)
- `FRAME_STALE_S` (1.0) — no observations for this long fails `no_frames`

Estimator (lengths are fractions of the axis, not pixels):

- `MAX_COAST_S` (1.5) — dead-reckon this long before reporting `lost`
- `EST_MEAS_STD` (0.004), `EST_DRIFT_STD_PER_S` (0.02) — measurement vs. coast trust
- `EST_GATE_SIGMA` (5.0), `EST_REACQUIRE_TICKS` (15) — outlier rejection
- `GAIN_FORGET` (0.99), `GAIN_MIN_SAMPLES` (30), `GAIN_MIN_TICKS` (5.0) — the
  online slider-gain fit

`LIVEKIT_*` and `LIVEKIT_ROOM` come from the environment (`.env`).
