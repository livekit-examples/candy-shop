# policy operator — the picker

Serves the manipulation half of a candy-shop order with a **SmolVLA** vision-
language-action policy: given a natural-language instruction, it drives the
**arm** to pick while the slider is held still (the `move_to` operator parks the
carriage).

## Layout

| file | responsibility |
|------|----------------|
| `smolvla.py` | shared wiring: checkpoints, the drop-`slider.vel` dataset, image-key → camera mapping |
| `settle.py`   | `SettleGate` — wait for the arm to reach the last command before re-inferencing |
| `train.py`    | fine-tune SmolVLA on a leslider dataset (`policy-train`) |
| `run.py`      | operator entry point: loads a checkpoint and serves the `run_policy` RPC (`policy`) |
| `debug_policy.py` | interactive terminal driver: start/stop the policy and retype the prompt (`policy-debug`) |

## Fine-tune first

`lerobot/smolvla_base` is a *training* start, not something to serve: it ships no
normalization stats, and its image keys are placeholders (`camera1..3`). So
there is no zero-shot path — train on a rig dataset first (below), then serve
the result. `--checkpoint` is therefore **required** (or `POLICY_CHECKPOINT` in
the environment); `policy-train` writes the checkpoint to serve under
`outputs/smolvla-candy/pretrained_model`.

## Why `slider.vel` is dropped

The policy drives the arm only — the `move_to` operator owns the slider — but the
leslider wire contract carries a 7th `slider.vel` field (see
`shared/rest_pose.py`). So it is dropped end to end: `SliderDroppedDataset`
slices the column out of `observation.state`/`action` for training; at inference
the operator feeds the six arm `.pos` and pins `slider.vel = 0`.

## Inference (`run_policy`)

```python
result = await room.local_participant.perform_rpc(
    destination_identity="policy-operator", method="run_policy",
    payload="pick up the red candy")   # bare string or {"task": "..."}
# -> {"task": "pick up the red candy", "reason": "duration", "ticks": 900, "elapsed_s": 30.0}
```

Runs until `duration` elapses (default: forever, until a `stop` RPC preempts it),
then releases active control. Errors: `1409` if no robot state/frames have
arrived, `1400` on a malformed payload.

Cameras are matched onto the checkpoint's image keys **by name** — a fine-tune
keeps the dataset's own keys, so `observation.images.overhead_camera` gets the
overhead frame. That matters: the dataset lists its cameras alphabetically (arm
before overhead) and SmolVLA has no image-key override, so a positional map would
feed the wrist view in as the overhead one. Keys naming no wired camera fall back
to position (primary first, wrist second) with a warning.

```bash
uv run policy --checkpoint outputs/smolvla-candy/pretrained_model   # your fine-tune
uv run policy --checkpoint <user>/smolvla-candy                     # or one on the Hub
uv run policy --checkpoint ... --task "pick up the blue candy" --duration 20
uv run policy --checkpoint ... --num-steps 4    # fewer denoising steps = faster replans
```

## Debugging by hand (`policy-debug`)

Loads the same checkpoint into its own operator (`policy-debug`) and drives it
from the terminal — no `run_policy` RPC, no reward operator, no voice agent.
Takes the same flags as `policy`; `--duration` defaults to "run until stopped".

```bash
uv run policy-debug --checkpoint outputs/smolvla-candy/pretrained_model
```

```
policy> start pick up the red candy   # start (retyping the instruction first)
policy> prompt pick up the blue one   # swap it mid-run: refolds, then plans on it
policy> stop                          # preempt, release active control
policy> <enter>                       # toggle start/stop
policy> status | quit
```

Only run **one** thing that takes active-operator control at a time: don't run
`uv run policy` against the same room while debugging.

**Settle gate:** SmolVLA emits `n_action_steps` chunks (50 by default, ~1.7 s at
30 fps) — `select_action` pops one step per tick and only runs the model when the
chunk drains. The gate sits at that **replan boundary**: a fresh chunk is planned
from a single observation, so the arm (and the camera frame paired with it) must
have caught up to the last command first — within `--settle-tolerance` per joint,
or once the arm stops moving, or after `--settle-timeout` seconds. The mid-chunk
pops in between are not gated and execute at `fps`. Set `--settle-tolerance 0` to
replan the instant the chunk drains.

**Start pose:** every plan starts from the folded `RESET_POSE_DEFAULTS`
(`shared/rest_pose.py`), which is where all the recorded episodes start — elbow
closed around 90. It is not optional, because a chunk is conditioned on whatever
pose it plans from: the rig can come to rest with the elbow standing up near 20,
tens of units outside anything in the dataset, and the policy is well out of
distribution there. So the fold happens both at the top of a pick **and again on
a prompt swap** — `set_task` mid-run drops the chunk buffered for the old
instruction and refolds before planning the new one, rather than starting it from
wherever the previous instruction left the arm. `--start-ramp` (2 s) sets how
gently it eases in, since the target can be 60+ units away, and
`--start-tolerance` (3.0) how close counts as arrived.

## Training

Point it at a LeRobot dataset recorded on this rig. It fine-tunes from
`lerobot/smolvla_base` and writes `outputs/smolvla-candy/pretrained_model`, which
`run.py` serves directly (the checkpoint carries its own normalization stats).
Requires an NVIDIA GPU; `skypilot.yaml` runs it on one H100.

```bash
uv run policy-train --dataset <user>/candy_shop --dataset-root data/candy_shop

uv run policy-train ... --train-vlm                 # adapt the VLM too, not just the expert
uv run policy-train ... --unfreeze-vision-encoder   # ...and SigLIP on top of that
```

By default only the flow-matching action expert trains: the VLM and its vision
encoder stay frozen, which is both the cheapest fine-tune and the safest on a
dataset of a few hundred episodes — a frozen backbone cannot drift. `--train-vlm`
multiplies the trainable parameter count by roughly four and is worth trying only
once the dataset is large enough to support it.

Grow the effective batch with `--grad-accum` when VRAM runs out rather than
shrinking `--steps` — it trades wall-clock for memory at identical gradients.

Useful knobs: `--steps` (10 000, optimizer steps), `--batch-size` (8),
`--grad-accum` (1), `--chunk-size` (50, the action horizon and `n_action_steps`),
`--task` (fallback instruction if the dataset has no language column),
`--checkpoint scratch` (train a fresh action expert on the SmolVLM2 backbone
instead of fine-tuning `smolvla_base`).

## Setup

SmolVLA needs lerobot's `smolvla` extra (`transformers`, `num2words`,
`accelerate`), which is already in this project's dependencies:

```bash
uv sync
```

You also need a robot process (`uv run robot`) and, for a real order, a
`move-to` operator in the same `LIVEKIT_ROOM`.

## Env knobs

`POLICY_CHECKPOINT` (no default — required unless `--checkpoint` is passed),
`POLICY_TASK`, `POLICY_DEVICE`, `POLICY_DURATION_S`,
`POLICY_NUM_STEPS` (`0` = the checkpoint's own value, 10), `POLICY_PRIMARY_CAMERA`
(`overhead_camera`), `POLICY_WRIST_CAMERA` (`arm_camera`),
`POLICY_SETTLE_TOLERANCE` (`2.0`), `POLICY_SETTLE_TIMEOUT_S` (`2.0`),
`POLICY_START_RAMP_S` (`2.0`),
`POLICY_START_TOLERANCE` (`3.0`), plus the shared `LIVEKIT_*`.
