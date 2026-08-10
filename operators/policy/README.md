# policy operator — the picker

Serves the manipulation half of a candy-shop order with a **MolmoAct2** vision-
language-action policy. The `move_to` operator parks the carriage at a station;
this operator then drives the **arm** to pick, conditioned on a natural-language
instruction. The slider is held still the whole time.

## Layout

| file | responsibility |
|------|----------------|
| `molmoact.py` | shared wiring: default checkpoint, the drop-`slider.vel` dataset, image-key resolution |
| `settle.py`   | `SettleGate` — wait for the arm to reach the last command before re-inferencing |
| `train.py`    | fine-tune MolmoAct2 on a leslider dataset (`policy-train`) |
| `run.py`      | the operator entry point: loads a checkpoint and serves the `pick` RPC (`policy`) |

## Why `slider.vel` is ignored

The leslider wire contract is 7 fields — six arm `.pos` + one `slider.vel` (see
`utilities/rest_pose.py`). MolmoAct2's SO-101 checkpoint is a **six-DOF arm**; it
has no slider. So the 7th field is dropped end to end:

- **training** — `SliderDroppedDataset` slices the `slider.vel` column out of
  `observation.state` and `action` (samples, feature metadata, and quantile
  stats) before the policy sees them.
- **inference** — the operator feeds the six arm `.pos` as state and pins
  `slider.vel = 0` on every action it sends.

## Inference (`pick`)

```python
result = await room.local_participant.perform_rpc(
    destination_identity="policy-operator", method="pick",
    payload="pick up the red candy")   # bare string or {"task": "..."}
# -> {"task": "pick up the red candy", "reason": "duration", "ticks": 900, "elapsed_s": 30.0}
```

Each tick the operator feeds both camera frames + the six arm `.pos` into the
policy and streams the predicted arm action back, holding the slider at 0. Runs
until `duration` elapses (default 30 s) or a `stop` RPC preempts it, then
releases active control. `stop` cancels a running pick. Errors: `1409` if no
robot state/frames have arrived, `1400` on a malformed payload.

Cameras are mapped onto the checkpoint's expected image keys in order:
`overhead_camera` → primary (external), `arm_camera` → wrist.

### Settle gate

MolmoAct2 predicts an action chunk from a single frame, so it should observe a
*stationary* arm. Before each inference the operator waits until the observed
arm reaches the last action it commanded — within `--settle-tolerance` per
joint, giving up after `--settle-timeout` seconds. During the wait it sends
nothing, so the robot keeps driving toward that last target. Set
`--settle-tolerance 0` to disable the gate (stream open-loop at `fps`).

```bash
uv run policy                                              # default SO-101 checkpoint, zero-shot
uv run policy --checkpoint outputs/molmoact2-candy/pretrained_model   # your fine-tune
uv run policy --task "pick up the blue candy" --duration 20
```

## Training

Point it at a LeRobot dataset recorded on this rig. It fine-tunes from the
default SO-101 checkpoint and writes `outputs/molmoact2-candy/pretrained_model`,
which `run.py` can serve directly (the checkpoint carries its own normalization
stats via the saved processors).

```bash
uv run policy-train --dataset <user>/candy_shop --dataset-root data/candy_shop
```

Memory-saving fine-tune modes (see the MolmoAct2 docs' VRAM table):

```bash
uv run policy-train ... --train-action-expert-only   # cheapest
uv run policy-train ... --lora                        # LoRA on the VLM
uv run policy-train ... --gradient-checkpointing
```

Useful knobs: `--steps` (10 000), `--batch-size` (8), `--chunk-size` (30, the
action horizon), `--action-mode` (`both`), `--model-dtype` (`bfloat16`),
`--task` (fallback instruction if the dataset has no language column),
`--setup-type`/`--control-mode` (prompt text describing the robot and action
space). Requires an NVIDIA GPU.

## Default checkpoint

`lerobot/MolmoAct2-SO100_101-LeRobot` — the LeRobot-format SO-100/101 checkpoint
(six-DOF, two cameras `cam0`/`cam1`, ships the SO-100/101 joint-frame
correction). Downloaded automatically on first run.

## Setup

MolmoAct2 needs a newer LeRobot than the pinned `0.6.0` **and** its extra
dependencies (`transformers`, `peft`):

```bash
uv sync --extra molmoact2   # after pointing lerobot at a build that ships policies/molmoact2 (>= 0.6.1)
```

You also need a robot process (`uv run robot`) and, for a real order, a
`move-to` operator in the same `LIVEKIT_ROOM`.

## Env knobs

`POLICY_CHECKPOINT`, `POLICY_TASK`, `POLICY_DEVICE`, `POLICY_DURATION_S`,
`POLICY_INFERENCE_ACTION_MODE` (`continuous`), `POLICY_PRIMARY_CAMERA`
(`overhead_camera`), `POLICY_WRIST_CAMERA` (`arm_camera`),
`POLICY_SETTLE_TOLERANCE` (`2.0`), `POLICY_SETTLE_TIMEOUT_S` (`2.0`), plus the
shared `LIVEKIT_*`.
