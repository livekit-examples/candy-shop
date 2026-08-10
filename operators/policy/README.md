# policy operator — the picker

Serves the manipulation half of a candy-shop order with a **MolmoAct2** vision-
language-action policy: given a natural-language instruction, it drives the
**arm** to pick while the slider is held still (the `move_to` operator parks the
carriage).

## Layout

| file | responsibility |
|------|----------------|
| `molmoact.py` | shared wiring: default checkpoint, the drop-`slider.vel` dataset, image-key resolution |
| `settle.py`   | `SettleGate` — wait for the arm to reach the last command before re-inferencing |
| `train.py`    | fine-tune MolmoAct2 on a leslider dataset (`policy-train`) |
| `run.py`      | operator entry point: loads a checkpoint and serves the `run_policy` RPC (`policy`) |

## Why `slider.vel` is dropped

MolmoAct2's SO-101 checkpoint is a **six-DOF arm** with no slider, but the
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

Runs until `duration` elapses (default 30 s) or a `stop` RPC preempts it, then
releases active control. Errors: `1409` if no robot state/frames have arrived,
`1400` on a malformed payload. Cameras map onto the checkpoint's image keys in
order: `overhead_camera` → primary, `arm_camera` → wrist.

```bash
uv run policy                                              # default SO-101 checkpoint, zero-shot
uv run policy --checkpoint outputs/molmoact2-candy/pretrained_model   # your fine-tune
uv run policy --task "pick up the blue candy" --duration 20
```

**Settle gate:** before each inference the operator waits until the observed arm
reaches the last command (within `--settle-tolerance` per joint, giving up after
`--settle-timeout` seconds), sending nothing meanwhile. Set `--settle-tolerance 0`
to disable it and stream open-loop at `fps`.

## Training

Point it at a LeRobot dataset recorded on this rig. It fine-tunes from the
default SO-101 checkpoint and writes `outputs/molmoact2-candy/pretrained_model`,
which `run.py` serves directly (the checkpoint carries its own normalization
stats). Requires an NVIDIA GPU.

```bash
uv run policy-train --dataset <user>/candy_shop --dataset-root data/candy_shop

# memory-saving modes (see the MolmoAct2 docs' VRAM table):
uv run policy-train ... --train-action-expert-only   # cheapest
uv run policy-train ... --lora                        # LoRA on the VLM
uv run policy-train ... --gradient-checkpointing
```

Useful knobs: `--steps` (10 000), `--batch-size` (8), `--chunk-size` (30, the
action horizon), `--action-mode` (`both`), `--model-dtype` (`bfloat16`),
`--task` (fallback instruction if the dataset has no language column),
`--setup-type`/`--control-mode` (prompt text describing the robot and action
space).

## Setup

MolmoAct2 needs a newer LeRobot than the pinned `0.6.0` plus its extras
(`transformers`, `peft`):

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
