"""Score a checkpoint on the pre-registered holdout, in joint units.

Flow-matching `eval_loss` ranks checkpoints within a run but says nothing legible about
behaviour, and it is not comparable across runs whose eval splits differ. This measures
the thing the arm actually does: from a held-out observation, predict the action chunk
the policy would execute and compare it to what the teleoperator actually did, in the
same units the joints are commanded in.

It is still an open-loop proxy -- it never closes the loop, so it cannot see a policy
recover from its own drift, and it rewards imitating the demonstration rather than
succeeding at the task. The blog was explicit that only rollouts settle that. What this
does give is a single common yardstick across every candidate, which `eval_loss` could
not.

    python -m operators.policy.eval_holdout \\
        --checkpoint /outputs/pi05-stage2-hq60/checkpoints/000100/pretrained_model \\
        --holdout-root ~/data/candy-holdout-rel
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import get_policy_class
from lerobot.utils.constants import ACTION

# Horizons reported separately: a policy can be accurate at the first step and drift
# badly by the 50th, and with a 50-step chunk at 30 fps the tail is 1.7 s of open-loop
# motion. One averaged number would hide that.
HORIZONS = (1, 10, 25, 50)


def load(checkpoint: str, device: str, num_steps: int):
    config_path = pathlib.Path(checkpoint) / "config.json"
    policy_type = json.loads(config_path.read_text())["type"]
    policy = get_policy_class(policy_type).from_pretrained(checkpoint)
    policy.config.device = device
    if num_steps > 0:
        policy.config.num_inference_steps = num_steps
    policy = policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--holdout-root", required=True)
    parser.add_argument("--holdout-repo-id", default="binhpham/candy-shop-holdout")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=10, help="Flow-matching steps.")
    parser.add_argument("--stride", type=int, default=25,
                        help="Sample every Nth frame; chunks overlap heavily otherwise.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    policy, pre, post = load(args.checkpoint, args.device, args.num_steps)
    chunk = policy.config.chunk_size
    fps_deltas = None

    dataset = LeRobotDataset(args.holdout_repo_id, root=args.holdout_root)
    fps_deltas = {ACTION: [i / dataset.fps for i in range(chunk)]}
    dataset = LeRobotDataset(args.holdout_repo_id, root=args.holdout_root,
                             delta_timestamps=fps_deltas)

    torch.manual_seed(args.seed)
    per_horizon: dict[int, list[float]] = {h: [] for h in HORIZONS}
    per_joint: list[np.ndarray] = []
    n = 0

    for index in range(0, len(dataset), args.stride):
        item = dataset[index]
        # Ground truth must be read before the preprocessor runs: it rewrites `action`
        # into the relative, normalized space the model is trained in, while what we
        # want to compare against is the absolute command the operator actually sent.
        target = item[ACTION].clone()  # [chunk, action_dim], absolute
        batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in item.items()}
        for cam in dataset.meta.camera_keys:
            if cam in batch and batch[cam].dtype == torch.uint8:
                batch[cam] = batch[cam].to(dtype=torch.float32) / 255.0

        processed = pre(batch)
        with torch.inference_mode():
            predicted = policy.predict_action_chunk(processed)
        predicted = post(predicted).squeeze(0).cpu()  # [chunk, action_dim], absolute

        dims = min(predicted.shape[-1], target.shape[-1])
        error = (predicted[:, :dims] - target[:, :dims]).abs()
        for h in HORIZONS:
            per_horizon[h].append(float(error[:h].mean()))
        per_joint.append(error.mean(dim=0).numpy())
        n += 1

    joint_means = np.mean(np.stack(per_joint), axis=0)
    result = {
        "checkpoint": args.checkpoint,
        "samples": n,
        "episodes": dataset.num_episodes,
        "mae_units": {f"h{h}": round(float(np.mean(v)), 4) for h, v in per_horizon.items()},
        "mae_per_joint": [round(float(v), 4) for v in joint_means],
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
