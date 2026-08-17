"""Does the policy actually condition on the instruction?

Every metric used on this project so far -- flow-matching `eval_loss`, and the holdout
chunk error in `eval_holdout.py` -- averages over the five tasks. A policy that ignores
the prompt entirely and reaches toward a plausible average candy still scores well on
both. That is exactly the failure seen on the arm: it goes to the wrong candy while the
offline numbers look reasonable.

This measures the thing directly. Take one real observation, run it through the model
once per instruction, and compare the predicted action chunks against each other. If the
policy reads the prompt, swapping "pick up a twix" for "pick up a nerd" should move the
target. If the chunks are near-identical, the prompt is not reaching the actions and no
amount of trajectory-quality tuning will fix the behaviour.

Reported per pair: mean absolute difference between chunks, in joint units, alongside
the within-instruction difference from re-sampling the same prompt (flow matching is
stochastic). The second number is the noise floor -- a between-instruction spread that
does not clear it means the prompt is being ignored.

    python -m operators.policy.language_probe \\
        --checkpoint outputs/arm-e-backbone/checkpoints/001000/pretrained_model \\
        --dataset-root /outputs/datasets/candy-holdout-rel
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import get_policy_class


def action_chunk(policy, post, batch):
    """The chunk the policy would execute, for either policy family.

    pi0/pi0.5 expose ``predict_action_chunk`` as a pure function of one observation.
    multi_task_dit does not: it keeps observation deques (``n_obs_steps`` frames of
    history) that ``select_action`` fills via ``populate_queues``, and calling
    ``predict_action_chunk`` directly on a fresh policy dies with "stack expects a
    non-empty TensorList". So drive it the way the runtime does -- reset, then pop
    ``n_action_steps`` actions; only the first call runs the model, the rest dequeue.
    """
    queue_based = getattr(policy, "_queues", None) is not None
    if not queue_based:
        with torch.inference_mode():
            out = policy.predict_action_chunk(batch)
        return post(out).squeeze(0).cpu().numpy()

    policy.reset()
    steps = getattr(policy.config, "n_action_steps", 1)
    actions = []
    with torch.inference_mode():
        for _ in range(steps):
            a = policy.select_action(dict(batch))
            actions.append(post(a).squeeze(0).cpu().numpy())
    return np.stack(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default="binhpham/candy-shop-holdout")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=0,
                        help="Override the checkpoint's inference steps. 0 keeps its own "
                             "setting, which matters for DDPM: these DiT arms train with "
                             "100 timesteps, and forcing 10 without switching to DDIM "
                             "under-integrates the reverse process and inflates the "
                             "sampling noise this probe divides by.")
    parser.add_argument("--frames", type=int, default=6,
                        help="How many distinct observations to probe.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Re-samples per instruction, to size the noise floor.")
    args = parser.parse_args()

    policy_type = json.loads((pathlib.Path(args.checkpoint) / "config.json").read_text())["type"]
    policy = get_policy_class(policy_type).from_pretrained(args.checkpoint)
    policy.config.device = args.device
    if args.num_steps > 0 and hasattr(policy.config, "num_inference_steps"):
        policy.config.num_inference_steps = args.num_steps
    print("inference steps:", getattr(policy.config, "num_inference_steps", None),
          "| scheduler:", getattr(policy.config, "noise_scheduler_type", "n/a"),
          "| train timesteps:", getattr(policy.config, "num_train_timesteps", "n/a"))
    policy = policy.to(args.device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}})

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    instructions = sorted({t[0] for t in dataset.meta.episodes["tasks"] if t})
    print(f"instructions ({len(instructions)}):")
    for t in instructions:
        print(f"  - {t}")

    # Spread the probe frames across the dataset rather than taking the first few, which
    # would all sit in the same episode at the same point in the motion.
    indices = np.linspace(0, len(dataset) - 1, args.frames, dtype=int)

    between, within = [], []
    for idx in indices:
        item = dataset[int(idx)]
        chunks: dict[str, list[np.ndarray]] = {}
        for task in instructions:
            reps = []
            for r in range(args.repeats):
                batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in item.items()}
                batch["task"] = [task]
                for cam in dataset.meta.camera_keys:
                    if cam in batch and batch[cam].dtype == torch.uint8:
                        batch[cam] = batch[cam].to(dtype=torch.float32) / 255.0
                torch.manual_seed(r)
                processed = pre(batch)
                reps.append(action_chunk(policy, post, processed))
            chunks[task] = reps
            for a, b in itertools.combinations(reps, 2):
                within.append(np.abs(a - b).mean())
        for t1, t2 in itertools.combinations(instructions, 2):
            between.append(np.abs(chunks[t1][0] - chunks[t2][0]).mean())

    b, w = float(np.mean(between)), float(np.mean(within))
    print()
    print(f"between-instruction difference : {b:.4f} units")
    print(f"within-instruction  (noise)    : {w:.4f} units")
    print(f"ratio                          : {b / w:.2f}x" if w else "ratio: n/a")
    print()
    if w and b / w < 1.5:
        print("VERDICT: the prompt barely moves the actions -- the policy is effectively")
        print("         ignoring the instruction. Trajectory-quality fixes will not help.")
    else:
        print("VERDICT: the prompt does change the actions; language is reaching the")
        print("         policy, so wrong-candy behaviour has some other cause.")


if __name__ == "__main__":
    main()
