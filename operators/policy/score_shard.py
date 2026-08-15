"""Score one shard of episodes with SARM, for running the scorer N-ways in parallel.

``compute_sarm_progress`` is single-process and, measured on the 8xH100 box, spends its
time in video decode rather than on the GPU: one core busy, GPU at 4%, ~1 episode/min.
Scoring 500 episodes that way takes hours while seven H100s idle, and ``stride`` barely
helps because the cost is decoding the video, not running the model on the frames it
yields.

Episodes are independent, so the fix is to shard them. Two facts make the shards
concatenable, both verified against this dataset rather than assumed:

* ``LeRobotDataset(..., episodes=[3, 4])`` keeps the *dataset-global* frame index --
  its first sample reports ``index == 602``, the same value that frame has in the full
  dataset, not 0.
* It also keeps the original ``episode_index`` (3, not renumbered to 0).

So each shard emits rows already in the full dataset's coordinate system, and merging is
a concatenate-and-sort. This is the opposite of what ``delete_episodes`` does, which
*does* renumber -- see operators/policy/curate.py.

Usage (one process per GPU, then merge)::

    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i python -m operators.policy.score_shard \\
        --dataset-root ~/data/candy-shop-rel --reward-model-path ~/models/sarm \\
        --output ~/data/shard_$i.parquet --shard-index $i --num-shards 8 --stride 4 &
    done; wait
    python -m operators.policy.score_shard --merge ~/data/shard_*.parquet \\
        --output ~/data/rabc_progress.parquet
"""
from __future__ import annotations

import argparse
import os

import pandas as pd


def merge(paths: list[str], output: str) -> None:
    """Concatenate shard parquets back into one, ordered as the trainer expects.

    RABCWeighter looks rows up by ``index``, so the merged file must be sorted by it and
    must not contain duplicates -- an overlap would silently shadow one frame's progress
    with another's.
    """
    frames = [pd.read_parquet(p) for p in paths]
    merged = pd.concat(frames, ignore_index=True).sort_values("index")

    duplicated = int(merged["index"].duplicated().sum())
    if duplicated:
        raise SystemExit(f"{duplicated} duplicate index values across shards; shards overlap")

    merged = merged.reset_index(drop=True)
    merged.to_parquet(output)
    print(f"merged {len(paths)} shards -> {output}")
    print(f"rows={len(merged)} episodes={merged['episode_index'].nunique()}")
    for column in ("progress_sparse", "progress_dense"):
        if column in merged:
            valid = merged[column].dropna()
            if len(valid):
                print(f"{column}: mean={valid.mean():.4f} min={valid.min():.4f} max={valid.max():.4f}")


def score(args: argparse.Namespace) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    import lerobot.rewards.sarm.compute_rabc_weights as scorer

    total_episodes = LeRobotDataset(args.repo_id, root=args.dataset_root).num_episodes
    mine = list(range(args.shard_index, total_episodes, args.num_shards))
    print(f"shard {args.shard_index}/{args.num_shards}: {len(mine)} of {total_episodes} episodes")

    # Strided rather than contiguous so every shard gets a similar mix of episode
    # lengths; contiguous blocks would leave one process holding all the long ones.
    original = scorer.LeRobotDataset

    def sharded(repo_id, **kwargs):
        kwargs.setdefault("root", args.dataset_root)
        kwargs.setdefault("episodes", mine)
        return original(repo_id, **kwargs)

    # load_sarm_resources builds the dataset twice (once to read fps, once with
    # delta_timestamps), so patch the name it looks up rather than passing an instance.
    scorer.LeRobotDataset = sharded
    try:
        scorer.compute_sarm_progress(
            dataset_repo_id=args.repo_id,
            reward_model_path=args.reward_model_path,
            output_path=args.output,
            head_mode=args.head_mode,
            device="cuda",
            num_visualizations=0,
            stride=args.stride,
        )
    finally:
        scorer.LeRobotDataset = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merge", nargs="+", help="Shard parquets to concatenate.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo_id", default="binhpham/candy-shop-rel")
    parser.add_argument("--dataset-root")
    parser.add_argument("--reward-model-path")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--head-mode", default="sparse", choices=["sparse", "dense", "both"])
    args = parser.parse_args()

    if args.merge:
        merge(args.merge, args.output)
        return

    missing = [f for f in ("dataset_root", "reward_model_path", "shard_index", "num_shards")
               if getattr(args, f) is None]
    if missing:
        parser.error("scoring requires " + ", ".join("--" + f.replace("_", "-") for f in missing))
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    score(args)


if __name__ == "__main__":
    main()
