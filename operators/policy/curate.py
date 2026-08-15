"""Rank episodes by SARM progress and print the ones to drop.

The folding blog's single biggest lever was not an algorithm. Across 11 experiments
their algorithmic changes moved success rate by 5-20 points; fine-tuning on a curated
dataset one fifth the size moved it by 50. They built that subset two ways -- discard
episodes whose final frame does not show the task completed ("end-state filtering"),
and drop length outliers, since suspiciously short episodes tend to be low quality --
and then scored what remained with SARM.

We have the same signal without the manual pass: ``compute_sarm_progress`` already
emits a per-frame progress value in [0, 1] for every frame of every episode. This
turns that into three per-episode numbers and selects on them:

``final``
    Progress over the last ``TAIL_FRAMES`` frames. This is end-state filtering: an
    episode that never approaches 1.0 did not finish the task, whatever it looks like.
``monotonicity``
    Fraction of frame-to-frame deltas that are non-negative. SARM dips when the arm
    regresses, so a hesitant or retried episode scores low even if it eventually
    succeeds. This is the part length-based filtering was a crude proxy for.
``length``
    Frame count, used only to drop outliers at both ends.

Emits the ``--operation.episode_indices`` list for ``lerobot-edit-dataset
--operation.type delete_episodes``; it never mutates a dataset itself, so the
selection can be eyeballed before anything is written.

RECOMPUTE THE RABC PARQUET AFTER CURATING. ``RABCWeighter`` looks its progress values
up by ``batch["index"]``, the dataset-global frame index, and deleting episodes
renumbers those. Reusing the parquet from the uncurated dataset therefore lines each
frame up against some other frame's progress -- and it does so quietly: rabc.py only
warns when ``index`` is missing outright, never when it is present but shifted. The
same goes for the relative-action stats, which should be recomputed over the episodes
that actually remain.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

# Progress is noisy frame to frame, so "did it finish" reads the tail rather than the
# single last frame -- one bad final frame should not condemn a good episode.
TAIL_FRAMES = 15


def episode_metrics(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per-episode final progress, monotonicity, and length."""
    rows = []
    for episode_index, group in df.groupby("episode_index"):
        progress = group.sort_values("frame_index")[column].to_numpy(dtype=np.float32)
        progress = progress[~np.isnan(progress)]
        if progress.size == 0:
            continue
        deltas = np.diff(progress)
        rows.append(
            {
                "episode_index": int(episode_index),
                "final": float(np.mean(progress[-TAIL_FRAMES:])),
                "monotonicity": float(np.mean(deltas >= 0)) if deltas.size else 1.0,
                "length": int(progress.size),
            }
        )
    return pd.DataFrame(rows)


def select(metrics: pd.DataFrame, keep_fraction: float, length_quantile: float) -> pd.DataFrame:
    """Rank by final progress then monotonicity, after dropping length outliers.

    Length filtering runs first and on its own: it is a data-integrity check (a
    truncated recording is not a demonstration), and folding it into the score would
    let a long hesitant episode outrank a short clean one on length alone.
    """
    low, high = metrics["length"].quantile([length_quantile, 1 - length_quantile])
    kept = metrics[(metrics["length"] >= low) & (metrics["length"] <= high)].copy()

    # Rank-average rather than a weighted sum of the raw numbers: `final` clusters near
    # 1.0 while `monotonicity` spreads over most of [0, 1], so summing them directly
    # would let monotonicity dominate purely because of its scale.
    kept["score"] = (
        kept["final"].rank(pct=True) + kept["monotonicity"].rank(pct=True)
    ) / 2
    kept = kept.sort_values("score", ascending=False)
    return kept.head(max(1, int(round(len(metrics) * keep_fraction))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--progress-parquet", required=True,
                        help="Output of compute_sarm_progress.")
    parser.add_argument("--column", default="progress_sparse",
                        choices=["progress_sparse", "progress_dense"])
    parser.add_argument("--keep-fraction", type=float, default=0.6,
                        help="Fraction of episodes to keep. The blog kept ~21%% (1,200 of "
                             "5,688), but they had 5,688 to spend; at 500 episodes "
                             "cutting that hard would leave too little to fit.")
    parser.add_argument("--length-quantile", type=float, default=0.02,
                        help="Trim this quantile from each end of the length distribution.")
    parser.add_argument("--out-json", default=None,
                        help="Write the full per-episode table here for inspection.")
    args = parser.parse_args()

    df = pd.read_parquet(args.progress_parquet)
    metrics = episode_metrics(df, args.column)
    keep = select(metrics, args.keep_fraction, args.length_quantile)

    keep_set = set(keep["episode_index"])
    drop = sorted(set(metrics["episode_index"]) - keep_set)

    print(f"episodes scored : {len(metrics)}")
    print(f"kept            : {len(keep_set)}")
    print(f"dropped         : {len(drop)}")
    print(f"kept   final    : mean={keep['final'].mean():.4f} min={keep['final'].min():.4f}")
    print(f"kept   monotone : mean={keep['monotonicity'].mean():.4f}")
    dropped = metrics[metrics["episode_index"].isin(drop)]
    if len(dropped):
        print(f"dropped final   : mean={dropped['final'].mean():.4f} max={dropped['final'].max():.4f}")
        print(f"dropped monotone: mean={dropped['monotonicity'].mean():.4f}")

    if args.out_json:
        metrics.assign(kept=metrics["episode_index"].isin(keep_set)).to_json(
            args.out_json, orient="records", indent=2
        )
        print(f"wrote {args.out_json}")

    print("\n--operation.episode_indices " + json.dumps(drop, separators=(",", ":")))


if __name__ == "__main__":
    main()
