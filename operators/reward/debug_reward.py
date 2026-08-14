"""Offline calibration driver for the SARM reward model: replay recorded episodes
through the *live operator's* scorer and print where it would have called "done".

``--threshold`` and ``--hold-seconds`` decide when ``run.py`` preempts the policy,
and picking them blind means discovering on the rig that the reward fires halfway
through a pick (or never). This scores episodes from a LeRobot dataset instead —
no robot, no room, no policy — and reports, per episode, the progress curve and
the moment the operator's completion test would have tripped.

It is faithful by construction: the same :class:`ProgressScorer` and
:class:`DoneRule` the operator uses, fed one frame per ``--eval-interval`` just as
the operator's poll loop feeds it. So a threshold that looks right here is the
threshold to pass to ``uv run reward``.

Reading the output
------------------
Recorded episodes are successful demonstrations that end at the grasp, so the
reward *should* cross late and stay crossed:

``done@``
  Seconds into the episode the completion test tripped. Blank means it never did
  — the threshold is too high, and on the rig the pick would burn its whole
  attempt budget and retry a pick that had already succeeded.
``lead``
  Episode duration minus ``done@``: how early it fired. Large positive lead is
  the dangerous direction — the operator stops the policy mid-pick and calls it
  done. Near zero is ideal.

The threshold sweep at the end reruns the completion test over the same scored
curves at a range of thresholds, so one pass costs one round of inference and
tells you what every threshold would have done.

Usage::

    uv run reward-debug --checkpoint outputs/sarm-candy/checkpoints/last/pretrained_model \\
        --dataset <user>/candy-shop --dataset-root data/<user>/candy-shop
    uv run reward-debug ... --episodes 0-49        # a bigger sample
    uv run reward-debug ... --threshold 0.8        # try a different cut
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import statistics
import time

import numpy as np
import torch

from shared.common import env_str, load_env

from operators.reward.sarm import (ClipEncoder, DEFAULT_CHECKPOINT, DoneRule, ProgressScorer,
                                   StateNormalizer, image_key_for, load_reward_model)

# Coarse enough to read at a glance, fine enough to see the curve's shape.
SPARK = " ▁▂▃▄▅▆▇█"
SWEEP_THRESHOLDS = (0.7, 0.8, 0.9, 0.95, 0.97, 0.99)

logger = logging.getLogger(__name__)


def _parse_episodes(raw: str, total: int) -> list[int]:
    """``"0-9"`` / ``"0,5,7"`` / ``"all"`` -> a list of episode indices."""
    if raw.strip().lower() == "all":
        return list(range(total))
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    bad = [e for e in out if not 0 <= e < total]
    if bad:
        raise SystemExit(f"--episodes: {bad} out of range (dataset has {total})")
    if not out:
        raise SystemExit(f"--episodes: {raw!r} selected nothing")
    return out


def _to_rgb_frame(tensor: torch.Tensor) -> np.ndarray:
    """LeRobot's CHW float [0,1] image -> the HWC uint8 RGB the scorer expects."""
    return (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _sparkline(values: list[float]) -> str:
    """Render a 0..1 series as block characters."""
    return "".join(SPARK[min(len(SPARK) - 1, max(0, int(v * (len(SPARK) - 1))))] for v in values)


def _score_episode(dataset, scorer: ProgressScorer, image_key: str,
                   start: int, stop: int, stride: int) -> list[float]:
    """Replay one episode at the operator's poll cadence -> one progress per poll.

    Causal, exactly as online: push a frame, score immediately, so early polls see
    a partial window rather than the centred one SARM trained on.
    """
    scorer.reset()
    scores = []
    for index in range(start, stop, stride):
        item = dataset[index]
        state = item.get("observation.state")
        scorer.push(_to_rgb_frame(item[image_key]),
                    None if state is None else state.numpy())
        scores.append(scorer.progress())
    return scores


def _first_done(scores: list[float], rule: DoneRule) -> int | None:
    """Index of the poll where the completion test trips, or None."""
    rule.reset()
    for i, value in enumerate(scores):
        if rule.push(value):
            return i
    return None


def _summarize(results: list[dict], interval: float) -> None:
    fired = [r for r in results if r["done_s"] is not None]
    print(f"\n{len(results)} episodes | fired {len(fired)}/{len(results)} "
          f"({100.0 * len(fired) / len(results):.0f}%)", end="")
    if fired:
        print(f" | median done {statistics.median(r['done_s'] for r in fired):.1f}s"
              f" | median lead {statistics.median(r['lead_s'] for r in fired):+.1f}s", end="")
    print(f"\nmedian peak {statistics.median(r['peak'] for r in results):.2f}"
          f" | median final {statistics.median(r['final'] for r in results):.2f}"
          f" | median early-quarter max {statistics.median(r['early_max'] for r in results):.2f}"
          "   (early-quarter should sit well below the threshold)")
    if len(fired) < len(results):
        print(f"note: {len(results) - len(fired)} episode(s) never crossed — on the rig those "
              "burn every attempt budget and retry an already-finished pick.")


def _sweep(results: list[dict], hold_s: float, interval: float) -> None:
    """Re-run the completion test over the scored curves at a range of thresholds."""
    print(f"\nthreshold sweep (hold {hold_s:g}s @ {interval:g}s polls):")
    print(f"  {'thr':>5}  {'fired':>7}  {'median done':>12}  {'median lead':>12}")
    for threshold in SWEEP_THRESHOLDS:
        rule = DoneRule(threshold, hold_s, interval)
        done = [(r, _first_done(r["scores"], rule)) for r in results]
        hit = [(r, i) for r, i in done if i is not None]
        cells = f"  {threshold:>5.2f}  {f'{len(hit)}/{len(results)}':>7}"
        if hit:
            times = [i * interval for _, i in hit]
            leads = [r["duration_s"] - i * interval for r, i in hit]
            cells += f"  {statistics.median(times):>11.1f}s  {statistics.median(leads):>+11.1f}s"
        else:
            cells += f"  {'-':>12}  {'-':>12}"
        print(cells)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score recorded episodes with SARM to calibrate the reward operator's threshold.")
    parser.add_argument("--checkpoint", default=env_str("REWARD_CHECKPOINT", "") or DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", default=env_str("DATASET_REPO_ID", ""),
                        help="LeRobot dataset repo_id (defaults to $DATASET_REPO_ID).")
    parser.add_argument("--dataset-root", default=env_str("DATASET_ROOT", "") or None,
                        help="Local dataset root (skip Hub download).")
    parser.add_argument("--no-values", action="store_true",
                        help="Print only the sparkline, not the per-poll reward numbers.")
    parser.add_argument("--episodes", default="0-9",
                        help="Episodes to score: '0-9', '0,5,7', or 'all'. Default '0-9'.")
    parser.add_argument("--camera", default=env_str("REWARD_CAMERA", "overhead_camera"),
                        help="Camera SARM watches (must match training).")
    parser.add_argument("--task", default=env_str("REWARD_TASK", "pick up the candy"),
                        help="Fallback instruction if the dataset has no language column.")
    parser.add_argument("--device", default=env_str("REWARD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    # These three mirror `uv run reward`, because calibrating them is the point.
    parser.add_argument("--threshold", type=float, default=float(env_str("REWARD_THRESHOLD", "0.97")))
    parser.add_argument("--hold-seconds", type=float, default=float(env_str("REWARD_HOLD_S", "1.0")))
    parser.add_argument("--eval-interval", type=float, default=float(env_str("REWARD_EVAL_INTERVAL_S", "1.0")),
                        help="Seconds between polls; also the frame stride, as online.")
    parser.add_argument("--video-backend", default="torchcodec",
                        help="pyav reseeks per frame and is much slower for strided reads.")
    args = parser.parse_args()

    # force: importing lerobot installs a root handler, which would make this a no-op.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
    load_env(pathlib.Path(__file__).resolve().parent)

    if not args.dataset:
        raise SystemExit("--dataset is required (or set DATASET_REPO_ID)")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.dataset, root=args.dataset_root, video_backend=args.video_backend)
    image_key = image_key_for(args.camera)
    if image_key not in dataset.meta.camera_keys:
        raise SystemExit(f"camera {image_key!r} not in dataset (have {list(dataset.meta.camera_keys)})")

    fps = dataset.meta.fps
    stride = max(1, round(args.eval_interval * fps))
    episodes = _parse_episodes(args.episodes, dataset.meta.total_episodes)
    starts = dataset.meta.episodes["dataset_from_index"]
    stops = dataset.meta.episodes["dataset_to_index"]

    model, config = load_reward_model(args.checkpoint, args.device)
    scorer = ProgressScorer(model, config, ClipEncoder(args.device),
                            StateNormalizer.from_checkpoint(args.checkpoint))
    rule = DoneRule(args.threshold, args.hold_seconds, args.eval_interval)

    print(f"\n{args.checkpoint}\n{args.dataset} @ {fps}fps, watching {image_key}\n"
          f"threshold {args.threshold:g}, hold {args.hold_seconds:g}s ({rule.hold_ticks} "
          f"poll{'s' if rule.hold_ticks != 1 else ''}), polling every {args.eval_interval:g}s "
          f"(every {stride} frames), window {config.n_obs_steps + 1} frames\n")
    print(f"{'ep':>5} {'frames':>7} {'dur':>7}  {'curve':<26} {'peak':>5} {'final':>6} "
          f"{'done@':>7} {'lead':>7}")

    results: list[dict] = []
    t_start = time.monotonic()
    for episode in episodes:
        start, stop = int(starts[episode]), int(stops[episode])
        # Every frame of an episode carries the same instruction; SARM's stage is
        # named by it, so take it from the episode rather than assuming --task.
        task = dataset[start].get("task") or args.task
        scorer.set_task(task)

        scores = _score_episode(dataset, scorer, image_key, start, stop, stride)
        duration_s = (stop - start) / fps
        done_index = _first_done(scores, rule)
        done_s = None if done_index is None else done_index * args.eval_interval
        early = scores[: max(1, len(scores) // 4)]

        result = {"episode": episode, "scores": scores, "duration_s": duration_s,
                  "done_s": done_s, "lead_s": None if done_s is None else duration_s - done_s,
                  "peak": max(scores), "final": scores[-1], "early_max": max(early)}
        results.append(result)

        done_cell = "-" if done_s is None else f"{done_s:.1f}s"
        lead_cell = "-" if done_s is None else f"{result['lead_s']:+.1f}s"
        print(f"{episode:>5} {stop - start:>7} {duration_s:>6.1f}s  "
              f"{_sparkline(scores):<26} {result['peak']:>5.2f} {result['final']:>6.2f} "
              f"{done_cell:>7} {lead_cell:>7}")
        if not args.no_values:
            # The poll at which the rule trips is marked, so "done@" is checkable by eye.
            cells = [f"{'*' if done_index is not None and i == done_index else ''}{v:.2f}"
                     for i, v in enumerate(scores)]
            print(f"{'':>5} {'rewards':>7} {'':>7}  " + " ".join(cells))

    _summarize(results, args.eval_interval)
    _sweep(results, args.hold_seconds, args.eval_interval)
    print(f"\nscored {len(results)} episodes in {time.monotonic() - t_start:.1f}s")


def cli() -> None:
    try:
        main()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
