"""Fine-tune a SARM reward model on a candy-shop dataset (single task).

Minimal single-GPU training loop built on lerobot's own reward-model factories,
mirroring ``operators/policy/train.py``. Point it at a LeRobot dataset recorded on
the leslider rig and it produces a SARM checkpoint under ``--output`` that the
reward operator (``run.py``) serves.

Single-task only: we run SARM in ``single_stage`` annotation mode, so no VLM
subtask annotations are needed — the episode's task description is treated as one
stage spanning the whole episode, and the progress target is just the frame's
position within its episode.

SARM watches **one** camera (``--camera``, the overhead view by default) and the
full ``observation.state``; the slider dim is harmless (SARM pads state to
``max_state_dim``), so nothing is dropped here.

Usage::

    uv run reward-train --dataset <user>/candy_shop --dataset-root data/candy_shop
    # writes outputs/sarm-candy; serve it with:  uv run reward
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import torch
from torch.utils.data import DataLoader, default_collate

from lerobot.datasets import EpisodeAwareSampler
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.rewards import SARMConfig, make_reward_model, make_reward_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn

from shared.common import load_env

from operators.reward.sarm import image_key_for

logger = logging.getLogger(__name__)


def _cycle(dataloader: DataLoader):
    while True:
        yield from dataloader


def _save(model, preprocessor, postprocessor, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    # The processors carry the state normalization stats; run.py can reload them.
    preprocessor.save_pretrained(out_dir)
    postprocessor.save_pretrained(out_dir)
    logger.info("Saved checkpoint -> %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SARM on a candy-shop dataset (single task).")
    parser.add_argument("--dataset", required=True, help="LeRobot dataset repo_id.")
    parser.add_argument("--dataset-root", default=None, help="Local dataset root (skip Hub download).")
    parser.add_argument("--output", default="outputs/sarm-candy", help="Output dir.")
    parser.add_argument("--camera", default="overhead_camera", help="Camera SARM watches (single view).")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-obs-steps", type=int, default=8, help="Frames in the observation window.")
    parser.add_argument("--frame-gap", type=int, default=30, help="Frame stride in the window (30 @ 30fps = 1s).")
    parser.add_argument("--max-rewind-steps", type=int, default=4, help="Temporal-augmentation rewind frames.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task", default="", help="Fallback instruction if the dataset has no language column.")
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--save-freq", type=int, default=2_000)
    parser.add_argument("--video-backend", default="pyav")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_env(pathlib.Path(__file__).resolve().parent)

    # Metadata first: we need fps + the camera key to build the observation window
    # before opening the (heavier) dataset.
    meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    image_key = image_key_for(args.camera)
    if image_key not in meta.camera_keys:
        raise RuntimeError(f"camera {image_key!r} not in dataset (have {list(meta.camera_keys)})")

    config = SARMConfig(
        annotation_mode="single_stage",
        device=args.device,
        image_key=image_key,
        n_obs_steps=args.n_obs_steps,
        frame_gap=args.frame_gap,
        max_rewind_steps=args.max_rewind_steps,
        batch_size=args.batch_size,
    )

    delta_timestamps = resolve_delta_timestamps(config, meta)
    logger.info("Dataset %s @ %d fps; SARM watches %s", args.dataset, meta.fps, image_key)

    dataset = LeRobotDataset(
        args.dataset,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )

    logger.info("Creating SARM model + processors (downloads CLIP weights on first run)...")
    model = make_reward_model(cfg=config, dataset_stats=dataset.meta.stats, dataset_meta=dataset.meta)
    if not model.is_trainable:
        raise ValueError("This SARM checkpoint is zero-shot; nothing to train.")
    preprocessor, postprocessor = make_reward_pre_post_processors(
        config, dataset_stats=dataset.meta.stats, dataset_meta=dataset.meta
    )

    optimizer = config.get_optimizer_preset().build(model.get_optim_params())
    scheduler_preset = config.get_scheduler_preset()
    scheduler = scheduler_preset.build(optimizer, args.steps) if scheduler_preset else None

    has_language = dataset.meta.has_language_columns

    def collate(samples):
        if has_language:
            return lerobot_collate_fn(samples)
        batch = default_collate(samples)
        # SARM's single stage is named by the task; inject one if the dataset lacks a
        # language column so the encoding step has a prompt to work with.
        batch["task"] = [args.task] * len(samples)
        return batch

    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        shuffle=True,
        seed=0,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=args.device == "cuda",
        drop_last=False,
        collate_fn=collate,
    )

    out_dir = pathlib.Path(args.output) / "pretrained_model"

    model.train()
    batches = _cycle(dataloader)
    logger.info("Training for %d steps (batch=%d) on %s", args.steps, args.batch_size, args.device)
    for step in range(1, args.steps + 1):
        batch = preprocessor(next(batches))
        loss, metrics = model.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info("step %d/%d loss=%.4f lr=%.2e", step, args.steps, float(metrics.get("loss", loss)), lr)

        if step % args.save_freq == 0 and step < args.steps:
            _save(model, preprocessor, postprocessor,
                  pathlib.Path(args.output) / "checkpoints" / f"{step:06d}" / "pretrained_model")

    _save(model, preprocessor, postprocessor, out_dir)
    logger.info("Done. Serve it with:  uv run reward --checkpoint %s", out_dir)


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
