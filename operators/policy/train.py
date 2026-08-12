"""Fine-tune SmolVLA on a candy-shop dataset (single GPU).

The dataset's 7th field (``slider.vel``) is dropped end to end — see
``operators/policy/smolvla.py``.

    uv run policy-train --dataset <user>/candy_shop --dataset-root data/candy_shop
    # writes outputs/smolvla-candy; VRAM knobs: --batch-size, --grad-accum
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import torch
from torch.utils.data import DataLoader, default_collate

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import EpisodeAwareSampler
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION

from shared.common import load_env

from operators.policy.smolvla import BASE_CHECKPOINT, DEFAULT_OUTPUT_DIR, SliderDroppedDataset

logger = logging.getLogger(__name__)


def _build_config(checkpoint: str, device: str, args: argparse.Namespace) -> SmolVLAConfig:
    """Load the starting checkpoint's config, or start a fresh action expert."""
    if checkpoint.strip().lower() in ("", "none", "scratch"):
        # No SmolVLA weights to inherit: pull the VLM backbone and train the
        # flow-matching action expert from scratch (needs far more data).
        config = SmolVLAConfig(load_vlm_weights=True)
        logger.info("Training a fresh action expert on the %s backbone", config.vlm_model_name)
    else:
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if not isinstance(config, SmolVLAConfig):
            raise TypeError(f"{checkpoint} is a {type(config).__name__}, not SmolVLA.")
        config.pretrained_path = checkpoint
        logger.info("Fine-tuning from LeRobot checkpoint %s", checkpoint)

    # Clear inherited features so make_policy re-derives them from our
    # (slider-dropped) dataset; otherwise the base checkpoint's placeholder
    # camera keys (camera1..3) stick and our frames match nothing.
    config.input_features = {}
    config.output_features = {}

    config.device = device
    config.chunk_size = args.chunk_size
    config.n_action_steps = args.chunk_size
    config.freeze_vision_encoder = not args.unfreeze_vision_encoder
    config.train_expert_only = not args.train_vlm
    return config


def _cycle(dataloader: DataLoader):
    while True:
        yield from dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA on a candy-shop dataset.")
    parser.add_argument("--dataset", required=True, help="LeRobot dataset repo_id.")
    parser.add_argument("--dataset-root", default=None, help="Local dataset root (skip Hub download).")
    parser.add_argument("--checkpoint", default=BASE_CHECKPOINT,
                        help="Starting SmolVLA checkpoint, or 'scratch' for a fresh action expert.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output dir.")
    parser.add_argument("--steps", type=int, default=10_000, help="Optimizer steps (not micro-batches).")
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch; effective batch = this x --grad-accum.")
    parser.add_argument("--grad-accum", type=int, default=1, help="Micro-batches per optimizer step.")
    parser.add_argument("--chunk-size", type=int, default=50, help="Action horizon (also = n_action_steps).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-vlm", action="store_true",
                        help="Train the VLM alongside the action expert (default: expert only).")
    parser.add_argument("--unfreeze-vision-encoder", action="store_true",
                        help="Also train the SigLIP vision encoder (default: frozen).")
    parser.add_argument("--task", default="", help="Fallback instruction if the dataset has no language column.")
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--save-freq", type=int, default=2_000)
    parser.add_argument("--video-backend", default="pyav")
    args = parser.parse_args()

    # force: importing lerobot installs a root handler, which would make this a no-op
    # and leave the root logger at WARNING — every INFO line below silently dropped.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True
    )
    load_env(pathlib.Path(__file__).resolve().parent)

    # Metadata first: fps builds delta_timestamps before opening the heavier dataset.
    meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    delta_timestamps = {ACTION: [i / meta.fps for i in range(args.chunk_size)]}
    logger.info("Dataset %s @ %d fps; cameras -> %s", args.dataset, meta.fps, list(meta.camera_keys))

    # Frames stay float [0, 1] (the dataset default): SmolVLA's own preprocessing
    # shifts that to SigLIP's [-1, 1] and has no uint8 step to rescale for it.
    dataset = SliderDroppedDataset(
        args.dataset,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )
    logger.info(
        "State/action dims after dropping slider.vel: %s",
        dataset.meta.features[ACTION]["names"],
    )

    config = _build_config(args.checkpoint, args.device, args)
    logger.info("Creating policy (this downloads the SmolVLA base weights on first run)...")
    policy = make_policy(cfg=config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        dataset_stats=dataset.meta.stats,
    )

    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    scheduler_preset = config.get_scheduler_preset()
    scheduler = scheduler_preset.build(optimizer, args.steps) if scheduler_preset else None

    has_language = dataset.meta.has_language_columns

    def collate(samples):
        if has_language:
            return lerobot_collate_fn(samples)
        batch = default_collate(samples)
        if args.task:
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
    grad_clip = config.optimizer_grad_clip_norm

    policy.train()
    batches = _cycle(dataloader)
    logger.info(
        "Training for %d steps (batch=%d x %d accum = %d) on %s",
        args.steps, args.batch_size, args.grad_accum, args.batch_size * args.grad_accum, args.device,
    )
    for step in range(1, args.steps + 1):
        step_loss = 0.0
        for _ in range(args.grad_accum):
            batch = preprocessor(next(batches))
            loss, metrics = policy.forward(batch)
            # Mean-over-micro-batches, so grad magnitude matches one big batch.
            (loss / args.grad_accum).backward()
            step_loss += float(metrics["loss"]) / args.grad_accum
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info("step %d/%d loss=%.4f lr=%.2e", step, args.steps, step_loss, lr)

        if step % args.save_freq == 0 and step < args.steps:
            _save(policy, preprocessor, postprocessor, pathlib.Path(args.output) / "checkpoints" / f"{step:06d}" / "pretrained_model")

    _save(policy, preprocessor, postprocessor, out_dir)
    logger.info("Done. Serve it with:  uv run policy --checkpoint %s", out_dir)


def _save(policy, preprocessor, postprocessor, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    # Processors carry the (sliced) normalization stats; run.py reloads them.
    preprocessor.save_pretrained(out_dir)
    postprocessor.save_pretrained(out_dir)
    logger.info("Saved checkpoint -> %s", out_dir)


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
