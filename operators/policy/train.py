"""Fine-tune MolmoAct2 on a candy-shop dataset (single GPU).

The dataset's 7th field (``slider.vel``) is dropped end to end — see
``operators/policy/molmoact.py``.

    uv run policy-train --dataset <user>/candy_shop --dataset-root data/candy_shop
    # writes outputs/molmoact2-candy; VRAM knobs: --train-action-expert-only, --lora, --gradient-checkpointing
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
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION

from shared.common import load_env

from operators.policy.molmoact import DEFAULT_CHECKPOINT, SliderDroppedDataset

logger = logging.getLogger(__name__)


def _build_config(checkpoint: str, device: str, args: argparse.Namespace, image_keys: list[str]) -> MolmoAct2Config:
    """Load the checkpoint's config (LeRobot format) or start one fresh (raw HF)."""
    try:
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if not isinstance(config, MolmoAct2Config):
            raise TypeError(f"{checkpoint} is a {type(config).__name__}, not MolmoAct2.")
        config.pretrained_path = checkpoint
        logger.info("Fine-tuning from LeRobot checkpoint %s", checkpoint)
    except Exception:
        config = MolmoAct2Config(checkpoint_path=checkpoint)
        logger.info("Fine-tuning from original MolmoAct2 HF weights %s", checkpoint)

    # Clear inherited features so make_policy re-derives them from our
    # (slider-dropped) dataset; otherwise the old camera keys and 7-dim state stick.
    config.input_features = {}
    config.output_features = {}

    config.device = device
    config.chunk_size = args.chunk_size
    config.n_action_steps = args.chunk_size
    config.action_mode = "continuous" if args.train_action_expert_only else args.action_mode
    config.inference_action_mode = None
    config.image_keys = image_keys
    config.model_dtype = args.model_dtype
    config.gradient_checkpointing = args.gradient_checkpointing
    config.train_action_expert_only = args.train_action_expert_only
    config.enable_lora_vlm = args.lora
    if args.setup_type:
        config.setup_type = args.setup_type
    if args.control_mode:
        config.control_mode = args.control_mode
    return config


def _order_cameras(camera_keys: list[str], primary: str, wrist: str) -> list[str]:
    """Camera keys in policy order: primary (external) first, wrist second."""
    ordered = [f"observation.images.{name}" for name in (primary, wrist)]
    ordered = [key for key in ordered if key in camera_keys]
    ordered += [key for key in camera_keys if key not in ordered]
    return ordered


def _cycle(dataloader: DataLoader):
    while True:
        yield from dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MolmoAct2 on a candy-shop dataset.")
    parser.add_argument("--dataset", required=True, help="LeRobot dataset repo_id.")
    parser.add_argument("--dataset-root", default=None, help="Local dataset root (skip Hub download).")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Starting checkpoint (LeRobot or HF).")
    parser.add_argument("--output", default="outputs/molmoact2-candy", help="Output dir.")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=30, help="Action horizon (also = n_action_steps).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--action-mode", default="both", choices=["continuous", "discrete", "both"])
    parser.add_argument("--train-action-expert-only", action="store_true", help="Cheapest fine-tune.")
    parser.add_argument("--lora", action="store_true", help="LoRA on the VLM (action expert stays full).")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--primary-camera", default="overhead_camera")
    parser.add_argument("--wrist-camera", default="arm_camera")
    parser.add_argument("--setup-type", default="", help="Prompt text describing the robot/scene.")
    parser.add_argument("--control-mode", default="", help="Prompt text describing the action space.")
    parser.add_argument("--task", default="", help="Fallback instruction if the dataset has no language column.")
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--save-freq", type=int, default=2_000)
    parser.add_argument("--video-backend", default="pyav")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_env(pathlib.Path(__file__).resolve().parent)

    # Metadata first: fps + camera keys build delta_timestamps and image-key
    # order before opening the heavier dataset.
    meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    image_keys = _order_cameras(list(meta.camera_keys), args.primary_camera, args.wrist_camera)
    delta_timestamps = {ACTION: [i / meta.fps for i in range(args.chunk_size)]}
    logger.info("Dataset %s @ %d fps; cameras -> %s", args.dataset, meta.fps, image_keys)

    dataset = SliderDroppedDataset(
        args.dataset,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        return_uint8=True,
    )
    logger.info(
        "State/action dims after dropping slider.vel: %s",
        dataset.meta.features[ACTION]["names"],
    )

    config = _build_config(args.checkpoint, args.device, args, image_keys)
    logger.info("Creating policy (this downloads the MolmoAct2 base weights on first run)...")
    policy = make_policy(cfg=config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
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
    logger.info("Training for %d steps (batch=%d) on %s", args.steps, args.batch_size, args.device)
    for step in range(1, args.steps + 1):
        batch = preprocessor(next(batches))
        loss, metrics = policy.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info("step %d/%d loss=%.4f lr=%.2e", step, args.steps, metrics["loss"], lr)

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
