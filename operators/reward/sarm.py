"""Shared SARM reward-model wiring for the candy-shop reward operator.

Everything both ``train.py`` and ``run.py`` need to agree on lives here so the
two never drift: the default checkpoint, the single image key, and the online
progress scorer.

What SARM is
------------
SARM (Stage-Aware Reward Modeling, https://arxiv.org/abs/2509.25358) is a
*reward model*, not a policy. It watches a short window of frames plus the task
text and emits a scalar **progress** reward in ``[0, 1]`` — 0 at the start of the
task, ~1 when it's complete. We run it in ``single_stage`` mode: no per-frame VLM
annotations, just the episode's task description as one stage over the whole clip
(see :class:`~lerobot.rewards.sarm.configuration_sarm.SARMConfig`).

How the reward operator uses it
-------------------------------
The reward operator is a thin wrapper: its ``run_task`` RPC hands active control
to the *policy* operator, then polls SARM on the incoming camera frames until the
progress reward says the task is done (or a timeout fires), then releases. So all
this module owns is turning live observations into a progress number.

Inference is CLIP-then-SARM: SARM consumes CLIP embeddings, not raw pixels, so we
load the same ``openai/clip-vit-base-patch32`` encoder the training processor uses
and feed its image/text features into ``SARMRewardModel.calculate_rewards``. State
is optional and left out online (the checkpoint's state normalizer isn't wired
here); SARM's progress signal is driven by vision + language.
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np
import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards import SARMConfig, make_reward_model
from lerobot.utils.constants import OBS_IMAGES

logger = logging.getLogger(__name__)

# Trained locally by train.py; there is no released candy-shop SARM checkpoint.
DEFAULT_CHECKPOINT = "outputs/sarm-candy/pretrained_model"

# The CLIP encoder SARM was trained against (see processor_sarm.py). SARM eats
# CLIP embeddings, so inference must use the exact same encoder.
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def image_key_for(camera: str) -> str:
    """The dataset/observation image key SARM reads (single camera)."""
    return f"{OBS_IMAGES}.{camera}"


def load_reward_model(checkpoint: str, device: str) -> tuple[torch.nn.Module, SARMConfig]:
    """Load a trained SARM checkpoint for inference."""
    logger.info("[reward] loading SARM checkpoint %s ...", checkpoint)
    config = RewardModelConfig.from_pretrained(checkpoint)
    if not isinstance(config, SARMConfig):
        raise TypeError(f"{checkpoint} is a {type(config).__name__}, not SARM.")
    config.pretrained_path = checkpoint
    config.device = device
    model = make_reward_model(cfg=config)
    model.eval()
    return model, config


class ClipEncoder:
    """The CLIP image/text encoder feeding SARM, mirroring the training processor."""

    def __init__(self, device: str) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self._device = torch.device(device)
        logger.info("[reward] loading CLIP encoder %s ...", CLIP_MODEL_ID)
        self._model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self._device).eval()
        self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, use_fast=True)

    @staticmethod
    def _pooled(output) -> torch.Tensor:
        # transformers 5.x returns BaseModelOutputWithPooling, not a bare tensor.
        if isinstance(output, torch.Tensor):
            return output
        if output.pooler_output is None:
            raise ValueError("pooler_output should not be None for CLIP models.")
        return output.pooler_output

    @torch.no_grad()
    def encode_images(self, frames: list[np.ndarray]) -> torch.Tensor:
        """CLIP-embed a list of HWC uint8 RGB frames -> ``(T, 512)`` on CPU."""
        from PIL import Image

        images = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        return self._pooled(self._model.get_image_features(**inputs)).detach().cpu()

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """CLIP-embed a task string -> ``(512,)`` on CPU."""
        inputs = self._processor.tokenizer(
            [text], return_tensors="pt", padding=True, truncation=True
        ).to(self._device)
        return self._pooled(self._model.get_text_features(**inputs)).detach().cpu()[0]


class ProgressScorer:
    """Rolls a window of recent frames through CLIP+SARM into a progress reward.

    Online use is causal: SARM was trained on a frame window centred on the target
    frame, but we only ever have the past, so we buffer the most recent
    ``n_obs_steps + 1`` frames (sampled ~``frame_gap`` apart by the caller's poll
    cadence) and score the newest one. Good enough for a done-detector.
    """

    def __init__(self, model: torch.nn.Module, config: SARMConfig, encoder: ClipEncoder) -> None:
        self._model = model
        self._config = config
        self._encoder = encoder
        self._frames: deque[np.ndarray] = deque(maxlen=config.n_obs_steps + 1)
        self._text_emb: torch.Tensor | None = None

    def set_task(self, task: str) -> None:
        self._text_emb = self._encoder.encode_text(task)

    def reset(self) -> None:
        self._frames.clear()

    def push(self, frame_rgb: np.ndarray) -> None:
        self._frames.append(frame_rgb)

    @property
    def ready(self) -> bool:
        return bool(self._frames) and self._text_emb is not None

    def progress(self) -> float:
        """Progress reward in ``[0, 1]`` for the newest buffered frame."""
        if not self.ready:
            raise RuntimeError("scorer needs a task and at least one frame")
        frames = list(self._frames)
        video = self._encoder.encode_images(frames).unsqueeze(0)  # (1, T, 512)
        text = self._text_emb.unsqueeze(0)  # (1, 512)
        reward = self._model.calculate_rewards(
            text, video, state_features=None, frame_index=len(frames) - 1, head_mode="sparse"
        )
        return float(np.asarray(reward).reshape(-1)[-1])
