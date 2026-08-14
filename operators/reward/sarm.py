"""Shared SARM reward-model wiring for the candy-shop reward operator.

Everything ``run.py`` and ``debug_reward.py`` need to agree on lives here so the
two never drift: the default checkpoint, the single image key, the completion
rule, and the online progress scorer. Training itself is lerobot's
``lerobot-train`` (see ``skypilot.yaml``), not a loop of ours.

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
and feed its image/text features into ``SARMRewardModel.calculate_rewards``, along
with the arm's normalized ``observation.state``.

State is **not** optional, though this module used to treat it as such. Passing
``state_features=None`` makes ``calculate_rewards`` substitute zeros, and measured
against a real checkpoint that flattens the output into a single generic ramp:
three different windows scored 0.11/0.14/0.14 rising to 0.27/0.29/0.37 — nearly the
same curve regardless of what the arm was doing — while the same windows *with*
state tracked ground truth closely (0.005 -> 0.558 against a 0.000 -> 0.632 target).
Progress is substantially a function of arm pose, which is unsurprising for a pick.
"""
from __future__ import annotations

import logging
import pathlib
from collections import deque

import numpy as np
import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards import SARMConfig, make_reward_model
from lerobot.utils.constants import OBS_IMAGES

logger = logging.getLogger(__name__)

# Trained by `lerobot-train` (see operators/reward/skypilot.yaml); there is no released
# candy-shop SARM checkpoint.
CHECKPOINT_ROOT = "outputs/sarm-candy-v2/checkpoints"


def latest_checkpoint(root: str = CHECKPOINT_ROOT) -> str:
    """Newest ``<root>/<step>/pretrained_model``, by step number.

    Not the ``last`` symlink lerobot maintains: training writes to a bucket mounted
    MOUNT_CACHED, object storage has no symlinks, and the call that creates it raises
    EIO (see operators/reward/train_fast.py). Sorting the numbered directories is
    equivalent and works whether the checkpoints were written to the mount or synced
    down with `aws s3 sync`.
    """
    base = pathlib.Path(root)
    steps = sorted((d for d in base.glob("[0-9]*") if (d / "pretrained_model").is_dir()),
                   key=lambda d: int(d.name))
    if not steps:
        return str(base / "last" / "pretrained_model")  # nothing pulled yet; fail loudly on load
    return str(steps[-1] / "pretrained_model")


# Pinned, NOT latest_checkpoint(): this run peaked early and then degraded. Measured
# with reward-debug over 40 episodes spanning all five tasks, at threshold 0.7:
#   step  4000 -> 40/40 fired, median peak 0.96, early-quarter 0.07   <- best
#   step  8000 ->  0/10 fired, median peak 0.52, early-quarter 0.38
#   step 12000 ->  0/10 fired, median peak 0.56, early-quarter 0.41
# The regression starts exactly at the job-6 resume boundary, so it is either
# overfitting past ~3 epochs or the resume perturbing optimizer/scheduler state —
# unresolved. Re-measure before repointing this at a later checkpoint.
DEFAULT_CHECKPOINT = "outputs/sarm-candy-v2/checkpoints/004000/pretrained_model"

# The CLIP encoder SARM was trained against (see processor_sarm.py). SARM eats
# CLIP embeddings, so inference must use the exact same encoder.
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def image_key_for(camera: str) -> str:
    """The dataset/observation image key SARM reads (single camera)."""
    return f"{OBS_IMAGES}.{camera}"


class StateNormalizer:
    """MEAN_STD normalization for ``observation.state``, read from the checkpoint.

    Training normalizes state through the preprocessor's normalizer step, whose stats
    ship next to the weights. Rebuilding that step needs dataset metadata we do not
    have online, so read the two tensors we actually need straight out of the saved
    safetensors instead. SARM's ``normalization_mapping`` puts STATE on MEAN_STD.
    """

    KEY = "observation.state"

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self._mean = mean
        # A joint that never moves has std 0; guard so it maps to 0 rather than inf.
        self._std = torch.where(std > 1e-6, std, torch.ones_like(std))

    @classmethod
    def from_checkpoint(cls, checkpoint: str) -> "StateNormalizer | None":
        """Load the state stats, or None if the checkpoint has no normalizer file."""
        from safetensors.torch import load_file

        path = pathlib.Path(checkpoint)
        files = sorted(path.glob("*normalizer_processor.safetensors"))
        for file in files:
            tensors = load_file(str(file))
            if f"{cls.KEY}.mean" in tensors and f"{cls.KEY}.std" in tensors:
                logger.info("[reward] loaded state normalizer from %s", file.name)
                return cls(tensors[f"{cls.KEY}.mean"].float(), tensors[f"{cls.KEY}.std"].float())
        logger.warning("[reward] no state normalizer in %s; progress will be degraded", checkpoint)
        return None

    def __call__(self, state: np.ndarray) -> torch.Tensor:
        return (torch.as_tensor(state, dtype=torch.float32) - self._mean) / self._std


def _disable_nested_tensor_fast_path(model: torch.nn.Module) -> None:
    """Stop ``nn.TransformerEncoder`` taking its nested-tensor path.

    Given a ``src_key_padding_mask`` the encoder tries to pack the batch into a nested
    tensor, which calls ``aten::_nested_tensor_from_mask_left_aligned`` — unimplemented
    on MPS, so serving on an Apple GPU dies mid-poll with NotImplementedError. The path
    is purely a batching optimization for ragged padded batches; online we score one
    window at a time, so there is nothing for it to pack and disabling it costs nothing.
    Cheaper and less surprising than PYTORCH_ENABLE_MPS_FALLBACK, which would silently
    bounce that op to CPU on every single poll.
    """
    # torch checks `use_nested_tensor` on the instance; `enable_nested_tensor` is only the
    # constructor argument, and which one the instance carries has moved between versions.
    # Clear whichever is present so this keeps working across upgrades.
    disabled = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.TransformerEncoder):
            continue
        for attr in ("use_nested_tensor", "enable_nested_tensor"):
            if getattr(module, attr, False):
                setattr(module, attr, False)
                disabled += 1
    if disabled:
        logger.debug("[reward] disabled the nested-tensor fast path on %d encoder(s)", disabled)


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
    _disable_nested_tensor_fast_path(model)
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


class DoneRule:
    """The completion test: progress at/above ``threshold`` for ``hold_s``.

    Shared by the live operator and the offline debug driver so a threshold
    calibrated by one means the same thing to the other. Held in polls rather
    than wall-clock seconds because that is what both loops actually count —
    the tick count is derived from the poll interval once, here.
    """

    def __init__(self, threshold: float, hold_s: float, eval_interval_s: float) -> None:
        self.threshold = threshold
        self.hold_ticks = max(1, round(hold_s / eval_interval_s))
        self._held = 0

    def reset(self) -> None:
        self._held = 0

    @property
    def held(self) -> int:
        return self._held

    def push(self, progress: float) -> bool:
        """Feed one poll's progress; True once it has held above threshold long enough."""
        self._held = self._held + 1 if progress >= self.threshold else 0
        return self._held >= self.hold_ticks


class ProgressScorer:
    """Rolls a window of recent frames through CLIP+SARM into a progress reward.

    Online use is causal. SARM trains on a bidirectional window
    ``[-2g, -g, 0, +g, +2g]`` around a target frame (``compute_absolute_indices``),
    which we cannot build live because the last two slots are in the future. Instead
    we buffer the most recent ``n_obs_steps + 1`` frames, sampled ~``frame_gap``
    apart by the caller's poll cadence, and read the *last* slot: that window has
    the same shape training used, just shifted back so its centre sits at
    ``t - 2g``, which puts the newest frame exactly where training put ``+2g``.

    Short buffers repeat the oldest frame rather than passing a stub window, since
    training pads the same way — it clamps out-of-bounds indices to the episode
    boundary, duplicating that frame.
    """

    def __init__(self, model: torch.nn.Module, config: SARMConfig, encoder: ClipEncoder,
                 state_normalizer: "StateNormalizer | None" = None) -> None:
        self._model = model
        self._config = config
        self._encoder = encoder
        self._normalizer = state_normalizer
        window = config.n_obs_steps + 1
        self._frames: deque[np.ndarray] = deque(maxlen=window)
        self._states: deque[torch.Tensor] = deque(maxlen=window)
        self._text_emb: torch.Tensor | None = None

    def set_task(self, task: str) -> None:
        self._text_emb = self._encoder.encode_text(task)

    def reset(self) -> None:
        self._frames.clear()
        self._states.clear()

    def push(self, frame_rgb: np.ndarray, state: np.ndarray | None = None) -> None:
        """Buffer one poll. ``state`` is the raw ``observation.state`` vector."""
        self._frames.append(frame_rgb)
        if state is not None and self._normalizer is not None:
            self._states.append(self._normalizer(state))

    @property
    def ready(self) -> bool:
        return bool(self._frames) and self._text_emb is not None

    def progress(self) -> float:
        """Progress reward in ``[0, 1]`` for the newest buffered frame."""
        if not self.ready:
            raise RuntimeError("scorer needs a task and at least one frame")
        window = self._frames.maxlen
        frames = list(self._frames)
        frames = [frames[0]] * (window - len(frames)) + frames
        video = self._encoder.encode_images(frames).unsqueeze(0)  # (1, T, 512)
        text = self._text_emb.unsqueeze(0)  # (1, 512)

        # Same boundary padding as the frames, so state and video stay aligned slot
        # for slot. Missing state falls back to zeros, which is what the model would
        # have received anyway — but the curve goes flat, so it is worth a warning.
        state = None
        if len(self._states) == len(self._frames):
            states = list(self._states)
            states = [states[0]] * (window - len(states)) + states
            state = torch.stack(states).unsqueeze(0)  # (1, T, state_dim)

        reward = self._model.calculate_rewards(
            text, video, state_features=state, frame_index=window - 1, head_mode="sparse"
        )
        return float(np.asarray(reward).reshape(-1)[-1])
