"""Speed patch for lerobot's SARM image encoder.

``SARMEncodingProcessorStep._encode_images_batch`` walks the batch one frame at a
time and converts each through ``numpy -> tobytes() -> PIL.Image`` before handing
it to the CLIP processor. Profiling a real step (batch 64, 8 frames/sample = 512
frames) put ``ndarray.tobytes`` at 45% of the whole preprocessor — 6.4s of 14.2s
across three batches — with the actual resize at 1.25s and the CLIP forward
barely visible. The H100 sat at 0% while one core did container conversion.

The processor is built with ``use_fast=True``, so it is a
``CLIPImageProcessorFast`` and already accepts channel-first uint8 tensors. The
PIL trip buys nothing. This replaces the per-frame loop with one vectorized
conversion and feeds tensors straight through.

Semantics are preserved deliberately, including the original's quirks:
``(img * 255).astype(uint8)`` **truncates**, so this uses ``.to(torch.uint8)``
rather than rounding, and the ``max() <= 1.0`` test stays per-batch-of-frames.
Anything the fast path does not recognise falls back to the original method, so a
lerobot upgrade that changes the input contract degrades to upstream speed rather
than breaking. ``verify()`` asserts bit-identical embeddings against the original.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

_ORIGINAL = None


def _to_uint8_chw(images) -> torch.Tensor:
    """(B, T, C, H, W) -> (B*T, 3, H, W) uint8, matching the original conversion."""
    x = torch.as_tensor(np.ascontiguousarray(images)) if isinstance(images, np.ndarray) else images
    x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
    if x.shape[1] == 1:  # single channel -> replicate, as the original does
        x = x.repeat(1, 3, 1, 1)
    if x.dtype != torch.uint8:
        # Truncation, not rounding: the original calls .astype(np.uint8).
        x = (x * 255).to(torch.uint8) if float(x.max()) <= 1.0 else x.to(torch.uint8)
    return x


@torch.no_grad()
def _encode_images_batch_fast(self, images) -> torch.Tensor:
    batch_size, seq_length = images.shape[0], images.shape[1]
    try:
        frames = _to_uint8_chw(images)
    except Exception:  # unexpected layout/dtype — let upstream handle it
        logger.warning("[clip_patch] falling back to upstream encoder", exc_info=True)
        return _ORIGINAL(self, images)

    embeddings = []
    for start in range(0, frames.shape[0], self.config.clip_batch_size):
        chunk = frames[start : start + self.config.clip_batch_size]
        inputs = self.clip_processor(images=chunk, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self.clip_model.get_image_features(**inputs)
        if not isinstance(output, torch.Tensor):
            output = output.pooler_output
            if output is None:
                raise ValueError("pooler_output should not be None for CLIP models.")
        out = output.detach().cpu()
        embeddings.append(out.unsqueeze(0) if out.dim() == 1 else out)

    return torch.cat(embeddings).reshape(batch_size, seq_length, -1)


def apply() -> None:
    """Swap in the fast encoder. Idempotent."""
    global _ORIGINAL
    from lerobot.rewards.sarm.processor_sarm import SARMEncodingProcessorStep

    if _ORIGINAL is not None:
        return
    _ORIGINAL = SARMEncodingProcessorStep._encode_images_batch
    SARMEncodingProcessorStep._encode_images_batch = _encode_images_batch_fast
    logger.info("[clip_patch] SARM image encoder patched (skips the per-frame PIL conversion)")


def verify(step, images) -> None:
    """Assert the fast path reproduces upstream's embeddings exactly."""
    if _ORIGINAL is None:
        raise RuntimeError("call apply() first")
    got = _encode_images_batch_fast(step, images)
    want = _ORIGINAL(step, images)
    if not torch.equal(got, want):
        delta = (got - want).abs().max().item()
        raise AssertionError(f"patched encoder differs from upstream (max |delta| = {delta:.3e})")
