"""Shared multi_task_dit wiring for run.py: the wire contract and camera mapping.

This replaced the pi0/pi0.5 path after robot testing: every pi0.5 variant we trained
picked the wrong candy, and both DiT arms picked the right one. The cause is that the
five tasks differ only by instruction while the candies sit in four fixed bins, so the
vision encoder has to learn these wrappers in this scene -- which `train_expert_only`
forbids outright and `freeze_vision_encoder` did badly. multi_task_dit trains its CLIP
vision encoder at 0.1x LR. See operators/policy/RESULTS.md.

DiT differs from pi0 in three ways that matter here, each of which broke serving once:

* It is **queue-based**. `predict_action_chunk` reads observation deques that only
  `select_action` fills, so calling it directly raises "stack expects a non-empty
  TensorList".
* It **stacks the camera views** into one tensor before any resizing, so every image
  must already share a resolution -- the live wrist camera streams 360x480 against a
  dataset recorded at 480x640.
* It carries **no rename map**. pi0 renames the dataset's camera keys into openpi's
  slots at training time; DiT consumes `observation.images.*` as recorded.
"""
from __future__ import annotations

import logging
from typing import Any

from lerobot.utils.constants import OBS_IMAGES

from shared.rest_pose import ALL_ACTION_KEYS, ARM_POS_KEYS

POLICY_TYPE = "multi_task_dit"

# Where operators/policy/skypilot_dit.yaml writes. run.py has no default checkpoint:
# what it serves is always named explicitly (`--checkpoint` / POLICY_CHECKPOINT).
DEFAULT_OUTPUT_DIR = "outputs/dit-candy"

# The six arm .pos the policy commands, in load-bearing wire order. The model emits
# seven; the 7th is slider.vel, which run.py pins to 0 instead of reading (the slider
# belongs to the move_to operator, and it was constant 0 in the dataset).
ACTION_NAMES: tuple[str, ...] = ARM_POS_KEYS

# What observation.state has to carry: the six arm .pos *plus* slider.vel, matching the
# seven-wide state every checkpoint was trained against.
STATE_NAMES: tuple[str, ...] = ALL_ACTION_KEYS

logger = logging.getLogger(__name__)


def policy_image_keys(policy_config: Any) -> list[str]:
    """The ``observation.images.*`` keys the policy expects to be fed."""
    features = getattr(policy_config, "input_features", None) or {}
    return [key for key in features if str(key).startswith(OBS_IMAGES)]


def expected_image_hw(policy_config: Any) -> dict[str, tuple[int, int]]:
    """The (height, width) each image key was trained at.

    Live cameras need not match the dataset -- here the wrist streams 360x480 while every
    recorded frame is 480x640 -- and DiT stacks the views before resizing, so a mismatch
    raises rather than being silently rescaled.
    """
    features = getattr(policy_config, "input_features", None) or {}
    shapes: dict[str, tuple[int, int]] = {}
    for key, feature in features.items():
        shape = getattr(feature, "shape", None)
        if shape is not None and len(shape) == 3 and str(key).startswith(OBS_IMAGES):
            shapes[key] = (int(shape[1]), int(shape[2]))
    return shapes


def resolve_camera_map(policy_config: Any, primary: str, wrist: str) -> dict[str, str]:
    """Map the image keys the policy consumes onto physical camera names.

    Match on the name first, or a positional map would feed the wrist image in as the
    overhead view (the key order is alphabetical, arm before overhead). Keys naming no
    camera we have fall back to position: primary (overhead) first, wrist second.
    """
    physical = [primary, wrist]
    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    for key in policy_image_keys(policy_config):
        name = key.rsplit(".", 1)[-1]
        if name in physical:
            mapping[key] = name
        else:
            unmatched.append(key)

    if not mapping and not unmatched:
        raise RuntimeError(
            "checkpoint declares no observation.images.* input features, so there is "
            "nothing to feed the cameras into"
        )

    spare = [name for name in physical if name not in mapping.values()]
    if len(unmatched) > len(spare):
        raise RuntimeError(
            f"policy image keys {unmatched} do not name a wired camera and only {spare} are left "
            f"to assign by position (cameras: {physical})"
        )
    for key, name in zip(unmatched, spare):
        logger.warning("[policy] image key %s names no wired camera; feeding %s by position", key, name)
        mapping[key] = name
    return mapping
