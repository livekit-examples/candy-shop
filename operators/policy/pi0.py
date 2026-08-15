"""Shared pi0 wiring for run.py: the checkpoints, the wire contract, and the
camera mapping.

Training has no SmolVLA-style :class:`SliderDroppedDataset`: pi0 pads state and
action to ``max_state_dim``/``max_action_dim`` (32), so the 7th ``slider.vel``
column rides along harmlessly and ``skypilot_pi0.yaml`` feeds the dataset
unmodified. That decision lands here — the checkpoint's norm stats are seven
wide, so inference has to feed all seven columns back and drop the 7th on the
way out.
"""
from __future__ import annotations

import logging
from typing import Any

from lerobot.utils.constants import OBS_IMAGES

from shared.rest_pose import ALL_ACTION_KEYS, ARM_POS_KEYS

# Fine-tune starting point: PaliGemma 2B + a gemma_300m action expert. Like the
# SmolVLA base it carries no rig normalization stats, so it is a training start,
# not a servable checkpoint. Note it also needs a token for the gated PaliGemma
# repo — at inference too, because the preprocessor tokenizes with it.
BASE_CHECKPOINT = "lerobot/pi0_base"

# Where skypilot_pi0.yaml writes. run.py has no default checkpoint: what it
# serves is always named explicitly (`--checkpoint` / POLICY_CHECKPOINT).
DEFAULT_OUTPUT_DIR = "outputs/pi0-candy"

# The six arm .pos the policy commands, in load-bearing wire order. The model
# emits seven; the 7th is slider.vel, which run.py pins to 0 instead of reading
# (the slider belongs to the move_to operator, and it was constant 0 in the
# dataset, so that column's std is 0 and its unnormalized value is meaningless).
ACTION_NAMES: tuple[str, ...] = ARM_POS_KEYS

# What observation.state has to carry: the six arm .pos *plus* slider.vel. Six
# would not broadcast against the checkpoint's seven-wide MEAN_STD stats.
STATE_NAMES: tuple[str, ...] = ALL_ACTION_KEYS

logger = logging.getLogger(__name__)


def _rename_sources(preprocessor: Any) -> list[str]:
    """The ``observation.images.*`` keys the saved pipeline renames into pi0's slots.

    Not ``config.input_features``: pi0's ``validate_features`` fills that with all
    three openpi slots, including the ``right_wrist_0_rgb`` we never wire, so
    matching against it would leave a third key to place with only two cameras.
    The pipeline's first step is the rename recorded at training time, whose
    *source* keys are the dataset's own camera names — the ones we can actually
    match, and the ones the pipeline expects to be fed.
    """
    for step in getattr(preprocessor, "steps", []):
        rename_map = getattr(step, "rename_map", None)
        if rename_map:
            return [key for key in rename_map if key.startswith(OBS_IMAGES)]
    return []


def resolve_camera_map(preprocessor: Any, primary: str, wrist: str) -> dict[str, str]:
    """Map the keys the policy consumes onto physical camera names.

    Match on the name first, or a positional map would feed the wrist image in as
    the overhead view (the dataset's key order is alphabetical, arm before
    overhead). Keys naming no camera we have fall back to position: primary
    (overhead) first, wrist second.
    """
    physical = [primary, wrist]
    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    for key in _rename_sources(preprocessor):
        name = key.rsplit(".", 1)[-1]
        if name in physical:
            mapping[key] = name
        else:
            unmatched.append(key)

    if not mapping and not unmatched:
        raise RuntimeError(
            "checkpoint's preprocessor renames no image keys, so there is nothing to "
            "feed the cameras into; was it trained with --rename_map?"
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
