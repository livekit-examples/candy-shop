"""Shared SmolVLA wiring for train.py and run.py: the checkpoints, the
drop-the-slider rule, and the camera mapping.

The policy drives the arm only — the ``move_to`` operator owns the slider — but
the leslider wire contract carries a 7th ``slider.vel`` field (see
``shared/rest_pose.py``). So it is dropped end to end: training slices the column
out of tensors, feature metadata, and norm stats (:class:`SliderDroppedDataset`);
inference feeds the six arm ``.pos`` and pins ``slider.vel = 0``.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

from shared.rest_pose import ARM_POS_KEYS

# Fine-tune starting point: the 500M SmolVLA base (SmolVLM2 backbone + flow-
# matching action expert). It carries no normalization stats and its image keys
# are placeholders (camera1..3), so it is a training start, not a servable
# checkpoint: there is no zero-shot path, fine-tune on a rig dataset first.
BASE_CHECKPOINT = "lerobot/smolvla_base"

# Where policy-train writes. run.py has no default checkpoint: what it serves is
# always named explicitly (`--checkpoint` / POLICY_CHECKPOINT).
DEFAULT_OUTPUT_DIR = "outputs/smolvla-candy"

# The six arm .pos the policy controls, in load-bearing wire order.
ACTION_NAMES: tuple[str, ...] = ARM_POS_KEYS

logger = logging.getLogger(__name__)


def _keep_indices(names: list[str]) -> list[int]:
    """Indices of the columns we keep — everything that is not a ``.vel`` field."""
    return [i for i, name in enumerate(names) if not str(name).endswith(".vel")]


class SliderDroppedDataset(LeRobotDataset):
    """A :class:`LeRobotDataset` with the ``slider.vel`` column removed.

    Slices ``observation.state`` and ``action`` down to the six arm dims in the
    returned samples, ``meta.features``, and ``meta.stats``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._keep: dict[str, torch.Tensor] = {}
        for key in (OBS_STATE, ACTION):
            feature = self.meta.features.get(key)
            if feature is None or feature.get("names") is None:
                continue
            names = list(feature["names"])
            keep = _keep_indices(names)
            if len(keep) == len(names):
                continue  # nothing to drop
            self._keep[key] = torch.tensor(keep, dtype=torch.long)
            feature["names"] = [names[i] for i in keep]
            feature["shape"] = (len(keep),)
            self._slice_stats(key, keep, orig_dim=len(names))

    def _slice_stats(self, key: str, keep: list[int], *, orig_dim: int) -> None:
        stats = self.meta.stats.get(key) if self.meta.stats else None
        if not stats:
            return
        # Stats may be torch tensors or numpy arrays; fancy-indexing the last
        # axis works for both. Only touch per-dimension stats (last axis == the
        # feature dim), never scalars like "count".
        for stat_name, value in list(stats.items()):
            if getattr(value, "ndim", 0) >= 1 and getattr(value, "shape", (None,))[-1] == orig_dim:
                stats[stat_name] = value[..., keep]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        for key, keep in self._keep.items():
            if key in item and torch.is_tensor(item[key]):
                item[key] = item[key].index_select(-1, keep)
        return item


def resolve_image_keys(config: Any) -> list[str]:
    """The ordered ``observation.images.*`` keys the policy expects as input."""
    from lerobot.configs import FeatureType

    return [key for key, feat in config.input_features.items() if feat.type == FeatureType.VISUAL]


def resolve_camera_map(config: Any, primary: str, wrist: str) -> dict[str, str]:
    """Map the policy's image keys onto physical camera names.

    SmolVLA has no image-key override, so a checkpoint fine-tuned on this rig
    carries the dataset's own keys (``observation.images.overhead_camera``) in
    the dataset's order — which is alphabetical, arm before overhead. Match on
    the name first, or a positional map would feed the wrist image in as the
    overhead view. Keys naming no camera we have fall back to position: primary
    (overhead) first, wrist second.
    """
    physical = [primary, wrist]
    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    for key in resolve_image_keys(config):
        name = key.rsplit(".", 1)[-1]
        if name in physical:
            mapping[key] = name
        else:
            unmatched.append(key)

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
