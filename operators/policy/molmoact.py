"""Shared MolmoAct2 wiring for train.py and run.py: default checkpoint and the
drop-the-slider rule.

MolmoAct2's SO-101 checkpoint is a six-DOF arm with no slider, but the leslider
wire contract carries a 7th ``slider.vel`` field (see ``utilities/rest_pose.py``).
So it is dropped end to end: training slices the column out of tensors, feature
metadata, and norm stats (:class:`SliderDroppedDataset`); inference feeds the six
arm ``.pos`` and pins ``slider.vel = 0`` (the ``move_to`` operator owns the slider).
"""
from __future__ import annotations

from typing import Any

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

from utilities.rest_pose import ARM_POS_KEYS

# Six-DOF, two cameras; ships the SO-100/101 joint-frame correction, so zero-shot
# deployment needs no extra flags.
DEFAULT_CHECKPOINT = "lerobot/MolmoAct2-SO100_101-LeRobot"

# The six arm .pos MolmoAct2 controls, in load-bearing wire order.
ACTION_NAMES: tuple[str, ...] = ARM_POS_KEYS


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
    if getattr(config, "image_keys", None):
        return list(config.image_keys)
    from lerobot.configs import FeatureType

    return [key for key, feat in config.input_features.items() if feat.type == FeatureType.VISUAL]
