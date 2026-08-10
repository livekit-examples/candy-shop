"""Shared MolmoAct2 wiring for the candy-shop policy operator.

Everything both ``train.py`` and ``run.py`` need to agree on lives here so the
two never drift: the default checkpoint, the *drop-the-slider* rule, and the
norm-stats side-file.

Why drop the slider
-------------------
The leslider wire contract carries **7** fields — six arm ``.pos`` plus one
``slider.vel`` (see ``utilities/rest_pose.py``). MolmoAct2's SO-101 checkpoint
only knows the **six-DOF arm**; it has no concept of a linear slider. So both
training and inference operate on the six arm dims only:

* **training** — a dataset recorded on this rig has 7-dim ``observation.state``
  and ``action`` vectors; we slice the ``slider.vel`` column out of the tensors,
  the feature metadata, and the normalization stats before the policy ever sees
  them (:class:`SliderDroppedDataset`).
* **inference** — the operator feeds the six arm ``.pos`` as state, and pins
  ``slider.vel = 0`` on the wire (the ``move_to`` operator owns the slider).

Norm stats
----------
MolmoAct2's normalizer is built from the dataset quantile stats. ``train.py``
persists those stats into the checkpoint via the saved processor pipeline
(``preprocessor.save_pretrained``), so ``run.py`` just reloads the processors
from the checkpoint dir — no separate stats file needed. Released checkpoints
ship their processors the same way.
"""
from __future__ import annotations

from typing import Any

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

from utilities.rest_pose import ARM_POS_KEYS

# The default LeRobot-format MolmoAct2 checkpoint for the SO-101 arm. Six-DOF,
# two cameras (primary + wrist). Ships the SO-100/101 joint-frame correction in
# its own config, so zero-shot deployment needs no extra flags.
DEFAULT_CHECKPOINT = "lerobot/MolmoAct2-SO100_101-LeRobot"

# The dims MolmoAct2 controls: the six arm .pos, in the load-bearing wire order.
ACTION_NAMES: tuple[str, ...] = ARM_POS_KEYS


def _keep_indices(names: list[str]) -> list[int]:
    """Indices of the columns we keep — everything that is not a ``.vel`` field."""
    return [i for i, name in enumerate(names) if not str(name).endswith(".vel")]


class SliderDroppedDataset(LeRobotDataset):
    """A :class:`LeRobotDataset` with the ``slider.vel`` column removed.

    Slices ``observation.state`` and ``action`` — in the returned samples, in
    ``meta.features`` (shape + names), and in ``meta.stats`` — down to the six
    arm dims, so the whole training stack (feature derivation, quantile
    normalization, the flow-matching loss) sees a 6-DOF arm and nothing else.
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
        # Stats may be torch tensors or numpy arrays depending on the dataset;
        # fancy-indexing the last axis works for both. Only touch per-dimension
        # stats (last axis == the feature dim), never scalars like "count".
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
