"""``lerobot-train`` for multi_task_dit, with our patches installed first.

Separate from train.py, which is the hand-written SmolVLA trainer: the DiT loop is
entirely lerobot's, so this only installs patches and hands off.

Not a training loop — the loop is entirely lerobot's. This exists so the patches in
:mod:`shared.lerobot_patches` are applied before it starts:

* **symlink tolerance**, without which the run dies at its first bucket checkpoint with
  ``OSError: [Errno 5]`` — object storage has no symlinks and lerobot repoints
  ``checkpoints/last`` after every save;
* **the weight-fetch guard**, which refuses to start when ``--policy.path`` names weights
  it cannot resolve. DiT trains from scratch (only the CLIP encoders are pretrained, and
  transformers fetches those itself), so the guard is a no-op unless you fine-tune from an
  existing checkpoint — which is exactly when it matters, because lerobot otherwise
  swallows a failed fetch and trains from random initialisation.

Takes the same arguments as ``lerobot-train``; see operators/policy/skypilot_dit.yaml for
the configuration that is actually in use.

    uv run policy-train --policy.type=multi_task_dit --dataset.repo_id=... --steps=30000
"""
from __future__ import annotations

import os

from shared.lerobot_patches import (
    drop_slider_vel,
    enable_relative_actions,
    require_pretrained_weights,
    tolerate_missing_symlinks,
)

# Joints kept absolute when RELATIVE_ACTIONS is on. The gripper commands an aperture
# rather than a motion, and slider.vel is a velocity the policy pins to 0 -- neither is
# a pose a delta means anything against.
RELATIVE_EXCLUDE = ("gripper.pos", "slider.vel")


def cli() -> None:
    tolerate_missing_symlinks()
    require_pretrained_weights()
    # Six wide, matching SmolVLA. The DiT path was built without the slider drop and
    # trained on the raw seven-column dataset, so every checkpoint before dit-6dim
    # carries a slider.vel dimension that is constant 0 -- inert, but never intended:
    # the slider belongs to move_to and run.py pins it to 0 at inference anyway.
    drop_slider_vel()
    # Env rather than --policy.*: multi_task_dit's config has no relative-action field,
    # and draccus rejects flags its dataclass does not declare.
    if os.environ.get("RELATIVE_ACTIONS", "").lower() in ("1", "true", "yes"):
        enable_relative_actions(exclude_joints=RELATIVE_EXCLUDE)
    from lerobot.scripts.lerobot_train import main

    main()


if __name__ == "__main__":
    cli()
