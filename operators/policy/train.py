"""``lerobot-train`` for multi_task_dit, with our patches installed first.

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

from shared.lerobot_patches import require_pretrained_weights, tolerate_missing_symlinks


def cli() -> None:
    tolerate_missing_symlinks()
    require_pretrained_weights()
    from lerobot.scripts.lerobot_train import main

    main()


if __name__ == "__main__":
    cli()
