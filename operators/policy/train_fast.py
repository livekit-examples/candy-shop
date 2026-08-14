"""``lerobot-train`` for pi0, with the object-storage checkpoint patch applied.

Not a training loop — the loop is entirely lerobot's. This exists only so the
symlink tolerance in :mod:`shared.lerobot_patches` is installed before lerobot
writes its first checkpoint to the bucket; without it the run dies at the first
save with ``OSError: [Errno 5]``, which is exactly how the first pi0 smoke ended.

Unlike the reward launcher there is no CLIP patch here: that one targets SARM's
image encoder, which pi0 does not use.

Takes the same arguments as ``lerobot-train``; see operators/policy/skypilot_pi0.yaml.
"""
from __future__ import annotations

from shared.lerobot_patches import tolerate_missing_symlinks


def cli() -> None:
    tolerate_missing_symlinks()
    from lerobot.scripts.lerobot_train import main

    main()


if __name__ == "__main__":
    cli()
