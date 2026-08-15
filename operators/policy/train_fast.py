"""``lerobot-train`` for pi0, with the object-storage checkpoint patch applied.

Not a training loop — the loop is entirely lerobot's. This exists so the patches in
:mod:`shared.lerobot_patches` are installed first: the symlink tolerance, without
which the run dies at its first bucket checkpoint with ``OSError: [Errno 5]`` (how
the first pi0 smoke ended), and the weight-fetch guard, without which an unreachable
hub silently downgrades the fine-tune to a from-scratch run.

Unlike the reward launcher there is no CLIP patch here: that one targets SARM's
image encoder, which pi0 does not use.

Takes the same arguments as ``lerobot-train``; see operators/policy/skypilot_pi0.yaml.
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
