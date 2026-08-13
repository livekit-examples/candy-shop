"""``lerobot-train`` with the SARM image-encoder speed patch applied.

Not a training loop — the loop is entirely lerobot's, which is the point (ours
dropped the gradient clip its optimizer preset asks for and cost a 17h run). This
only installs :mod:`operators.reward.clip_patch` before handing off, so the patch
survives the ``uv sync`` that would overwrite an edit to site-packages.

Takes the same arguments as ``lerobot-train``; see ``operators/reward/skypilot.yaml``.
"""
from __future__ import annotations

from operators.reward.clip_patch import apply


def cli() -> None:
    apply()
    from lerobot.scripts.lerobot_train import main

    main()


if __name__ == "__main__":
    cli()
