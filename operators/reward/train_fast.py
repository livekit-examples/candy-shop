"""``lerobot-train`` with the SARM image-encoder speed patch applied.

Not a training loop — the loop is entirely lerobot's, which is the point (ours
dropped the gradient clip its optimizer preset asks for and cost a 17h run). This
only installs :mod:`operators.reward.clip_patch` before handing off, so the patch
survives the ``uv sync`` that would overwrite an edit to site-packages.

Takes the same arguments as ``lerobot-train``; see ``operators/reward/skypilot.yaml``.
"""
from __future__ import annotations

import logging

from operators.reward.clip_patch import apply

logger = logging.getLogger(__name__)


def _tolerate_missing_symlinks() -> None:
    """Let checkpointing survive an output_dir that cannot hold symlinks.

    After every save lerobot repoints ``checkpoints/last`` at the new step with
    ``Path.symlink_to``. Our ``--output_dir`` is the Nebius bucket mounted
    MOUNT_CACHED, and object storage has no symlinks, so the call raises
    ``OSError: [Errno 5] Input/output error`` and kills the run — which it did, at
    step 4000 of 20000, after four and a half hours. The checkpoint itself is
    already written and complete by then; only the convenience pointer fails.

    So downgrade that one failure to a warning. Callers must resolve the newest
    checkpoint by step number instead of trusting ``last`` (see sarm.DEFAULT_CHECKPOINT).
    """
    from lerobot.common import train_utils
    from lerobot.scripts import lerobot_train

    original = train_utils.update_last_checkpoint

    def tolerant(checkpoint_dir):
        try:
            return original(checkpoint_dir)
        except OSError as exc:
            logger.warning("[train_fast] could not update the 'last' symlink (%s); "
                           "the checkpoint itself is saved, continuing", exc)
            return checkpoint_dir

    # lerobot_train does `from ...train_utils import update_last_checkpoint`, binding the
    # name in its own namespace at import, so patching train_utils alone would miss the
    # call that actually crashes (lerobot_train.py:703). Rebind both.
    train_utils.update_last_checkpoint = tolerant
    lerobot_train.update_last_checkpoint = tolerant


def cli() -> None:
    apply()
    _tolerate_missing_symlinks()
    from lerobot.scripts.lerobot_train import main

    main()


if __name__ == "__main__":
    cli()
