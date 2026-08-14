"""Runtime patches applied before handing off to lerobot's trainer.

Kept here rather than beside one operator because the bug below is not specific to
either model — it is a property of writing checkpoints to an object-storage mount,
which both the reward and policy runs do. It first bit the SARM run, was fixed only
in the reward launcher, and then bit the pi0 run in exactly the same place.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def tolerate_missing_symlinks() -> None:
    """Let checkpointing survive an ``--output_dir`` that cannot hold symlinks.

    After every save lerobot repoints ``checkpoints/last`` with ``Path.symlink_to``.
    Our output dir is the Nebius bucket mounted MOUNT_CACHED, and object storage has
    no symlinks, so the call raises ``OSError: [Errno 5] Input/output error`` and takes
    the whole run down — the SARM run lost four and a half hours to it at step 4000,
    and the pi0 smoke died the same way at step 8. In both cases the checkpoint itself
    was already written and complete; only the convenience pointer failed.

    Downgrade that one failure to a warning. Callers must then resolve the newest
    checkpoint by step number rather than trusting ``last``; see
    ``operators.reward.sarm.latest_checkpoint``.
    """
    from lerobot.common import train_utils
    from lerobot.scripts import lerobot_train

    if getattr(train_utils.update_last_checkpoint, "_tolerant", False):
        return

    original = train_utils.update_last_checkpoint

    def tolerant(checkpoint_dir):
        try:
            return original(checkpoint_dir)
        except OSError as exc:
            logger.warning("[patches] could not update the 'last' symlink (%s); "
                           "the checkpoint itself is saved, continuing", exc)
            return checkpoint_dir

    tolerant._tolerant = True
    # lerobot_train does `from ...train_utils import update_last_checkpoint`, binding the
    # name in its own namespace at import time, so patching train_utils alone would miss
    # the call that actually crashes (lerobot_train.py:703). Rebind both.
    train_utils.update_last_checkpoint = tolerant
    lerobot_train.update_last_checkpoint = tolerant
