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


def require_pretrained_weights() -> None:
    """Fail before training if ``--policy.path``'s weights cannot be fetched.

    ``PI05Pytorch.from_pretrained`` (modeling_pi05.py:811, and pi0's at :846) wraps the
    safetensors fetch in a bare ``except Exception`` that prints "Returning model without
    loading pretrained weights" and returns the *randomly initialised* model. Training
    then runs to completion at a normal step rate and exits 0, so a fine-tune that
    silently became a from-scratch run is indistinguishable from a good one until you
    evaluate it. That is exactly what happened here: Nebius cannot route to the AWS
    ranges behind ``us.aws.cdn.hf.co``, the fetch failed, and 40 steps trained happily
    from noise.

    Resolving the same file up front turns that into a loud failure at startup. Only the
    fetch is checked — if it succeeds here it is cached, so the load inside lerobot hits
    the cache and cannot take the fallback.
    """
    import sys

    argv = sys.argv[1:]
    path = None
    for i, token in enumerate(argv):
        if token == "--policy.path" and i + 1 < len(argv):
            path = argv[i + 1]
        elif token.startswith("--policy.path="):
            path = token.split("=", 1)[1]
    if path is None:
        return

    from transformers.utils import cached_file

    try:
        cached_file(path, "model.safetensors")
    except Exception as exc:
        raise RuntimeError(
            f"could not fetch model.safetensors for --policy.path={path!r}: {exc}. "
            "Refusing to start: lerobot would swallow this and train from random "
            "initialisation. Stage the weights into the bucket and point --policy.path "
            "at the local copy."
        ) from exc

    logger.info("[patches] pretrained weights for %s resolved", path)


def relative_action_step(exclude_joints: tuple[str, ...], action_names=None):
    """The relative step, taught to take its reference from the most recent pose.

    DiT runs ``n_obs_steps=2``, so at training ``observation.state`` is ``[B, 2, dim]``;
    lerobot's step was written for pi0/pi0.5, where it is ``[B, dim]``, and broadcasting a
    two-step history across a 32-step action chunk raises. At inference the same key is
    ``[B, dim]`` -- the policy keeps its own deque -- so both shapes have to work.

    Only the offset is computed from the reduced state; the transition passed downstream
    keeps the full history, which the model needs.

    Built here rather than inline so it can be exercised directly; see
    ``operators/policy/check_relative_actions.py``.
    """
    from lerobot.processor import RelativeActionsProcessorStep, TransitionKey
    from lerobot.utils.constants import OBS_STATE

    class LatestPoseRelativeActions(RelativeActionsProcessorStep):
        def __call__(self, transition):
            observation = transition.get(TransitionKey.OBSERVATION) or {}
            state = observation.get(OBS_STATE)
            if state is None or state.ndim < 3:
                return super().__call__(transition)

            reduced = dict(transition)
            reduced[TransitionKey.OBSERVATION] = {**observation, OBS_STATE: state[:, -1]}
            converted = super().__call__(reduced)

            result = dict(transition)
            result[TransitionKey.ACTION] = converted.get(TransitionKey.ACTION)
            return result

    return LatestPoseRelativeActions(
        enabled=True, exclude_joints=list(exclude_joints), action_names=action_names
    )


def enable_relative_actions(exclude_joints: tuple[str, ...] = ()) -> None:
    """Train multi_task_dit on action deltas instead of absolute poses.

    multi_task_dit is absolute-only: it has no ``use_relative_actions`` flag and its
    pipeline is ``rename -> batch -> tokenize -> device -> normalize``. The two steps
    that do the work are policy-agnostic and live in ``lerobot.processor``, so this
    splices them into openpi's order, the one pi0/pi0.5/pi0-FAST/GR00T all use:

        raw -> relative -> normalize -> model -> unnormalize -> absolute

    Doing it in the pipeline rather than by rewriting the dataset is not a stylistic
    choice. ``to_relative_actions`` broadcasts *one* state across the whole chunk, so
    every action is a delta from the pose at chunk start -- which is the pose inference
    actually knows. A rewritten dataset can only store ``action[t] - state[t]``, each
    action relative to its own timestep, because row ``t+k`` belongs to 32 different
    chunks with 32 different reference poses. Serving that requires integrating deltas
    and assuming the follower reached every previous command exactly; on this rig it
    did not, and the error tracked state drift precisely.

    The pair is stateful: the relative step caches the state it subtracted and the
    absolute step reads it back. Nothing has to be threaded through serving, because
    ``PolicyProcessorPipeline`` serializes both steps into the checkpoint and
    ``make_pre_post_processors`` calls ``_reconnect_relative_absolute_steps`` when it
    loads one. run.py needs no change: it calls ``_pre`` once per chunk and ``_post``
    per action, so every action converts back against the same chunk-start pose.

    NOTE: the normalizer must be fed relative-action stats (``lerobot-edit-dataset
    --operation.type recompute_stats --operation.relative_action true``). Deltas
    normalized against absolute statistics are centred nowhere near zero.
    """
    from lerobot.policies.multi_task_dit import processor_multi_task_dit as module
    from lerobot.processor import AbsoluteActionsProcessorStep

    original = module.make_multi_task_dit_pre_post_processors
    if getattr(original, "_relative", False):
        return

    def with_relative_actions(config, dataset_stats=None):
        preprocessor, postprocessor = original(config, dataset_stats)

        relative = relative_action_step(
            exclude_joints, getattr(config, "action_feature_names", None)
        )
        absolute = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative)

        # Position by class rather than index: the pipeline these land in is lerobot's,
        # and a step added upstream would silently shift any hardcoded offset.
        def index_of(steps, name, default):
            for i, step in enumerate(steps):
                if type(step).__name__ == name:
                    return i
            logger.warning("[patches] %s not found in pipeline; inserting at %d", name, default)
            return default

        pre_steps = list(preprocessor.steps)
        pre_steps.insert(index_of(pre_steps, "NormalizerProcessorStep", len(pre_steps)), relative)
        preprocessor.steps = pre_steps

        post_steps = list(postprocessor.steps)
        post_steps.insert(index_of(post_steps, "UnnormalizerProcessorStep", -1) + 1, absolute)
        postprocessor.steps = post_steps

        logger.info("[patches] relative actions on (excluding %s)", list(exclude_joints) or "nothing")
        return preprocessor, postprocessor

    with_relative_actions._relative = True
    module.make_multi_task_dit_pre_post_processors = with_relative_actions
