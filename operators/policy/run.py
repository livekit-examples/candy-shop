"""Picker operator: serve multi_task_dit (six-DOF arm) over the ``run_policy`` RPC.

DiT-only by choice, not by accident: on the arm, every pi0.5 variant picked the wrong
candy and both DiT arms picked the right one, so the multi-architecture serving path was
removed rather than left to rot. See operators/policy/RESULTS.md for the comparison, and
git history for the pi0/pi0.5/SmolVLA paths if they are ever wanted back.

Fine-tune one first (``skypilot_dit.yaml``); there is no servable stock checkpoint::

    uv run policy --checkpoint outputs/dit-orig/checkpoints/007500/pretrained_model

Checkpoints live in the bucket rather than beside the code, and lerobot's
``last`` symlink is never written (object storage has no symlinks — see
``shared/lerobot_patches.py``), so name a step directory explicitly::

    aws s3 sync s3://candy-shop/pi0-candy/ outputs/pi0-candy/ --profile nebius

Serving pulls the CLIP vision and text encoders from the Hub on first run; they are
ungated, so no token is needed (unlike the pi0 path, which required one for PaliGemma).
"""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
import json
import logging
import os
import pathlib
import time
from typing import Callable

import numpy as np
import torch

from livekit.portal import Observation, Operator, OperatorConfig, RpcError, RpcInvocationData, frame_bytes_to_numpy_rgb

from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import get_policy_class
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_STATE

from shared.common import env_str, load_env, mint_token, pace, required_env
from shared.config import FPS
from shared.rest_pose import ARM_POS_KEYS, RESET_POSE_DEFAULTS, SLIDER_VEL_KEY

import cv2

from operators.policy.settle import SettleGate
from operators.policy.dit import (
    ACTION_NAMES,
    POLICY_TYPE,
    STATE_NAMES,
    expected_image_hw,
    resolve_camera_map,
)

IDENTITY = "policy-operator"
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "portal.yaml"

# Every pick starts by folding here, because that is where every recorded episode
# starts: the rig can come to rest with the elbow standing up, tens of units
# outside anything in the dataset, and the first chunk is conditioned on whatever
# pose we plan from. Arm keys only — the ramp pins the slider itself.
START_POSE: dict[str, float] = {key: RESET_POSE_DEFAULTS[key] for key in ARM_POS_KEYS}

# The worst-case fold `--start-ramp` is sized for: the elbow-up rest the rig can settle
# into is ~60 units from START_POSE. Shorter moves scale down from here, so the ramp caps
# slew rate (~60 units per --start-ramp seconds) instead of costing a flat delay.
RAMP_REFERENCE_UNITS = 60.0

logger = logging.getLogger(__name__)


def _now_us() -> int:
    return int(time.time() * 1_000_000)


class PolicyRunner:
    """Owns the policy + the operator's action stream."""

    def __init__(self, op: Operator, *, fps: int, device: str, duration_s: float,
                 policy, preprocessor, postprocessor, camera_for_key: dict[str, str],
                 image_hw: dict[str, tuple[int, int]] | None = None,
                 settle_tolerance: float, settle_timeout_s: float,
                 start_ramp_s: float = 2.0, start_tolerance: float = 3.0,
                 on_tick: Callable[[dict], None] | None = None) -> None:
        self._op = op
        self._fps = fps
        self._device = torch.device(device)
        self._duration_s = duration_s
        self._policy = policy
        self._pre = preprocessor
        self._post = postprocessor
        self._camera_for_key = camera_for_key  # policy image key -> physical camera name
        self._image_hw = image_hw or {}  # policy image key -> (h, w) it was trained at
        self._settle = SettleGate(
            keys=ARM_POS_KEYS, tolerance=settle_tolerance, timeout_s=settle_timeout_s
        )
        self._state: dict[str, float] = {}
        # Absolute actions awaiting execution; see _infer for why chunking lives here.
        self._chunk: deque = deque()
        self._obs_ts_us = 0
        self._frames: dict[str, np.ndarray] = {}
        self._start_ramp_s = start_ramp_s
        self._start_tolerance = start_tolerance
        self._task = ""
        self._reprime = False  # set by set_task: refold before planning the new one
        self._on_tick = on_tick  # per-tick telemetry for the debug driver
        self._stop = asyncio.Event()
        op.on_observation(self._on_observation)

    def _on_observation(self, obs: Observation) -> None:
        self._state = dict(obs.state)
        self._obs_ts_us = obs.timestamp_us
        for name in set(self._camera_for_key.values()):
            frame = obs.frames.get(name)
            if frame is not None:
                self._frames[name] = frame_bytes_to_numpy_rgb(frame.data, frame.width, frame.height)

    @property
    def ready(self) -> bool:
        have_state = all(key in self._state for key in ARM_POS_KEYS)
        have_frames = all(cam in self._frames for cam in self._camera_for_key.values())
        return have_state and have_frames

    def require_ready(self) -> None:
        if not self.ready:
            raise RpcError.Error(code=1409, message="no robot state/frames yet", data=None)

    def set_task(self, task: str) -> None:
        """Swap the instruction. A run in flight folds back to ``START_POSE`` before
        planning on it, so the new instruction is conditioned on the same pose a
        fresh pick would start from rather than wherever the old one left the arm."""
        if task == self._task:
            return
        self._task = task
        self._reprime = True

    def request_stop(self) -> None:
        self._stop.set()

    def _replan_pending(self) -> bool:
        """True when the next tick runs the model instead of popping.

        Reads our own queue rather than the policy's: we drive chunking here (see
        :meth:`_infer`), so ``policy._action_queue`` is never filled at all.
        """
        return not self._chunk

    def _resize_for(self, key: str, frame: np.ndarray) -> np.ndarray:
        """Scale a live frame to the resolution this key was recorded at."""
        target = self._image_hw.get(key)
        if target is None or frame.shape[:2] == target:
            return frame
        return cv2.resize(frame, (target[1], target[0]), interpolation=cv2.INTER_AREA)

    @torch.inference_mode()
    def _infer(self, state_vec: np.ndarray, images: dict[str, np.ndarray], task: str) -> torch.Tensor:
        """One policy step, served from a locally held chunk of absolute actions.

        DiT keeps observation deques (``n_obs_steps`` frames of history) that only
        ``select_action`` fills, so ``predict_action_chunk`` cannot be called directly --
        it raises "stack expects a non-empty TensorList" on a fresh policy. Drive it the
        way it expects: the first call generates a chunk, the remaining ``n_action_steps``
        calls dequeue from it without re-running the model.
        """
        if self._chunk:
            return self._chunk.popleft()

        observation: dict[str, object] = {OBS_STATE: state_vec}
        observation.update(images)
        observation = prepare_observation_for_inference(observation, self._device, task, robot_type="")
        observation = self._pre(observation)

        self._policy.reset()
        for _ in range(getattr(self._policy.config, "n_action_steps", 1)):
            action = self._policy.select_action(dict(observation))
            self._chunk.append(self._post(action).squeeze(0).cpu())
        return self._chunk.popleft()

    def _send_arm(self, slider_vel: float) -> None:
        """Hold the current arm pose, sending the given slider velocity."""
        action = {key: float(self._state[key]) for key in ARM_POS_KEYS}
        action[SLIDER_VEL_KEY] = float(slider_vel)
        self._op.send_action(action, timestamp_us=_now_us(), in_reply_to_ts_us=self._obs_ts_us)

    async def _goto_start(self) -> bool:
        """Fold the arm to ``START_POSE`` before the first plan.

        Ramped rather than commanded as one absolute jump: the target can be 60+
        units away and the follower would otherwise slew there at full speed.

        The ramp is a cap on *slew rate*, not a fixed duration, so it scales with
        how far the arm actually has to travel — ``start_ramp_s`` is the time for
        the worst case (:data:`RAMP_REFERENCE_UNITS`) and anything closer is
        proportionally quicker. A retry usually starts near the rest pose already,
        and spending the full budget interpolating from here to nearly-here just
        made every attempt feel slow.

        Returns False only if stopped mid-approach; a joint that never arrives is
        logged and the pick proceeds anyway.
        """
        if not all(key in self._state for key in ARM_POS_KEYS):
            return True

        origin = {key: float(self._state[key]) for key in ARM_POS_KEYS}
        gap = max(abs(START_POSE[key] - origin[key]) for key in ARM_POS_KEYS)
        logger.info("[policy] priming to start pose (max joint move %.1f, ramp %.2fs)",
                    gap, self._start_ramp_s * min(1.0, gap / RAMP_REFERENCE_UNITS))

        ramp_s = self._start_ramp_s * min(1.0, gap / RAMP_REFERENCE_UNITS)
        ramp_ticks = max(1, int(ramp_s * self._fps))
        deadline = time.monotonic() + ramp_s + self._settle.timeout_s
        tick = 0
        async for _ in pace(self._fps):
            if self._stop.is_set():
                return False
            tick += 1
            alpha = min(1.0, tick / ramp_ticks)
            cmd = {key: origin[key] + (START_POSE[key] - origin[key]) * alpha
                   for key in ARM_POS_KEYS}
            cmd[SLIDER_VEL_KEY] = 0.0
            self._op.send_action(cmd, timestamp_us=_now_us(), in_reply_to_ts_us=self._obs_ts_us)
            if alpha < 1.0:
                continue
            error = max(abs(self._state[key] - START_POSE[key]) for key in ARM_POS_KEYS)
            if error <= self._start_tolerance:
                logger.info("[policy] at start pose (max joint err %.1f)", error)
                return True
            if time.monotonic() > deadline:
                logger.warning("[policy] start pose not reached (max joint err %.1f); planning anyway", error)
                return True
        return True

    async def pick(self, task: str) -> dict:
        """Run the policy until done/timeout/stop, holding the slider still."""
        self._task = task
        self._reprime = False  # the _goto_start below is the priming for this task
        self._stop.clear()
        self._settle.reset()
        await self._op.set_active_operator(self._op.local_identity())
        self._policy.reset()
        self._pre.reset()
        self._post.reset()
        logger.info("[policy] pick(%r): active operator -> %s", task, self._op.local_identity())

        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        ticks = 0
        reason = "duration"
        try:
            # The pose we arrive at is what the first chunk gets conditioned on.
            # A stop during the approach falls through to the loop's stop check.
            await self._goto_start()
            self._settle.reset()  # the approach already left the arm settled

            async for _ in pace(self._fps):
                if self._stop.is_set():
                    reason = "stopped"
                    break
                # duration <= 0 means run forever: the reward operator's run_task
                # owns the stop signal (via the stop RPC), not a wall-clock cap.
                if self._duration_s > 0 and time.monotonic() - t0 > self._duration_s:
                    reason = "duration"
                    break
                if not self.ready:
                    continue

                # A prompt swap re-primes: fold back before planning on it, for the
                # same reason the first plan does. The buffered chunk goes too --
                # it was planned for the old instruction, and without the reset it
                # would keep popping (and steer the fold) until it drained.
                if self._reprime:
                    self._reprime = False
                    logger.info("[policy] prompt changed to %r: refolding", self._task)
                    self._policy.reset()
                    self._pre.reset()
                    self._post.reset()
                    self._chunk.clear()
                    if not await self._goto_start():
                        reason = "stopped"
                        break
                    self._settle.reset()
                    continue

                # Gate the replan boundary only: a fresh chunk must be planned
                # from an observation the arm has caught up to, but the 29
                # mid-chunk pops are just dequeues and run at full fps.
                replan = self._replan_pending()
                if replan and not self._settle.ready(self._state):
                    continue

                obs_ts = self._obs_ts_us
                # Seven wide, including slider.vel: see STATE_NAMES. Default it to 0
                # rather than requiring it — `ready` gates on the arm keys only, and 0
                # is what the column held for nearly every frame pi0 trained on.
                state_vec = np.array([self._state.get(key, 0.0) for key in STATE_NAMES], dtype=np.float32)
                images = {key: self._resize_for(key, self._frames[cam])
                          for key, cam in self._camera_for_key.items()}

                # Inference is heavy and synchronous; run it off the event loop so
                # Portal keeps delivering fresh observations between ticks.
                t_infer = time.monotonic()
                action = await loop.run_in_executor(None, self._infer, state_vec, images, self._task)
                infer_ms = (time.monotonic() - t_infer) * 1000.0

                cmd = {name: float(action[i]) for i, name in enumerate(ACTION_NAMES)}
                cmd[SLIDER_VEL_KEY] = 0.0  # the slider is the move_to operator's job
                self._op.send_action(cmd, timestamp_us=_now_us(), in_reply_to_ts_us=obs_ts)
                self._settle.record(cmd)
                ticks += 1
                if self._on_tick is not None:
                    self._on_tick({"tick": ticks, "infer_ms": infer_ms, "replan": replan,
                                   "cmd": cmd, "state": dict(self._state)})
        finally:
            if all(key in self._state for key in ARM_POS_KEYS):
                self._send_arm(0.0)
            await self._op.set_active_operator(None)

        elapsed = time.monotonic() - t0
        logger.info("[policy] pick done: reason=%s ticks=%d elapsed=%.2fs", reason, ticks, elapsed)
        return {"task": self._task, "reason": reason, "ticks": ticks, "elapsed_s": elapsed}


def _load_policy(checkpoint: str, device: str, num_steps: int):
    """Load the checkpoint, wire its normalizer, and build the processors."""
    config_path = pathlib.Path(checkpoint) / "config.json"
    if config_path.is_file():
        policy_type = json.loads(config_path.read_text()).get("type")
        if policy_type != POLICY_TYPE:
            raise RuntimeError(
                f"{checkpoint} is a {policy_type!r} checkpoint; this operator serves "
                f"{POLICY_TYPE!r} only. See operators/policy/RESULTS.md."
            )

    logger.info("[policy] loading %s (downloads the CLIP encoders on first run)...", checkpoint)
    policy = get_policy_class(POLICY_TYPE).from_pretrained(checkpoint)
    policy.config.device = device
    if num_steps > 0:
        policy.config.num_inference_steps = num_steps
    policy = policy.to(device)
    policy.eval()

    # The checkpoint's saved processors carry the normalization stats; retarget
    # the device step to wherever we're running.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def add_policy_args(parser: argparse.ArgumentParser) -> None:
    """The checkpoint/inference knobs, shared with the debug driver."""
    # No default: there is no servable stock pi0 checkpoint, so guessing one
    # would only fail later, deep in a weight load.
    checkpoint = env_str("POLICY_CHECKPOINT", "")
    parser.add_argument("--checkpoint", default=checkpoint or None, required=not checkpoint,
                        help="pi0 checkpoint to serve: a local fine-tune directory or a Hub "
                             "repo id (or set POLICY_CHECKPOINT).")
    parser.add_argument("--task", default=env_str("POLICY_TASK", "pick up the candy"))
    parser.add_argument("--device", default=env_str("POLICY_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--duration", type=float, default=float(env_str("POLICY_DURATION_S", "0")),
                        help="Wall-clock cap in seconds; <=0 runs forever until a stop RPC (the reward operator drives this).")
    parser.add_argument("--num-steps", type=int, default=int(env_str("POLICY_NUM_STEPS", "0")),
                        help="Flow-matching denoising steps per chunk (latency vs quality); "
                             "0 keeps the checkpoint's value (10).")
    parser.add_argument("--settle-tolerance", type=float, default=float(env_str("POLICY_SETTLE_TOLERANCE", "2.0")),
                        help="Max per-joint error to call the arm settled; <=0 disables the gate.")
    parser.add_argument("--settle-timeout", type=float, default=float(env_str("POLICY_SETTLE_TIMEOUT_S", "2.0")),
                        help="Give up waiting for the arm to settle after this long.")
    parser.add_argument("--start-ramp", type=float, default=float(env_str("POLICY_START_RAMP_S", "2.0")),
                        help="Seconds to ease into the folded start pose (it can be 60+ units away).")
    parser.add_argument("--start-tolerance", type=float, default=float(env_str("POLICY_START_TOLERANCE", "3.0")),
                        help="Max per-joint error to call the start pose reached.")


def build_runner(op: Operator, args: argparse.Namespace,
                 on_tick: Callable[[dict], None] | None = None) -> PolicyRunner:
    """Load the checkpoint, map cameras onto its image keys, and wire a runner onto `op`."""
    policy, preprocessor, postprocessor = _load_policy(args.checkpoint, args.device, args.num_steps)

    # Off the preprocessor, not the policy config: the pipeline renames our camera
    # keys into pi0's openpi slots, so what we feed are the pre-rename names.
    camera_for_key = resolve_camera_map(
        policy.config,
        env_str("POLICY_PRIMARY_CAMERA", "overhead_camera"),
        env_str("POLICY_WRIST_CAMERA", "arm_camera"),
    )
    logger.info("[policy] image mapping: %s", camera_for_key)

    return PolicyRunner(
        op, fps=FPS, device=args.device, duration_s=args.duration,
        policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
        camera_for_key=camera_for_key,
        image_hw=expected_image_hw(policy.config),
        settle_tolerance=args.settle_tolerance, settle_timeout_s=args.settle_timeout,
        start_ramp_s=args.start_ramp, start_tolerance=args.start_tolerance,
        on_tick=on_tick,
    )


async def main() -> None:
    # Before the parser: the POLICY_* defaults (and the required --checkpoint) are
    # read while the arguments are declared, so .env has to be in os.environ first.
    load_env(pathlib.Path(__file__).resolve().parent)

    parser = argparse.ArgumentParser(description="Serve pi0 as a candy-shop picker operator.")
    add_policy_args(parser)
    args = parser.parse_args()

    # force: importing lerobot installs a root handler, which would make this a no-op
    # and leave the root logger at WARNING — every INFO line below silently dropped.
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    url = required_env("LIVEKIT_URL")
    room = env_str("LIVEKIT_ROOM", "candy-shop")
    token = mint_token(IDENTITY, room)

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
    op = Operator(cfg)
    runner = build_runner(op, args)

    async def run_policy(data: RpcInvocationData) -> str:
        """Run the policy for one order. Payload: task string or {"task": ...}."""
        logger.info("[policy] run_policy RPC from '%s'", data.caller_identity)
        task = _payload_task(data, args.task)
        runner.require_ready()
        return json.dumps(await runner.pick(task))

    async def stop(data: RpcInvocationData) -> str:
        """Preempt the running pick (releases active control)."""
        logger.info("[policy] stop RPC from '%s'", data.caller_identity)
        runner.request_stop()
        return json.dumps({"stopped": True})

    op.register_rpc_method("run_policy", run_policy)
    op.register_rpc_method("stop", stop)
    op.on_operator_joined(lambda i: logger.info("[policy] operator joined: %s", i))
    op.on_operator_left(lambda i: logger.info("[policy] operator left: %s", i))

    logger.info("[policy] connecting to %s as '%s' in room '%s' ...", url, IDENTITY, room)
    await op.connect(url, token)
    logger.info("[policy] connected as '%s'; awaiting run_policy RPCs", op.local_identity())

    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("[policy] stopping ...")
    finally:
        try:
            await op.disconnect()
        finally:
            op.close()


def cli() -> None:
    """Console-script entry point (`uv run policy`)."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
