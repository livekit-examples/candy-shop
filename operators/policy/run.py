"""Picker operator: serve MolmoAct2 (six-DOF arm) over the ``run_policy`` RPC.

    uv run policy                             # default SO-101 checkpoint, zero-shot
    uv run policy --checkpoint outputs/molmoact2-candy/pretrained_model
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import time

import numpy as np
import torch

from livekit.portal import Observation, Operator, OperatorConfig, RpcError, RpcInvocationData, frame_bytes_to_numpy_rgb

from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.policies import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_STATE

from shared.common import env_str, load_env, mint_token, pace, required_env
from shared.config import FPS
from shared.rest_pose import ARM_POS_KEYS, SLIDER_VEL_KEY

from operators.policy.molmoact import ACTION_NAMES, DEFAULT_CHECKPOINT, resolve_image_keys
from operators.policy.settle import SettleGate

IDENTITY = "policy-operator"
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "portal.yaml"

logger = logging.getLogger(__name__)


def _now_us() -> int:
    return int(time.time() * 1_000_000)


def _payload_task(data: RpcInvocationData, default: str) -> str:
    """Parse an instruction from a bare string or ``{"task"|"instruction": ...}``."""
    payload = (data.payload or "").strip()
    if not payload:
        return default
    if payload.startswith("{"):
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            raise RpcError.Error(code=1400, message="payload was not valid JSON", data=None)
        for key in ("task", "instruction", "prompt"):
            if obj.get(key):
                return str(obj[key])
        raise RpcError.Error(code=1400, message="JSON payload had no task/instruction/prompt", data=None)
    return payload


class PolicyRunner:
    """Owns the policy + the operator's action stream."""

    def __init__(self, op: Operator, *, fps: int, device: str, duration_s: float,
                 policy, preprocessor, postprocessor, camera_for_key: dict[str, str],
                 settle_tolerance: float, settle_timeout_s: float) -> None:
        self._op = op
        self._fps = fps
        self._device = torch.device(device)
        self._duration_s = duration_s
        self._policy = policy
        self._pre = preprocessor
        self._post = postprocessor
        self._camera_for_key = camera_for_key  # policy image key -> physical camera name
        self._settle = SettleGate(
            keys=ARM_POS_KEYS, tolerance=settle_tolerance, timeout_s=settle_timeout_s
        )
        self._state: dict[str, float] = {}
        self._obs_ts_us = 0
        self._frames: dict[str, np.ndarray] = {}
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
    def _ready(self) -> bool:
        have_state = all(key in self._state for key in ARM_POS_KEYS)
        have_frames = all(cam in self._frames for cam in self._camera_for_key.values())
        return have_state and have_frames

    def require_ready(self) -> None:
        if not self._ready:
            raise RpcError.Error(code=1409, message="no robot state/frames yet", data=None)

    def request_stop(self) -> None:
        self._stop.set()

    def _replan_pending(self) -> bool:
        """True when the next ``select_action`` runs the model instead of popping.

        MolmoAct2 buffers a 30-step chunk in ``_action_queue`` and only replans
        once it drains (empty also before the first plan).
        """
        return not getattr(self._policy, "_action_queue", None)

    @torch.inference_mode()
    def _infer(self, state_vec: np.ndarray, images: dict[str, np.ndarray], task: str) -> torch.Tensor:
        """One policy step: raw obs -> pre -> select_action -> post. Returns [action_dim]."""
        observation: dict[str, object] = {OBS_STATE: state_vec}
        observation.update(images)
        observation = prepare_observation_for_inference(observation, self._device, task, robot_type="")
        observation = self._pre(observation)
        action = self._policy.select_action(observation)
        action = self._post(action)
        return action.squeeze(0).cpu()

    def _send_arm(self, slider_vel: float) -> None:
        """Hold the current arm pose, sending the given slider velocity."""
        action = {key: float(self._state[key]) for key in ARM_POS_KEYS}
        action[SLIDER_VEL_KEY] = float(slider_vel)
        self._op.send_action(action, timestamp_us=_now_us(), in_reply_to_ts_us=self._obs_ts_us)

    async def pick(self, task: str) -> dict:
        """Run the policy until done/timeout/stop, holding the slider still."""
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
            async for _ in pace(self._fps):
                if self._stop.is_set():
                    reason = "stopped"
                    break
                # duration <= 0 means run forever: the reward operator's run_task
                # owns the stop signal (via the stop RPC), not a wall-clock cap.
                if self._duration_s > 0 and time.monotonic() - t0 > self._duration_s:
                    reason = "duration"
                    break
                if not self._ready:
                    continue

                # Gate the replan boundary only: a fresh chunk must be planned
                # from an observation the arm has caught up to, but the 29
                # mid-chunk pops are just dequeues and run at full fps.
                if self._replan_pending() and not self._settle.ready(self._state):
                    continue

                obs_ts = self._obs_ts_us
                state_vec = np.array([self._state[key] for key in ARM_POS_KEYS], dtype=np.float32)
                images = {key: self._frames[cam] for key, cam in self._camera_for_key.items()}

                # Inference is heavy and synchronous; run it off the event loop so
                # Portal keeps delivering fresh observations between ticks.
                action = await loop.run_in_executor(None, self._infer, state_vec, images, task)

                cmd = {name: float(action[i]) for i, name in enumerate(ACTION_NAMES)}
                cmd[SLIDER_VEL_KEY] = 0.0  # the slider is the move_to operator's job
                self._op.send_action(cmd, timestamp_us=_now_us(), in_reply_to_ts_us=obs_ts)
                self._settle.record(cmd)
                ticks += 1
        finally:
            if all(key in self._state for key in ARM_POS_KEYS):
                self._send_arm(0.0)
            await self._op.set_active_operator(None)

        elapsed = time.monotonic() - t0
        logger.info("[policy] pick done: reason=%s ticks=%d elapsed=%.2fs", reason, ticks, elapsed)
        return {"task": task, "reason": reason, "ticks": ticks, "elapsed_s": elapsed}


def _load_policy(checkpoint: str, device: str, inference_action_mode: str):
    """Load the checkpoint, wire its normalizer, and build the processors."""
    logger.info("[policy] loading %s (downloads MolmoAct2 weights on first run)...", checkpoint)
    policy = MolmoAct2Policy.from_pretrained(checkpoint)
    policy.config.device = device
    policy.config.inference_action_mode = inference_action_mode
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


async def main() -> None:
    parser = argparse.ArgumentParser(description="Serve MolmoAct2 as a candy-shop picker operator.")
    parser.add_argument("--checkpoint", default=env_str("POLICY_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--task", default=env_str("POLICY_TASK", "pick up the candy"))
    parser.add_argument("--device", default=env_str("POLICY_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--duration", type=float, default=float(env_str("POLICY_DURATION_S", "0")),
                        help="Wall-clock cap in seconds; <=0 runs forever until a stop RPC (the reward operator drives this).")
    parser.add_argument("--inference-action-mode", default=env_str("POLICY_INFERENCE_ACTION_MODE", "continuous"))
    parser.add_argument("--settle-tolerance", type=float, default=float(env_str("POLICY_SETTLE_TOLERANCE", "2.0")),
                        help="Max per-joint error to call the arm settled; <=0 disables the gate.")
    parser.add_argument("--settle-timeout", type=float, default=float(env_str("POLICY_SETTLE_TIMEOUT_S", "2.0")),
                        help="Give up waiting for the arm to settle after this long.")
    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_env(pathlib.Path(__file__).resolve().parent)

    url = required_env("LIVEKIT_URL")
    room = env_str("LIVEKIT_ROOM", "candy-shop")
    token = mint_token(IDENTITY, room)

    policy, preprocessor, postprocessor = _load_policy(
        args.checkpoint, args.device, args.inference_action_mode
    )

    # Map the policy's image keys onto physical cameras: primary (overhead)
    # first, wrist (arm-mounted) second.
    image_keys = resolve_image_keys(policy.config)
    physical = [env_str("POLICY_PRIMARY_CAMERA", "overhead_camera"), env_str("POLICY_WRIST_CAMERA", "arm_camera")]
    if len(image_keys) > len(physical):
        raise RuntimeError(f"policy expects {len(image_keys)} images ({image_keys}) but only {physical} are wired")
    camera_for_key = {key: physical[i] for i, key in enumerate(image_keys)}
    logger.info("[policy] image mapping: %s", camera_for_key)

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
    op = Operator(cfg)
    runner = PolicyRunner(
        op, fps=FPS, device=args.device, duration_s=args.duration,
        policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
        camera_for_key=camera_for_key,
        settle_tolerance=args.settle_tolerance, settle_timeout_s=args.settle_timeout,
    )

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
