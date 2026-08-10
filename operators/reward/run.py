"""Operator side: the candy shop's reward monitor (SARM), a wrapper over any policy.

Joins the room as an Operator peer and serves one RPC:

``run_task(task)``
  ``task`` is a natural-language instruction (``"pick up the red candy"``). This
  operator does **not** drive the arm itself — it orchestrates the *policy*
  operator and watches for completion:

  1. **give active control to the policy** — set the robot's active operator to
     ``policy-operator`` and kick off its ``run_policy`` (which now runs forever).
  2. **monitor** — poll SARM on the incoming camera frames; SARM emits a progress
     reward in ``[0, 1]`` for the task.
  3. **done** — once progress holds above ``--threshold`` for ``--hold-ticks``
     polls (or ``--timeout`` elapses), preempt the policy (``stop`` RPC) and
     release active control (set active operator to ``None``).
  4. **return** — a JSON summary of how it ended.

So ``run_task`` is a thin, policy-agnostic wrapper: point it at whatever policy
serves ``run_policy``/``stop`` and it turns "run forever" into "run until done".

Usage::

    uv run reward --checkpoint outputs/sarm-candy/pretrained_model
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

from livekit.portal import (
    Observation,
    Operator,
    OperatorConfig,
    RpcError,
    RpcInvocationData,
    frame_bytes_to_numpy_rgb,
)

from shared.common import env_str, load_env, mint_token, required_env

from operators.reward.sarm import ClipEncoder, DEFAULT_CHECKPOINT, ProgressScorer, load_reward_model

IDENTITY = "reward-operator"
POLICY_IDENTITY = "policy-operator"
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "portal.yaml"

logger = logging.getLogger(__name__)


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


class RewardRunner:
    """Orchestrates the policy and watches SARM for task completion."""

    def __init__(self, op: Operator, *, camera: str, scorer: ProgressScorer,
                 threshold: float, hold_ticks: int, timeout_s: float,
                 eval_interval_s: float, policy_timeout_s: float) -> None:
        self._op = op
        self._camera = camera
        self._scorer = scorer
        self._threshold = threshold
        self._hold_ticks = hold_ticks
        self._timeout_s = timeout_s
        self._eval_interval_s = eval_interval_s
        self._policy_timeout_s = policy_timeout_s
        self._frame: np.ndarray | None = None
        self._stop = asyncio.Event()
        self._busy = False
        op.on_observation(self._on_observation)

    def _on_observation(self, obs: Observation) -> None:
        frame = obs.frames.get(self._camera)
        if frame is not None:
            self._frame = frame_bytes_to_numpy_rgb(frame.data, frame.width, frame.height)

    def request_stop(self) -> None:
        self._stop.set()

    async def _start_policy(self, task: str) -> asyncio.Task:
        """Kick off the policy's run_policy in the background (it runs forever)."""
        response_timeout_ms = int((self._policy_timeout_s + self._timeout_s) * 1000)
        return asyncio.create_task(
            self._op.perform_rpc(
                "run_policy", task, destination=POLICY_IDENTITY, response_timeout_ms=response_timeout_ms
            )
        )

    async def _stop_policy(self, policy_task: asyncio.Task) -> None:
        """Preempt the policy and wait for its run_policy RPC to unwind."""
        try:
            await self._op.perform_rpc("stop", "", destination=POLICY_IDENTITY,
                                       response_timeout_ms=int(self._policy_timeout_s * 1000))
        except Exception:
            logger.exception("[reward] stop RPC to policy failed")
        try:
            await asyncio.wait_for(policy_task, timeout=self._policy_timeout_s)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            policy_task.cancel()
        except Exception:
            logger.exception("[reward] policy run_policy returned an error")

    async def run_task(self, task: str) -> dict:
        """Give control to the policy, watch SARM until done/timeout, then release."""
        if self._busy:
            raise RpcError.Error(code=1409, message="reward operator already running a task", data=None)
        self._busy = True
        self._stop.clear()
        self._scorer.reset()
        self._scorer.set_task(task)
        loop = asyncio.get_running_loop()

        # Step 1: give active control to the policy, then start it.
        await self._op.set_active_operator(POLICY_IDENTITY)
        logger.info("[reward] run_task(%r): active operator -> %s", task, POLICY_IDENTITY)
        policy_task = await self._start_policy(task)

        # Step 2: monitor SARM until done/timeout/stop.
        t0 = time.monotonic()
        ticks = 0
        held = 0
        progress = 0.0
        reason = "timeout"
        try:
            while True:
                if self._stop.is_set():
                    reason = "stopped"
                    break
                if time.monotonic() - t0 > self._timeout_s:
                    reason = "timeout"
                    break
                if policy_task.done() and policy_task.exception() is not None:
                    reason = "policy_error"
                    break

                await asyncio.sleep(self._eval_interval_s)
                if self._frame is None:
                    continue

                self._scorer.push(self._frame)
                # Inference is synchronous and heavy; keep the event loop free so
                # observations keep flowing in.
                progress = await loop.run_in_executor(None, self._scorer.progress)
                ticks += 1
                held = held + 1 if progress >= self._threshold else 0
                logger.debug("[reward] tick %d progress=%.3f held=%d", ticks, progress, held)
                if held >= self._hold_ticks:
                    reason = "done"
                    break
        finally:
            # Step 3: preempt the policy and release active control.
            await self._stop_policy(policy_task)
            await self._op.set_active_operator(None)
            self._busy = False

        elapsed = time.monotonic() - t0
        logger.info("[reward] run_task done: reason=%s progress=%.3f ticks=%d elapsed=%.2fs",
                    reason, progress, ticks, elapsed)
        return {"task": task, "reason": reason, "done": reason == "done",
                "progress": progress, "ticks": ticks, "elapsed_s": elapsed}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Serve SARM as a candy-shop reward monitor (policy wrapper).")
    parser.add_argument("--checkpoint", default=env_str("REWARD_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--task", default=env_str("REWARD_TASK", "pick up the candy"))
    parser.add_argument("--camera", default=env_str("REWARD_CAMERA", "overhead_camera"),
                        help="Camera SARM watches (must match training).")
    parser.add_argument("--device", default=env_str("REWARD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--threshold", type=float, default=float(env_str("REWARD_THRESHOLD", "0.95")),
                        help="Progress reward at/above which the task counts as complete.")
    parser.add_argument("--hold-ticks", type=int, default=int(env_str("REWARD_HOLD_TICKS", "3")),
                        help="Consecutive polls above threshold before calling it done.")
    parser.add_argument("--timeout", type=float, default=float(env_str("REWARD_TIMEOUT_S", "30")),
                        help="Safety cap on a task before releasing regardless of progress.")
    parser.add_argument("--eval-interval", type=float, default=float(env_str("REWARD_EVAL_INTERVAL_S", "1.0")),
                        help="Seconds between SARM polls (also the frame-buffer stride).")
    parser.add_argument("--policy-timeout", type=float, default=float(env_str("REWARD_POLICY_TIMEOUT_S", "10")),
                        help="Timeout for the stop RPC / policy unwind.")
    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_env(pathlib.Path(__file__).resolve().parent)

    url = required_env("LIVEKIT_URL")
    room = env_str("LIVEKIT_ROOM", "candy-shop")
    token = mint_token(IDENTITY, room)

    model, config = load_reward_model(args.checkpoint, args.device)
    encoder = ClipEncoder(args.device)
    scorer = ProgressScorer(model, config, encoder)

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
    op = Operator(cfg)
    runner = RewardRunner(
        op, camera=args.camera, scorer=scorer,
        threshold=args.threshold, hold_ticks=args.hold_ticks, timeout_s=args.timeout,
        eval_interval_s=args.eval_interval, policy_timeout_s=args.policy_timeout,
    )

    async def run_task(data: RpcInvocationData) -> str:
        """Run one task: drive the policy, watch SARM, release. Payload: task string or {"task": ...}."""
        logger.info("[reward] run_task RPC from '%s'", data.caller_identity)
        task = _payload_task(data, args.task)
        return json.dumps(await runner.run_task(task))

    async def stop(data: RpcInvocationData) -> str:
        """Preempt the running task (stops the policy and releases control)."""
        logger.info("[reward] stop RPC from '%s'", data.caller_identity)
        runner.request_stop()
        return json.dumps({"stopped": True})

    op.register_rpc_method("run_task", run_task)
    op.register_rpc_method("stop", stop)
    op.on_operator_joined(lambda i: logger.info("[reward] operator joined: %s", i))
    op.on_operator_left(lambda i: logger.info("[reward] operator left: %s", i))

    logger.info("[reward] connecting to %s as '%s' in room '%s' ...", url, IDENTITY, room)
    await op.connect(url, token)
    logger.info("[reward] connected as '%s'; awaiting run_task RPCs", op.local_identity())

    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("[reward] stopping ...")
    finally:
        try:
            await op.disconnect()
        finally:
            op.close()


def cli() -> None:
    """Console-script entry point (`uv run reward`)."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
