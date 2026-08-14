"""Operator side: the candy shop's reward monitor (SARM), a wrapper over any policy.

Joins the room as an Operator peer and serves one RPC:

``run_task(task)``
  ``task`` is a natural-language instruction (``"pick up the red candy"``). This
  operator does **not** drive the arm itself — it orchestrates the *policy*
  operator and watches for completion, retrying the whole pick on a deadline:

  1. **give active control to the policy** — set the robot's active operator to
     ``policy-operator`` and kick off its ``run_policy`` (which now runs forever).
  2. **monitor** — poll SARM on the incoming camera frames; SARM emits a progress
     reward in ``[0, 1]`` for the task.
  3. **done** — once progress holds above ``--threshold`` for ``--hold-seconds``,
     preempt the policy (``stop`` RPC) and release active control.
  4. **retry** — if the attempt burns its time budget first, stop the policy and
     start it again on the same instruction. ``run_policy`` folds to the rest pose
     before it plans, so each retry re-primes the arm from the pose every recorded
     episode starts at rather than from wherever the failed attempt left it.
     ``--attempt-budgets`` lists one budget per attempt and defaults to
     ``10,15,20``: three tries, each given longer than the last, on the theory
     that a pick that missed is worth more time and not infinite time.
  5. **return** — a JSON summary: which attempt won, and the time to completion.

So ``run_task`` is a thin, policy-agnostic wrapper: point it at whatever policy
serves ``run_policy``/``stop`` and it turns "run forever" into "run until done".

Usage::

    uv run reward --checkpoint outputs/sarm-candy/checkpoints/last/pretrained_model
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
from shared.rest_pose import ALL_ACTION_KEYS

from operators.reward.sarm import (ClipEncoder, DEFAULT_CHECKPOINT, DoneRule, ProgressScorer,
                                   StateNormalizer, load_reward_model)

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


def _parse_budgets(raw: str) -> list[float]:
    """``"10,15,20"`` -> ``[10.0, 15.0, 20.0]``, one time budget per attempt."""
    try:
        budgets = [float(part) for part in raw.split(",") if part.strip()]
    except ValueError:
        raise SystemExit(f"--attempt-budgets: {raw!r} is not a comma-separated list of seconds")
    if not budgets or any(b <= 0 for b in budgets):
        raise SystemExit(f"--attempt-budgets: {raw!r} needs at least one positive budget")
    return budgets


class RewardRunner:
    """Orchestrates the policy and watches SARM for task completion."""

    def __init__(self, op: Operator, *, camera: str, scorer: ProgressScorer,
                 threshold: float, hold_s: float, attempt_budgets: list[float],
                 timeout_s: float, eval_interval_s: float, policy_timeout_s: float,
                 retry_pause_s: float = 1.0) -> None:
        self._op = op
        self._camera = camera
        self._scorer = scorer
        self._done = DoneRule(threshold, hold_s, eval_interval_s)
        self._attempt_budgets = attempt_budgets
        self._timeout_s = timeout_s
        self._eval_interval_s = eval_interval_s
        self._policy_timeout_s = policy_timeout_s
        self._retry_pause_s = retry_pause_s
        self._frame: np.ndarray | None = None
        self._state: np.ndarray | None = None
        self._stop = asyncio.Event()
        self._busy = False
        op.on_observation(self._on_observation)

    def _on_observation(self, obs: Observation) -> None:
        frame = obs.frames.get(self._camera)
        if frame is not None:
            self._frame = frame_bytes_to_numpy_rgb(frame.data, frame.width, frame.height)
        # SARM reads arm pose as well as pixels; without it the progress curve goes flat.
        if obs.state:
            self._state = np.array([obs.state[k] for k in ALL_ACTION_KEYS], dtype=np.float32)

    def request_stop(self) -> None:
        self._stop.set()

    async def _start_policy(self, task: str, budget_s: float) -> asyncio.Task:
        """Kick off the policy's run_policy in the background (it runs forever).

        The RPC has to outlive the attempt it covers: run_policy only returns when we
        preempt it, so the response timeout is this attempt's budget plus the unwind
        allowance. It used to be `policy_timeout + timeout`, which broke when --timeout
        became an optional overall cap defaulting to 0 — the RPC then expired after
        policy_timeout alone, inside every attempt longer than that, and the run died
        with "rpc error 1502: Response timeout".
        """
        response_timeout_ms = int((budget_s + self._policy_timeout_s) * 1000)
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

    async def _attempt(self, task: str, budget_s: float, index: int, deadline: float | None) -> dict:
        """One pick: start the policy, watch SARM until done or the budget runs out.

        Always stops the policy on the way out, so the next attempt's ``run_policy``
        starts from a clean reset and folds to the rest pose before planning.
        """
        loop = asyncio.get_running_loop()
        # A stale window from the previous attempt would be scored against this
        # one's frames; SARM reads a window, not a single frame.
        self._scorer.reset()
        self._done.reset()

        await self._op.set_active_operator(POLICY_IDENTITY)
        logger.info("[reward] attempt %d/%d (%.1fs budget): active operator -> %s",
                    index, len(self._attempt_budgets), budget_s, POLICY_IDENTITY)
        policy_task = await self._start_policy(task, budget_s)

        t0 = time.monotonic()
        ticks = 0
        progress = 0.0
        reason = "budget"
        try:
            while True:
                if self._stop.is_set():
                    reason = "stopped"
                    break
                if time.monotonic() - t0 > budget_s:
                    reason = "budget"
                    break
                if deadline is not None and time.monotonic() > deadline:
                    reason = "timeout"
                    break
                if policy_task.done() and policy_task.exception() is not None:
                    reason = "policy_error"
                    break

                await asyncio.sleep(self._eval_interval_s)
                if self._frame is None or self._state is None:
                    continue

                self._scorer.push(self._frame, self._state)
                # Inference is synchronous and heavy; keep the event loop free so
                # observations keep flowing in.
                progress = await loop.run_in_executor(None, self._scorer.progress)
                ticks += 1
                finished = self._done.push(progress)
                logger.info("[reward] attempt %d/%d t=%4.1fs reward=%.3f (thr %.2f, held %d/%d)",
                            index, len(self._attempt_budgets), time.monotonic() - t0,
                            progress, self._done.threshold, self._done.held, self._done.hold_ticks)
                if finished:
                    reason = "done"
                    break
        finally:
            await self._stop_policy(policy_task)

        elapsed = time.monotonic() - t0
        logger.info("[reward] attempt %d ended: reason=%s progress=%.3f ticks=%d elapsed=%.2fs",
                    index, reason, progress, ticks, elapsed)
        return {"attempt": index, "budget_s": budget_s, "reason": reason,
                "progress": progress, "ticks": ticks, "elapsed_s": elapsed}

    async def run_task(self, task: str) -> dict:
        """Drive the policy through up to ``--attempt-budgets`` picks, then release."""
        if self._busy:
            raise RpcError.Error(code=1409, message="reward operator already running a task", data=None)
        self._busy = True
        self._stop.clear()
        self._scorer.set_task(task)

        t_start = time.monotonic()
        deadline = t_start + self._timeout_s if self._timeout_s > 0 else None
        attempts: list[dict] = []
        try:
            for index, budget_s in enumerate(self._attempt_budgets, start=1):
                attempt = await self._attempt(task, budget_s, index, deadline)
                attempts.append(attempt)
                # Only a spent budget is worth another pick: done is success, and
                # stopped/timeout/policy_error all mean retrying is wrong or futile.
                if attempt["reason"] != "budget":
                    break
                if deadline is not None and time.monotonic() > deadline:
                    break
                # Settle before going again: a failed pick often leaves the candy rolling
                # or the gripper mid-close, and the next attempt plans from whatever the
                # camera sees the instant it starts. Straight back-to-back retries also
                # read as thrashing on the rig.
                logger.info("[reward] attempt %d spent its %.1fs budget; pausing %.1fs then retrying",
                            index, budget_s, self._retry_pause_s)
                if self._retry_pause_s > 0:
                    await asyncio.sleep(self._retry_pause_s)
        finally:
            await self._op.set_active_operator(None)
            self._busy = False

        elapsed = time.monotonic() - t_start
        last = attempts[-1]
        # The final attempt's budget expiring means the task itself ran out of tries.
        reason = "exhausted" if last["reason"] == "budget" else last["reason"]
        done = reason == "done"
        logger.info("[reward] run_task done: reason=%s attempts=%d progress=%.3f elapsed=%.2fs",
                    reason, len(attempts), last["progress"], elapsed)
        return {"task": task, "reason": reason, "done": done,
                "progress": last["progress"], "ticks": last["ticks"],
                "attempts": attempts, "attempt_count": len(attempts),
                "time_to_completion_s": elapsed if done else None,
                "elapsed_s": elapsed}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Serve SARM as a candy-shop reward monitor (policy wrapper).")
    parser.add_argument("--checkpoint", default=env_str("REWARD_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--task", default=env_str("REWARD_TASK", "pick up the candy"))
    parser.add_argument("--camera", default=env_str("REWARD_CAMERA", "overhead_camera"),
                        help="Camera SARM watches (must match training).")
    parser.add_argument("--device", default=env_str("REWARD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--threshold", type=float, default=float(env_str("REWARD_THRESHOLD", "0.7")),
                        help="Progress reward at/above which the task counts as complete.")
    parser.add_argument("--hold-seconds", type=float, default=float(env_str("REWARD_HOLD_S", "1.0")),
                        help="Seconds progress must stay above threshold before calling it done.")
    parser.add_argument("--attempt-budgets", default=env_str("REWARD_ATTEMPT_BUDGETS", "10,15,20"),
                        help="Comma-separated per-attempt time budgets in seconds. Each budget is "
                             "one pick; a spent budget restarts the policy, which refolds to the "
                             "rest pose first. Default '10,15,20' = three tries.")
    parser.add_argument("--timeout", type=float, default=float(env_str("REWARD_TIMEOUT_S", "0")),
                        help="Overall safety cap across all attempts (0 = budgets govern).")
    parser.add_argument("--eval-interval", type=float, default=float(env_str("REWARD_EVAL_INTERVAL_S", "1.0")),
                        help="Seconds between SARM polls (also the frame-buffer stride).")
    parser.add_argument("--retry-pause", type=float, default=float(env_str("REWARD_RETRY_PAUSE_S", "1.0")),
                        help="Seconds to wait between a spent attempt and the next one.")
    parser.add_argument("--policy-timeout", type=float, default=float(env_str("REWARD_POLICY_TIMEOUT_S", "60")),
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
    scorer = ProgressScorer(model, config, encoder,
                            StateNormalizer.from_checkpoint(args.checkpoint))

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
    op = Operator(cfg)
    budgets = _parse_budgets(args.attempt_budgets)
    runner = RewardRunner(
        op, camera=args.camera, scorer=scorer,
        threshold=args.threshold, hold_s=args.hold_seconds, attempt_budgets=budgets,
        timeout_s=args.timeout,
        eval_interval_s=args.eval_interval, policy_timeout_s=args.policy_timeout,
        retry_pause_s=args.retry_pause,
    )
    logger.info("[reward] threshold=%.2f hold=%.1fs attempt budgets=%s (%.0fs total)",
                args.threshold, args.hold_seconds,
                ",".join(f"{b:g}" for b in budgets), sum(budgets))

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
