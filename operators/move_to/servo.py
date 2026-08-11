"""Closed-loop slider servo: drive the ArUco marker onto a target line.

The control law is a sqrt decel profile, not a PID — see `config`.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Optional

from livekit.portal import Observation, Operator, RpcError, frame_bytes_to_numpy_rgb

from shared.common import pace
from shared.rest_pose import ARM_POS_KEYS, SLIDER_VEL_KEY

from operators.move_to import config
from operators.move_to.vision import (
    ArucoDetector,
    AxisEstimator,
    Estimate,
    MarkerDetection,
    SafeZone,
)

logger = logging.getLogger(__name__)
CAMERA = "overhead_camera"


def _now_us() -> int:
    return int(time.time() * 1_000_000)


class SliderServo:
    def __init__(
        self,
        op: Operator,
        *,
        fps: int,
        safe_zone: SafeZone,
        detector: ArucoDetector,
    ) -> None:
        self._op = op
        self._fps = fps
        self._safe_zone = safe_zone
        self._detector = detector
        self._estimator = AxisEstimator()
        self._state: dict[str, float] = {}
        self._obs_ts_us = 0
        self._frame = None  # RGB
        self._frame_size: tuple[int, int] | None = None
        self._frame_t: float | None = None  # monotonic, when it last arrived
        self._tracked_frame_t: float | None = None  # ... and when track() consumed it
        self._last_marker: Optional[MarkerDetection] = None
        self._last_estimate: Optional[Estimate] = None
        self._stop = asyncio.Event()
        op.on_observation(self._on_observation)

    def _on_observation(self, obs: Observation) -> None:
        self._state = dict(obs.state)
        self._obs_ts_us = obs.timestamp_us
        f = obs.frames.get(CAMERA)
        if f is not None:
            if self._frame_size not in (None, (f.width, f.height)):
                # Harmless for tracking (everything is normalized) but it does
                # shift detection: the threshold windows are in real pixels.
                logger.info(
                    "[move-to] %s resized %s -> %dx%d", CAMERA,
                    "x".join(map(str, self._frame_size)), f.width, f.height,
                )
            self._frame_size = (f.width, f.height)
            self._frame = frame_bytes_to_numpy_rgb(f.data, f.width, f.height)
            self._frame_t = time.monotonic()

    # --- primitives (shared with the debug tool) ---

    @property
    def has_state(self) -> bool:
        return all(k in self._state for k in ARM_POS_KEYS)

    @property
    def frame(self):
        return self._frame

    @property
    def frame_age_s(self) -> float:
        """Seconds since a camera frame last arrived; inf before the first one."""
        return float("inf") if self._frame_t is None else time.monotonic() - self._frame_t

    @property
    def slider_vel(self) -> float:
        """Measured carriage velocity (raw ticks/s); 0 before any state arrives."""
        return float(self._state.get(SLIDER_VEL_KEY, 0.0))

    @property
    def last_marker(self) -> Optional[MarkerDetection]:
        """The detection from the most recent `track()` — for overlays."""
        return self._last_marker

    @property
    def last_estimate(self) -> Optional[Estimate]:
        return self._last_estimate

    def require_state(self) -> None:
        if not self.has_state:
            missing = [k for k in ARM_POS_KEYS if k not in self._state]
            raise RpcError.Error(
                code=1409, message=f"no robot state yet (missing: {missing})", data=None
            )

    def track(self, dt: float) -> Estimate:
        """Advance the fused estimate one tick from the newest frame + slider velocity.

        Call exactly once per control tick: `dt` drives both the dead reckoning and
        the staleness clock.

        Only a *newly arrived* frame yields a measurement. Re-detecting the frame
        we already used would feed the filter a duplicate — and if the stream
        died, a frozen image detects perfectly forever, so the servo would drive
        blind while believing it had vision.
        """
        measured = None
        if self._frame is not None and self._frame_t != self._tracked_frame_t:
            self._tracked_frame_t = self._frame_t
            self._last_marker = self._detector.detect(self._frame)
            if self._last_marker is not None:
                h, w = self._frame.shape[:2]
                measured = self._safe_zone.marker_coord(self._last_marker, h, w)
        self._last_estimate = self._estimator.update(measured, self.slider_vel, dt)
        return self._last_estimate

    def reset_tracking(self) -> None:
        """Forget where the marker was, keeping the fitted slider gain."""
        self._estimator.reset_position()

    def send(self, slider_vel: float) -> None:
        """Send the arm (mirrored, held) plus the commanded slider velocity."""
        action = {k: float(self._state[k]) for k in ARM_POS_KEYS}
        action[SLIDER_VEL_KEY] = float(slider_vel)
        self._op.send_action(action, timestamp_us=_now_us(), in_reply_to_ts_us=self._obs_ts_us)

    def send_stop(self) -> None:
        if self.has_state:
            self.send(0.0)

    def request_stop(self) -> None:
        """Preempt the move in flight; the loop unwinds within a tick.

        Same shape as the policy and reward operators' `request_stop`. Setting the
        event is all this does — `servo_to`'s `finally` is what actually zeroes the
        carriage and releases control, so a stop takes the identical exit path as a
        timeout or a converged move. Safe to call when nothing is running: the flag
        is cleared on entry to the next `servo_to`, so it can't leak into it.
        """
        self._stop.set()

    async def claim(self) -> None:
        await self._op.set_active_operator(self._op.local_identity())

    async def release(self) -> None:
        try:
            await self._op.set_active_operator(None)
        except Exception:
            logger.exception("[move-to] set_active_operator(None) failed")

    def velocity_for(self, target_pos: float, coord: float) -> tuple[float, float]:
        """Returns ``(slider_vel, error_px)`` for an estimated axis `coord`.

        Memoryless: the command depends on the current error and nothing else. No
        integrator (so the residual inside the deadzone is permanent — that's
        `DEADZONE_PX`), no derivative (velocity falling with distance is already
        the braking). `error_px` is in reference pixels, not pixels of the current
        frame — see `config.REFERENCE_AXIS_PX`.
        """
        error_px = (
            self._safe_zone.target_coord(target_pos) - coord
        ) * config.REFERENCE_AXIS_PX
        if abs(error_px) <= config.DEADZONE_PX:
            return 0.0, error_px

        # Floored at 1px so a live-tuned 0 degrades to bang-bang, not a div by zero.
        full_speed_px = max(config.APPROACH_FULL_SPEED_PX, 1.0)
        speed = config.MAX_VELOCITY * math.sqrt(min(1.0, abs(error_px) / full_speed_px))
        # Below breakaway the carriage buzzes instead of moving.
        speed = min(config.MAX_VELOCITY, max(config.MIN_VELOCITY, speed))
        direction = -1.0 if config.INVERT else 1.0
        return math.copysign(speed, direction * error_px), error_px

    def pos_of(self, coord: float) -> float:
        return self._safe_zone.pos_of(coord)

    def _recover(
        self, est: Estimate, acquired: bool, last_vel: float
    ) -> tuple[Optional[str], bool]:
        """Decide what to do on a tick with no usable position.

        Returns ``(give_up_reason, creeping)`` — a reason ends the move, ``None``
        means try again next tick. The ladder, cheapest and safest first:

        1. Never acquired at all: just wait. Nothing has gone wrong yet; the arm
           may simply not be in frame yet.
        2. Observations stopped arriving: fail immediately. Blind motion when the
           robot or stream is gone is the one thing we must not do.
        3. Hold still. Free, safe, and it fixes the most common cause: motion blur
           is self-curing once stopped, and the marker cannot move on its own.
        4. Creep blind toward where we were heading (opt-in), bounded by time and
           by the dead-reckoned safe-zone edge, hoping to clear a shadow.
        """
        if not acquired:
            self.send_stop()
            return None, False

        if self.frame_age_s > config.FRAME_STALE_S:
            self.send_stop()
            logger.warning("[move-to] no frames for %.1fs; giving up", self.frame_age_s)
            return "no_frames", False

        blind_s = est.age_s - config.MAX_COAST_S
        if blind_s <= config.REACQUIRE_HOLD_S:
            self.send_stop()
            return None, False

        creep_s = blind_s - config.REACQUIRE_HOLD_S
        # `est.coord` keeps dead reckoning while we creep, which is what bounds
        # this: too degraded to servo on, still good enough to know the rail end.
        at_edge = est.coord is None or not 0.0 < self.pos_of(est.coord) < 100.0
        if config.LOST_CREEP_ENABLED and last_vel and not at_edge \
                and creep_s <= config.LOST_CREEP_MAX_S:
            self.send(math.copysign(config.LOST_CREEP_VELOCITY, last_vel))
            return None, True

        self.send_stop()
        logger.warning(
            "[move-to] marker lost for %.1fs (%s); stopping",
            est.age_s, "at safe-zone edge" if at_edge else "no re-acquisition",
        )
        return "lost", False

    # --- one-shot move (operator RPC) ---

    async def servo_to(self, target_pos: float, label: str) -> dict:
        """Drive the marker onto the target line, then stop and release."""
        await self.claim()
        logger.info("[move-to] %s: active operator -> %s", label, self._op.local_identity())

        # Discard any stop that arrived while nothing was moving, so it can't abort
        # the move we're about to start (the agent's chain issues back-to-back
        # move_to calls, and a late stop from the previous one would kill the next).
        self._stop.clear()
        self.reset_tracking()
        reached = False
        reason = "timeout"
        iterations = 0
        consecutive_inside = 0
        coasted_ticks = 0
        creep_ticks = 0
        acquired = False
        last_vel = 0.0
        final_pos: Optional[float] = None
        last_t = time.monotonic()
        t0 = last_t

        try:
            async for _ in pace(self._fps):
                iterations += 1
                now = time.monotonic()
                dt = now - last_t
                last_t = now
                # Checked first: a stop must win over every other exit condition.
                if self._stop.is_set():
                    reason = "stopped"
                    break
                if now - t0 > config.TIMEOUT_S:
                    reason = "timeout"
                    break

                est = self.track(dt)
                acquired = acquired or est.measured
                if not est.usable:
                    consecutive_inside = 0
                    give_up, creeping = self._recover(est, acquired, last_vel)
                    creep_ticks += creeping
                    if give_up is not None:
                        reason = give_up
                        break
                    continue
                if not est.measured:
                    coasted_ticks += 1

                vel, error_px = self.velocity_for(target_pos, est.coord)
                self.send(vel)
                last_vel = vel or last_vel
                final_pos = self.pos_of(est.coord)

                if abs(error_px) <= config.DEADZONE_PX:
                    consecutive_inside += 1
                    if consecutive_inside >= config.CONVERGE_TICKS:
                        reached = True
                        reason = "reached"
                        break
                else:
                    consecutive_inside = 0

                if iterations % self._fps == 0:
                    logger.info(
                        "[move-to] tick %4d: pos=%.1f target=%.1f err=%+.1fpx vel=%+.0f src=%s",
                        iterations, final_pos, target_pos, error_px, vel,
                        "cam" if est.measured else f"coast({est.age_s:.2f}s)",
                    )
        finally:
            self.send_stop()
            await self.release()

        elapsed_s = time.monotonic() - t0
        logger.info(
            "[move-to] %s done: reason=%s reached=%s iterations=%d coasted=%d creep=%d "
            "elapsed=%.2fs gain=%.3e calibrated=%s",
            label, reason, reached, iterations, coasted_ticks, creep_ticks, elapsed_s,
            self._estimator.gain, self._estimator.calibrated,
        )
        return {
            "target_pos": target_pos,
            "reached": reached,
            "reason": reason,
            "iterations": iterations,
            "coasted_ticks": coasted_ticks,
            "creep_ticks": creep_ticks,
            "elapsed_s": elapsed_s,
            "final_pos": final_pos,
        }
