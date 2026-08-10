"""Closed-loop slider servo: drive the ArUco marker onto a target line."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from livekit.portal import Observation, Operator, RpcError, frame_bytes_to_numpy_rgb

from shared.common import pace
from shared.rest_pose import ARM_POS_KEYS, SLIDER_VEL_KEY

from operators.move_to import config
from operators.move_to.vision import ArucoDetector, MarkerDetection, SafeZone

logger = logging.getLogger(__name__)
CAMERA = "overhead_camera"


def _now_us() -> int:
    return int(time.time() * 1_000_000)


@dataclass
class PID:
    """PID with derivative low-pass and anti-windup: error (px) -> velocity (ticks/s)."""

    kp: float
    ki: float
    kd: float
    out_min: float = -float("inf")
    out_max: float = float("inf")
    d_tau: float = 0.05  # derivative low-pass time constant (s); marker centers are noisy

    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_error: float | None = field(default=None, init=False, repr=False)
    _d_filtered: float = field(default=0.0, init=False, repr=False)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None
        self._d_filtered = 0.0

    def step(self, error: float, dt: float) -> float:
        if dt <= 0:
            return self._clamp(
                self.kp * error + self.ki * self._integral + self.kd * self._d_filtered
            )

        d_raw = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        alpha = dt / (self.d_tau + dt) if self.d_tau > 0 else 1.0
        self._d_filtered += alpha * (d_raw - self._d_filtered)

        # Anti-windup: don't grow the integral when the unsaturated output would
        # already exceed the clamp in the same direction as `error`.
        unsaturated = self.kp * error + self.ki * self._integral + self.kd * self._d_filtered
        if not (unsaturated >= self.out_max and error > 0) and not (
            unsaturated <= self.out_min and error < 0
        ):
            self._integral += error * dt

        self._prev_error = error
        return self._clamp(
            self.kp * error + self.ki * self._integral + self.kd * self._d_filtered
        )

    def _clamp(self, v: float) -> float:
        return max(self.out_min, min(self.out_max, v))


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
        self._state: dict[str, float] = {}
        self._obs_ts_us = 0
        self._frame = None  # RGB
        op.on_observation(self._on_observation)

    def _on_observation(self, obs: Observation) -> None:
        self._state = dict(obs.state)
        self._obs_ts_us = obs.timestamp_us
        f = obs.frames.get(CAMERA)
        if f is not None:
            self._frame = frame_bytes_to_numpy_rgb(f.data, f.width, f.height)

    # --- primitives (shared with the debug tool) ---

    @property
    def has_state(self) -> bool:
        return all(k in self._state for k in ARM_POS_KEYS)

    @property
    def frame(self):
        return self._frame

    def require_state(self) -> None:
        if not self.has_state:
            missing = [k for k in ARM_POS_KEYS if k not in self._state]
            raise RpcError.Error(
                code=1409, message=f"no robot state yet (missing: {missing})", data=None
            )

    def marker(self) -> Optional[MarkerDetection]:
        return None if self._frame is None else self._detector.detect(self._frame)

    def new_pid(self) -> PID:
        return PID(
            kp=config.PID_KP, ki=config.PID_KI, kd=config.PID_KD, d_tau=config.PID_D_TAU,
            out_min=-config.MAX_VELOCITY, out_max=config.MAX_VELOCITY,
        )

    def send(self, slider_vel: float) -> None:
        """Send the arm (mirrored, held) plus the commanded slider velocity."""
        action = {k: float(self._state[k]) for k in ARM_POS_KEYS}
        action[SLIDER_VEL_KEY] = float(slider_vel)
        self._op.send_action(action, timestamp_us=_now_us(), in_reply_to_ts_us=self._obs_ts_us)

    def send_stop(self) -> None:
        if self.has_state:
            self.send(0.0)

    async def claim(self) -> None:
        await self._op.set_active_operator(self._op.local_identity())

    async def release(self) -> None:
        try:
            await self._op.set_active_operator(None)
        except Exception:
            logger.exception("[move-to] set_active_operator(None) failed")

    def velocity_for(
        self, target_pos: float, marker: MarkerDetection, pid: PID, dt: float
    ) -> tuple[float, float]:
        """Returns ``(slider_vel, error_px)``. Inside the deadzone output is zero
        but the integrator is kept so hovering doesn't reset it."""
        h, w = self._frame.shape[:2]
        error_px = (
            self._safe_zone.target_coord(target_pos) - self._safe_zone.marker_coord(marker, h, w)
        ) * self._safe_zone.axis_length(h, w)
        if abs(error_px) <= config.DEADZONE_PX:
            return 0.0, error_px
        direction = -1.0 if config.INVERT else 1.0
        return direction * pid.step(error_px, dt), error_px

    def current_pos(self, marker: MarkerDetection) -> float:
        h, w = self._frame.shape[:2]
        return self._safe_zone.pos_of(self._safe_zone.marker_coord(marker, h, w))

    # --- one-shot move (operator RPC) ---

    async def servo_to(self, target_pos: float, label: str) -> dict:
        """Drive the marker onto the target line, then stop and release."""
        await self.claim()
        logger.info("[move-to] %s: active operator -> %s", label, self._op.local_identity())

        pid = self.new_pid()
        reached = False
        reason = "timeout"
        iterations = 0
        consecutive_inside = 0
        final_pos: Optional[float] = None
        last_t = time.monotonic()
        t0 = last_t

        try:
            async for _ in pace(self._fps):
                iterations += 1
                now = time.monotonic()
                dt = now - last_t
                last_t = now
                if now - t0 > config.TIMEOUT_S:
                    reason = "timeout"
                    break

                marker = self.marker()
                if marker is None:
                    pid.reset()
                    self.send_stop()
                    consecutive_inside = 0
                    continue

                vel, error_px = self.velocity_for(target_pos, marker, pid, dt)
                self.send(vel)
                final_pos = self.current_pos(marker)

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
                        "[move-to] tick %4d: pos=%.1f target=%.1f err=%+.1fpx vel=%+.0f",
                        iterations, final_pos, target_pos, error_px, vel,
                    )
        finally:
            self.send_stop()
            await self.release()

        elapsed_s = time.monotonic() - t0
        logger.info(
            "[move-to] %s done: reason=%s reached=%s iterations=%d elapsed=%.2fs",
            label, reason, reached, iterations, elapsed_s,
        )
        return {
            "target_pos": target_pos,
            "reached": reached,
            "reason": reason,
            "iterations": iterations,
            "elapsed_s": elapsed_s,
            "final_pos": final_pos,
        }
