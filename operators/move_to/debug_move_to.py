"""Standalone debug driver for the visual servo: drive the slider directly and
tune the approach profile by eye. Its own operator ("move-to-debug"); does NOT
call the RPC.

Controls (all in the window):

    L-click     set the target to the clicked height
    Up/Down     target +/- 5    Left/Right  target +/- 1
    [ / ]       APPROACH_FULL_SPEED_PX -/+   (lower = brake later = arrives hotter)
    ; / '       MIN_VELOCITY -/+             (breakaway floor)
    , / .       DEADZONE_PX -/+              (placement tolerance)
    space       hold (target = current marker position)
    q / Esc     quit

Watch the last inch: if it overshoots and hunts, raise APPROACH_FULL_SPEED_PX or
widen DEADZONE_PX. Copy what you settle on into config.py.

Only run ONE thing that takes active-operator control at a time.

Usage::

    uv run move-to-calibrate   # once, click bottom/top
    uv run move-to-debug       # robot must be in the room
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
import time

import cv2
import numpy as np

from livekit.portal import Operator, OperatorConfig

from shared.common import env_str, load_env, mint_token, pace, required_env
from shared.config import FPS

from operators.move_to import config
from operators.move_to.servo import SliderServo
from operators.move_to.vision import ArucoDetector, MarkerDetection, SafeZone, load_safe_zone

IDENTITY = "move-to-debug"
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "portal.yaml"
WINDOW = "move_to debug"

_ARROW_UP = {63232, 65362, 2490368}
_ARROW_DOWN = {63233, 65364, 2621440}
_ARROW_LEFT = {63234, 65361, 2424832}
_ARROW_RIGHT = {63235, 65363, 2555904}

# Overlay palette (BGR).
_COLOR_BOTTOM = (0, 128, 255)   # pos 0 bound (orange)
_COLOR_TOP = (0, 220, 0)        # pos 100 bound (green)
_COLOR_TARGET = (0, 150, 255)   # target line
_COLOR_MARKER = (235, 235, 0)   # detected marker
_FONT = cv2.FONT_HERSHEY_SIMPLEX

logger = logging.getLogger(__name__)


def _line(img: np.ndarray, safe_zone: SafeZone, coord_norm: float, color, label: str) -> None:
    h, w = img.shape[:2]
    if safe_zone.axis == "vertical":
        y = int(round(coord_norm * h))
        cv2.line(img, (0, y), (w, y), color, 2, cv2.LINE_AA)
        cv2.putText(img, label, (8, max(16, y - 6)), _FONT, 0.5, color, 1, cv2.LINE_AA)
    else:
        x = int(round(coord_norm * w))
        cv2.line(img, (x, 0), (x, h), color, 2, cv2.LINE_AA)
        cv2.putText(img, label, (x + 4, 16), _FONT, 0.5, color, 1, cv2.LINE_AA)


def _draw_overlay(
    rgb: np.ndarray,
    safe_zone: SafeZone,
    marker: MarkerDetection | None,
    target_pos: float,
    status: str,
) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    _line(img, safe_zone, safe_zone.pos0, _COLOR_BOTTOM, "pos 0")
    _line(img, safe_zone, safe_zone.pos100, _COLOR_TOP, "pos 100")
    _line(img, safe_zone, safe_zone.target_coord(target_pos), _COLOR_TARGET, f"target {target_pos:.0f}")

    if marker is not None:
        poly = marker.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], True, _COLOR_MARKER, 2, cv2.LINE_AA)
        cv2.circle(img, (int(marker.cx), int(marker.cy)), 5, _COLOR_MARKER, -1)

    cv2.rectangle(img, (0, h - 24), (w, h), (0, 0, 0), -1)
    cv2.putText(img, status, (8, h - 7), _FONT, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return img


class DebugSession:
    def __init__(self, op: Operator, servo: SliderServo, safe_zone, fps: int) -> None:
        self._op = op
        self._servo = servo
        self._safe_zone = safe_zone
        self._fps = fps
        self.running = True
        self.target_pos = 50.0
        self.status = "connecting ..."

    def _bump_target(self, delta: float) -> None:
        self.target_pos = self._safe_zone.clamp_pos(self.target_pos + delta)

    def _bump_knob(self, name: str, delta: float) -> None:
        """Live-edit a `config` constant; `velocity_for` reads it every tick."""
        setattr(config, name, round(max(0.0, getattr(config, name) + delta), 3))
        logger.info("[debug] approach=%.1fpx min_vel=%.0f deadzone=%.1fpx",
                    config.APPROACH_FULL_SPEED_PX, config.MIN_VELOCITY, config.DEADZONE_PX)

    async def control_loop(self) -> None:
        await self._servo.claim()
        logger.info("[debug] active operator -> %s", self._op.local_identity())
        self._servo.reset_tracking()
        last_t = time.monotonic()
        try:
            async for _ in pace(self._fps):
                if not self.running:
                    break
                now = time.monotonic()
                dt = now - last_t
                last_t = now
                if not self._servo.has_state or self._servo.frame is None:
                    self.status = "waiting for robot state/frame ..."
                    continue
                est = self._servo.track(dt)
                if not est.usable:
                    self._servo.send_stop()
                    self.status = (
                        f"no marker ({'coast expired' if est.calibrated else 'gain unfitted'}) "
                        f"| target={self.target_pos:.0f}"
                    )
                    continue
                vel, err = self._servo.velocity_for(self.target_pos, est.coord)
                self._servo.send(vel)
                source = "cam" if est.measured else f"coast {est.age_s:.2f}s"
                self.status = (
                    f"pos={self._servo.pos_of(est.coord):.0f} -> {self.target_pos:.0f}  "
                    f"err={err:+.0f}px vel={vel:+.0f}  [{source}]  "
                    f"approach={config.APPROACH_FULL_SPEED_PX:.0f} "
                    f"min={config.MIN_VELOCITY:.0f} dead={config.DEADZONE_PX:.0f}"
                )
        finally:
            self._servo.send_stop()
            await self._servo.release()

    async def ui_loop(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

        def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
            del flags, param
            if event == cv2.EVENT_LBUTTONDOWN and self._servo.frame is not None:
                h, w = self._servo.frame.shape[:2]
                self.target_pos = self._safe_zone.pos_of(
                    self._safe_zone.coord_of_point(x, y, h, w))

        cv2.setMouseCallback(WINDOW, on_mouse)

        while self.running:
            frame = self._servo.frame
            if frame is None:
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(canvas, "connecting to robot ...", (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW, canvas)
            else:
                # Draw what the control loop actually saw; don't re-detect here.
                cv2.imshow(WINDOW, _draw_overlay(
                    frame, self._safe_zone, self._servo.last_marker,
                    self.target_pos, self.status))

            keycode = cv2.waitKeyEx(1)
            key = keycode & 0xFF
            if keycode in _ARROW_UP:
                self._bump_target(5)
            elif keycode in _ARROW_DOWN:
                self._bump_target(-5)
            elif keycode in _ARROW_RIGHT:
                self._bump_target(1)
            elif keycode in _ARROW_LEFT:
                self._bump_target(-1)
            elif key == ord("]"):
                self._bump_knob("APPROACH_FULL_SPEED_PX", 5.0)
            elif key == ord("["):
                self._bump_knob("APPROACH_FULL_SPEED_PX", -5.0)
            elif key == ord("'"):
                self._bump_knob("MIN_VELOCITY", 25.0)
            elif key == ord(";"):
                self._bump_knob("MIN_VELOCITY", -25.0)
            elif key == ord("."):
                self._bump_knob("DEADZONE_PX", 1.0)
            elif key == ord(","):
                self._bump_knob("DEADZONE_PX", -1.0)
            elif key == ord(" "):
                est = self._servo.last_estimate
                if est is not None and est.coord is not None:
                    self.target_pos = self._servo.pos_of(est.coord)
                    logger.info("[debug] hold at pos=%.1f", self.target_pos)
            elif key in (ord("q"), 27):
                self.running = False
                break
            await asyncio.sleep(1.0 / 30.0)

        cv2.destroyAllWindows()
        cv2.waitKey(1)


async def main() -> None:
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    load_env(pathlib.Path(__file__).resolve().parent)

    url = required_env("LIVEKIT_URL")
    room = env_str("LIVEKIT_ROOM", "candy-shop")
    token = mint_token(IDENTITY, room)

    safe_zone = load_safe_zone()
    detector = ArucoDetector(config.MARKER_DICT, config.MARKER_ID)

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
    op = Operator(cfg)
    servo = SliderServo(op, fps=FPS, safe_zone=safe_zone, detector=detector)
    session = DebugSession(op, servo, safe_zone, FPS)

    logger.info("[debug] connecting to %s as '%s' in room '%s' ...", url, IDENTITY, room)
    await op.connect(url, token)
    logger.info("[debug] connected; tracking ArUco %s id %d",
                config.MARKER_DICT, config.MARKER_ID)

    try:
        await asyncio.gather(session.ui_loop(), session.control_loop())
    finally:
        session.running = False
        logger.info("[debug] disconnecting ...")
        try:
            await op.disconnect()
        finally:
            op.close()


def cli() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
