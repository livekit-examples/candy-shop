"""Operator side: the candy shop's positioner (visual-servo, velocity mode).

Joins the leslider room as an Operator peer and serves one RPC:

``move_to(pos)``
  ``pos`` is 0..100 — bottom to top of the calibrated safe zone in the overhead
  image. The operator detects the robot's on-board ArUco marker and runs a PID
  loop that commands ``slider.vel`` until the marker sits on the target line
  (then stops). **This is the whole order path.** The candy shop works from two
  fixed stations, so the agent drives to two numbers (``POSITIONS`` in the
  agent's config); which candy gets picked is decided by the policy operator.

Parking the rig is deliberately *not* here: the robot owns
``reset_to_zero_position`` (see ``robot/run.py``), which the agent calls
directly and which doubles as the preempt-anything cancel.

Tuning (PID gains, marker id, deadzone, ...) lives in ``config.py``. Only
``LIVEKIT_*`` and ``LIVEKIT_ROOM`` come from the environment.

Usage::

    uv run python operators/move_to/calibrate.py  # once, click the two bound lines
    uv run move-to
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib

from livekit.portal import Operator, OperatorConfig, RpcError, RpcInvocationData

from utilities.common import env_str, load_env, mint_token, required_env

from operators.move_to import config
from operators.move_to.servo import SliderServo
from operators.move_to.vision import ArucoDetector, SafeZone, load_safe_zone

IDENTITY = "move-to-operator"
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "portal.yaml"

logger = logging.getLogger(__name__)


def _payload_number(data: RpcInvocationData) -> float:
    """Parse a position from a bare number or ``{"position": N}`` JSON."""
    payload = (data.payload or "").strip()
    if not payload:
        raise RpcError.Error(code=1400, message="empty position", data=None)
    if payload.startswith("{"):
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            raise RpcError.Error(code=1400, message="payload was not valid JSON", data=None)
        for key in ("position", "pos"):
            if (value := obj.get(key)) is not None:
                payload = str(value)
                break
        else:
            raise RpcError.Error(code=1400, message="JSON payload had no position/pos", data=None)
    try:
        return float(payload)
    except ValueError:
        raise RpcError.Error(
            code=1400, message=f"position must be a number, got {payload!r}", data=None
        )


async def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_env(pathlib.Path(__file__).resolve().parent)

    url = required_env("LIVEKIT_URL")
    room = env_str("LIVEKIT_ROOM", "candy-shop")
    token = mint_token(IDENTITY, room)

    safe_zone = load_safe_zone()
    detector = ArucoDetector(config.MARKER_DICT, config.MARKER_ID)
    logger.info("[move-to] tracking ArUco %s id %d", config.MARKER_DICT, config.MARKER_ID)

    cfg = OperatorConfig.from_yaml_file(CONFIG_PATH, room)
    op = Operator(cfg)
    servo = SliderServo(op, fps=cfg.fps, safe_zone=safe_zone, detector=detector)

    async def move_to(data: RpcInvocationData) -> str:
        """Servo the marker to ``pos`` (0..100, bottom..top of the safe zone)."""
        logger.info("[move-to] move_to RPC from '%s'", data.caller_identity)
        requested = _payload_number(data)
        servo.require_state()

        target = SafeZone.clamp_pos(requested)
        if target != requested:
            logger.info("[move-to] move_to(%.1f) capped to safe zone -> %.1f", requested, target)

        outcome = await servo.servo_to(target, f"move_to({target:.1f})")
        return json.dumps({"requested": requested, "capped": target != requested, **outcome})

    op.register_rpc_method("move_to", move_to)
    op.on_operator_joined(lambda i: logger.info("[move-to] operator joined: %s", i))
    op.on_operator_left(lambda i: logger.info("[move-to] operator left: %s", i))

    logger.info("[move-to] connecting to %s as '%s' in room '%s' ...", url, IDENTITY, room)
    await op.connect(url, token)
    logger.info("[move-to] connected as '%s'; awaiting move_to RPCs", op.local_identity())

    try:
        # No tick loop: the servo runs only inside an active RPC.
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("[move-to] stopping ...")
    finally:
        try:
            await op.disconnect()
        finally:
            op.close()


def cli() -> None:
    """Console-script entry point (`uv run move-to`)."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
