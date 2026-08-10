"""Portal robot process for the physical leslider follower (the candy shop rig).

Publishes both cameras + the state each tick (6 arm `.pos` + `slider.vel`) and
applies the active operator's action. The slider runs in velocity mode.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import time

from livekit.portal import (
    Action,
    PortalError,
    Robot,
    RobotConfig,
    RpcInvocationData,
)

from utilities.common import env_str, load_env, mint_token, pace, required_env
from utilities.leslider import CAMERAS, build_follower, split_state_frames
from utilities.rest_pose import RESET_POSE_DEFAULTS

IDENTITY = "robot"
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "portal.yaml"

logger = logging.getLogger(__name__)


async def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_env(pathlib.Path(__file__).resolve().parent)

    url = required_env("LIVEKIT_URL")
    room = env_str("LIVEKIT_ROOM", "candy-shop")
    token = mint_token(IDENTITY, room)

    cfg = RobotConfig.from_yaml_file(CONFIG_PATH, room)
    fps = cfg.fps

    follower = build_follower(fps)
    robot = Robot(cfg)
    latest_action: dict[str, float] = {}

    def on_action(action: Action) -> None:
        nonlocal latest_action
        latest_action = {key: float(value) for key, value in action.values.items()}

    def on_operator_left(identity: str) -> None:
        if identity == robot.active_operator():
            logger.info("[robot] active operator '%s' left; releasing", identity)
            asyncio.create_task(robot.set_active_operator(None))

    async def reset_to_zero_position(data: RpcInvocationData) -> str:
        """Fold the arm to its safe pose and stop the slider (`slider.vel = 0`).

        Claims active operator so no live stream overrides it; doubles as the
        preempt-anything cancel path.
        """
        nonlocal latest_action
        logger.info("[robot] reset_to_zero_position from '%s'", data.caller_identity)
        await robot.set_active_operator(robot.local_identity())
        latest_action = dict(RESET_POSE_DEFAULTS)
        return json.dumps({"target": RESET_POSE_DEFAULTS})

    robot.on_action(on_action)
    robot.on_operator_left(on_operator_left)
    robot.on_active_operator_changed(
        lambda identity: logger.info("[robot] active operator now: %s", identity)
    )
    robot.register_rpc_method("reset_to_zero_position", reset_to_zero_position)

    follower.connect()
    logger.info("[robot] connecting to %s as '%s' in room '%s' ...", url, IDENTITY, room)
    await robot.connect(url, token)
    logger.info(
        "[robot] connected; cameras=%s; streaming at %d fps; ctrl-c to stop",
        list(CAMERAS),
        fps,
    )

    try:
        async for _ in pace(fps):
            ts_us = int(time.time() * 1_000_000)

            state, frames = split_state_frames(follower.get_observation())
            robot.send_state(state, timestamp_us=ts_us)

            for name in CAMERAS:
                frame = frames.get(name)
                if frame is None:
                    continue
                try:
                    robot.send_video_frame(name, frame, timestamp_us=ts_us)
                except PortalError as e:
                    logger.warning("[robot] send_video_frame('%s') failed: %s", name, e)

            if latest_action:
                follower.send_action(latest_action)

    except KeyboardInterrupt:
        logger.info("[robot] stopping ...")
    finally:
        logger.info("[robot] disconnecting...")
        try:
            await robot.disconnect()
        finally:
            robot.close()
            follower.disconnect()


def cli() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
