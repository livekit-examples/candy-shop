"""Hardware builders and observation shaping for the leslider rig.

Concentrates everything that's about *the hardware* — serial ports, cameras,
the SO-101 follower and leader — so `robot/run.py` and the teleoperator can
stay focused on the Portal-side flow.

The leslider is an SO-101 arm on a linear slider. The slider runs in extended
(multi-turn) position mode, so every joint is a normalized `.pos` and the
slider reports/commands a unified `slider.pos` in 0..100.
"""
from __future__ import annotations

import os
import platform

import numpy as np
from lerobot.cameras import Cv2Backends, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot_robot_so101_slider_pos import (
    SO101SliderPosFollower,
    SO101SliderPosFollowerConfig,
)
from lerobot_teleoperator_so101_with_slider_pos import (
    SO101WithSliderPosLeader,
    SO101WithSliderPosLeaderConfig,
)

from utilities.common import (
    env_bool,
    env_camera_id,
    env_int,
    env_str,
)
from utilities.ports import ENV_VAR as LEADER_PORT_ENV_VAR
from utilities.ports import resolve_leader_port

CAMERAS: tuple[str, ...] = ("arm_camera", "overhead_camera")


def build_follower(fps: int) -> SO101SliderPosFollower:
    """Construct the SO-101 slider follower from environment variables.
    """
    width = env_int("LESLIDER_CAM_WIDTH", 640)
    height = env_int("LESLIDER_CAM_HEIGHT", 480)
    cam_defaults = {
        "arm_camera": ("LESLIDER_CAM_ARM", "/dev/video0"),
        "overhead_camera": ("LESLIDER_CAM_OVERHEAD", "/dev/video2"),
    }
    backend = Cv2Backends.V4L2 if platform.system() == "Linux" else Cv2Backends.ANY
    cameras = {
        name: OpenCVCameraConfig(
            index_or_path=env_camera_id(env_var, default),
            fps=fps,
            width=width,
            height=height,
            rotation=Cv2Rotation.NO_ROTATION,
            fourcc="MJPG",
            backend=backend,
        )
        for name, (env_var, default) in cam_defaults.items()
    }
    return SO101SliderPosFollower(
        SO101SliderPosFollowerConfig(
            id=env_str("LESLIDER_ID", "leslider"),
            port=env_str("LESLIDER_PORT", "/dev/ttyACM0"),
            cameras=cameras,
            slider_goal_speed=env_int("LESLIDER_SLIDER_GOAL_SPEED", 2000),
            read_current=False,
        )
    )


def build_leader(port: str | None = None) -> SO101WithSliderPosLeader:
    """Construct the SO-101 leader arm + slider keyboard handler.
    """
    return SO101WithSliderPosLeader(
        SO101WithSliderPosLeaderConfig(
            id=env_str("SO101_LEADER_ID", "so101_leader"),
            port=resolve_leader_port(port, env_port=os.getenv(LEADER_PORT_ENV_VAR)),
            invert_direction=env_bool("LESLIDER_INVERT_SLIDER", default=False),
        )
    )


def split_state_frames(
    observation: dict,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Split one follower observation into Portal state and video frames.
    """
    state = {
        key: float(value)
        for key, value in observation.items()
        if key.endswith(".pos") or key.endswith(".vel")
    }
    frames = {
        camera: observation[camera]
        for camera in CAMERAS
        if camera in observation and observation[camera] is not None
    }
    return state, frames
