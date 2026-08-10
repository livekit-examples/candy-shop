"""Wire-field names and the parked rest pose, shared by every peer.

The 7-field leslider wire contract (`portal.yaml`) is referenced by the robot,
the move_to operator, and any policy operator. The *order* of `ARM_POS_KEYS` is
load-bearing (it must match the checkpoint's positional normalizer stats), so it
lives in exactly one place.

The slider runs in velocity mode: it carries `slider.vel` (raw ticks/s), not a
position. So a full action is the six arm `.pos` goals plus one `slider.vel`.
`reset_to_zero_position` (robot/run.py) commands the arm rest pose and stops the
slider (`slider.vel = 0`).
"""
from __future__ import annotations

# 6 arm motor positions, in the wire order from portal.yaml. Load-bearing:
# positional against the checkpoint's normalizer stats.
ARM_POS_KEYS: tuple[str, ...] = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

# The 7th wire field: the linear slider, commanded as a raw ticks/s velocity
# (sign-magnitude; 0 = stop).
SLIDER_VEL_KEY = "slider.vel"

# Every field a full action carries.
ALL_ACTION_KEYS: tuple[str, ...] = ARM_POS_KEYS + (SLIDER_VEL_KEY,)

# Parked-but-safe rest pose: the arm folded to a safe pose, slider stopped.
RESET_POSE_DEFAULTS: dict[str, float] = {
    "shoulder_pan.pos":   1.76,
    "shoulder_lift.pos": -99.87,
    "elbow_flex.pos":    90.73,
    "wrist_flex.pos":    73.01,
    "wrist_roll.pos":     0.04,
    "gripper.pos":       10.0,
    "slider.vel":         0.0,
}
