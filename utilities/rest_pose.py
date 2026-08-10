"""Wire-field names and the parked rest pose, shared by every peer.

The 7-field leslider wire contract (`portal.yaml`) is referenced by the robot,
the positioner, the policy operator, and the motion recorder; the *order* of
`ARM_POS_KEYS` is load-bearing (it must match the checkpoint's positional
normalizer stats), so it lives in exactly one place.

`reset_to_zero_position` (robot/run.py) commands the full rest pose below —
arm and slider.
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

# The 7th wire field: the linear slider, in 0..100 (0 = home/full-retract).
SLIDER_KEY = "slider.pos"

# Every field a full action carries.
ALL_POS_KEYS: tuple[str, ...] = ARM_POS_KEYS + (SLIDER_KEY,)

# Sampled from a parked-but-safe rest pose on the leslider rig.
RESET_POSE_DEFAULTS: dict[str, float] = {
    "shoulder_pan.pos":   1.76,
    "shoulder_lift.pos": -99.87,
    "elbow_flex.pos":    90.73,
    "wrist_flex.pos":    73.01,
    "wrist_roll.pos":     0.04,
    "gripper.pos":       10.0,
    "slider.pos":        50.0,
}
