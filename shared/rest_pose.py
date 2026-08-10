"""Wire-field names and the parked rest pose for the leslider (`portal.yaml`), shared by every peer."""
from __future__ import annotations

# Wire order from portal.yaml. Load-bearing: positional against the checkpoint's
# normalizer stats. Keep in exactly one place.
ARM_POS_KEYS: tuple[str, ...] = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

# Linear slider, commanded as raw ticks/s velocity (sign-magnitude; 0 = stop).
SLIDER_VEL_KEY = "slider.vel"

ALL_ACTION_KEYS: tuple[str, ...] = ARM_POS_KEYS + (SLIDER_VEL_KEY,)

# Arm folded to a safe pose, slider stopped.
RESET_POSE_DEFAULTS: dict[str, float] = {
    "shoulder_pan.pos":   1.76,
    "shoulder_lift.pos": -99.87,
    "elbow_flex.pos":    90.73,
    "wrist_flex.pos":    73.01,
    "wrist_roll.pos":     0.04,
    "gripper.pos":       10.0,
    "slider.vel":         0.0,
}
