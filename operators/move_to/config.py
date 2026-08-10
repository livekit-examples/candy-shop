"""Tuning constants for the move_to positioner. Edit and restart."""

# "vertical" -> marker moves up/down; "horizontal" -> left/right. Re-run calibrate.py after changing.
AXIS = "vertical"

MARKER_DICT = "DICT_4X4_50"
MARKER_ID = 10

# PID gains: image error (pixels along AXIS) -> slider velocity (raw ticks/s).
PID_KP = 12.0
PID_KI = 0.0
PID_KD = 1.0
PID_D_TAU = 0.05          # derivative low-pass time constant (seconds)

# Servo behavior.
DEADZONE_PX = 12.0        # within this many px of the target line, command zero
MAX_VELOCITY = 2000.0     # hard clamp on |slider.vel| (raw ticks/s)
INVERT = False            # flip if the camera sees the rail's travel reversed
CONVERGE_TICKS = 5        # consecutive in-deadzone ticks before a move counts as reached
TIMEOUT_S = 20.0          # give up (and stop) after this many seconds
