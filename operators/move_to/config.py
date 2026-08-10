"""Configuration for the move_to positioner operator.

Plain constants (like voice-agent's config.py). Edit here and restart; the debug
tool can also nudge the PID gains live so you can find good values to copy back.
"""

# Which image axis the slider appears to travel along in the overhead camera —
# set it to match how the camera is mounted:
#   "vertical"   -> marker moves up/down; safe zone = two horizontal lines
#   "horizontal" -> marker moves left/right; safe zone = two vertical lines
# Changing this means re-running calibrate.py.
AXIS = "vertical"

# ArUco marker carried by the robot, seen in the overhead camera.
MARKER_DICT = "DICT_4X4_50"
MARKER_ID = 10

# PID gains: image error (pixels along AXIS) -> slider velocity (raw ticks/s).
# Sized for a ~480px (vertical) / ~640px (horizontal) frame and MAX_VELOCITY
# below: kp ~= max_vel / 170 saturates around 170px of error and tapers to a
# gentle crawl near the target; kd damps the overshoot a fast slider would add.
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
