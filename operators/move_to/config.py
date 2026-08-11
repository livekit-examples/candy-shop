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

# --- detection robustness (see ArucoDetector) ---
# Gamma lifts the dark end before equalizing (<1 brightens, 1.0 = off). Note a
# *linear* brightness/gain lift would be pointless: ArUco thresholds each pixel
# against its local mean, so scaling the whole frame is a no-op there. Gamma is
# nonlinear, so it genuinely redistributes contrast in the shadows.
GAMMA = 0.6

# CLAHE equalizes local contrast; recovers low-contrast range without blowing
# out the lit part of the frame the way a global stretch would.
CLAHE_ENABLED = True
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = 8       # NxN tiles

# OpenCV's stock adaptive-threshold sweep (3..23 step 10) assumes even lighting.
# Wider and finer costs a few ms per frame and finds the marker well into shadow.
ADAPTIVE_THRESH_WIN_MIN = 3
ADAPTIVE_THRESH_WIN_MAX = 45
ADAPTIVE_THRESH_WIN_STEP = 6
MIN_MARKER_PERIMETER_RATE = 0.01   # OpenCV default 0.03; accepts a smaller quad
CORNER_REFINE_SUBPIX = True        # tighter centers, so DEADZONE_PX can shrink
