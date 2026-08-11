"""Tuning constants for the move_to positioner. Edit and restart."""

# "vertical" -> marker moves up/down; "horizontal" -> left/right. Re-run calibrate.py after changing.
AXIS = "vertical"

MARKER_DICT = "DICT_4X4_50"
MARKER_ID = 10

# --- servo: a sqrt decel profile, no PID ---
#
# The commanded velocity is a memoryless function of the current error:
#
#     |v| = clamp(MAX_VELOCITY * sqrt(|e| / APPROACH_FULL_SPEED_PX),
#                 MIN_VELOCITY, MAX_VELOCITY)        for |e| > DEADZONE_PX
#     |v| = 0                                        otherwise
#
# Velocity falling with distance *is* the braking, so the profile self-damps —
# it's the shape a motion controller uses under an acceleration limit. There is
# no integrator, so the residual inside the deadzone never closes: DEADZONE_PX
# sets your placement accuracy directly.
#
# All "px" here are *reference* pixels, not pixels of whatever frame just
# arrived. The stream can renegotiate resolution mid-move (simulcast,
# bandwidth); scaling the error by the live frame width would silently rescale
# the profile and the deadzone with it. Set this to the axis length you
# calibrated at and the tuning holds at any resolution.
REFERENCE_AXIS_PX = 640.0

DEADZONE_PX = 8.0         # within this many reference px of the target, command zero
MIN_VELOCITY = 250.0      # outside the deadzone, never command less than breakaway
MAX_VELOCITY = 2800.0     # hard clamp on |slider.vel| (raw ticks/s)

# Error at which the profile reaches MAX_VELOCITY; it tapers below this and is
# flat above. Raising it makes the approach lazier; lowering it makes the rig
# arrive hotter, and the slider has to shed that speed inside DEADZONE_PX or it
# will overshoot and hunt. Effective stiffness (dv/de) rises as the error
# shrinks, so this is where transport delay bites. Tune with move-to-debug, on
# hardware.
APPROACH_FULL_SPEED_PX = 60.0

INVERT = False           # flip if the camera sees the rail's travel reversed
CONVERGE_TICKS = 3        # consecutive in-deadzone ticks before a move counts as reached
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

# --- position estimator (ArUco fused with slider dead reckoning) ---
# The marker is rigid on the carriage, so `slider.vel` predicts its image motion
# and the estimate coasts through dropouts. Lengths are fractions of the axis.
MAX_COAST_S = 1.5            # dead-reckon this long without a detection, then "lost"
EST_MEAS_STD = 0.004         # ArUco center noise (~2.5px across a 640px axis)
EST_DRIFT_STD_PER_S = 0.02   # coast uncertainty growth per second
EST_GATE_SIGMA = 5.0         # reject detections beyond this many sigma
EST_REACQUIRE_TICKS = 15     # consecutive rejects before trusting the new detection

# Online fit of the slider gain (axis fraction per tick/s). Until it has enough
# evidence, coasting is disallowed and a missing marker just stops the slider.
GAIN_FORGET = 0.99           # exponential forgetting, per fitted interval
GAIN_MIN_SAMPLES = 30        # fitted intervals before coasting is allowed (~1s of motion)
GAIN_MIN_TICKS = 5.0         # |integral of vel| below this contributes nothing

# --- what to do once the coast window expires (see SliderServo.servo_to) ---
# 1. Hold still and wait. Free and safe: a stopped carriage cures motion blur,
#    and the marker cannot wander off on its own.
REACQUIRE_HOLD_S = 1.0
# 2. Optionally creep blind, in the direction the servo was last driving, hoping
#    to carry the marker out of a shadow. Off by default: this is open-loop
#    motion. Bounded by time AND by the dead-reckoned safe-zone edge.
LOST_CREEP_ENABLED = False
LOST_CREEP_VELOCITY = 400.0
LOST_CREEP_MAX_S = 2.0
# Observations stopped arriving at all (robot down, stream dead) -> never creep,
# fail immediately with reason "no_frames".
FRAME_STALE_S = 1.0
