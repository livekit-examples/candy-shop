"""Tuning constants shared by the robot runtime and the operators."""

# Tick rate for the robot loop and every operator's control loop, in Hz.
# Portal's config has a setter but no getter, so `cfg.fps` cannot be read back
# from portal.yaml — keep this in sync with the `fps:` key in that file.
FPS = 30
