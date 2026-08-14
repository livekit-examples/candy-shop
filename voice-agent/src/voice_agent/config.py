"""Shared configuration for the candy shop agent."""

ROBOT_IDENTITY = "robot"
VIDEO_TRACK_NAME = "overhead_camera"

# How the Playground site marks a browser visitor apart from the robot and the
# operators, who share the room and are all STANDARD participants too. A wire
# contract with `web/components/playground/livekit/control.ts`; see visitor.py.
VISITOR_KIND_ATTR = "vla_demo.kind"
VISITOR_KIND = "viewer"
# Epoch ms the visitor's control turn ends. Only the holder is granted a mic.
CONTROL_UNTIL_ATTR = "vla_demo.control_until"

MOVE_TO_IDENTITY = "move-to-operator"
# The reward operator wraps the policy: its run_task drives the policy and watches
# SARM for completion, so the agent talks to it rather than the policy directly.
REWARD_IDENTITY = "reward-operator"
# Teardown only, never to start work. Stopping reward already cascades here, but
# that misses a policy started directly from the Playground console. Safe to send
# while idle: `pick()` clears its stop flag on entry.
POLICY_IDENTITY = "policy-operator"
# One minute floor on every RPC. These cross a relay to an operator that may be mid-
# motion, and an aborted RPC leaves the caller guessing while the robot keeps moving —
# far worse than waiting. move_to alone can run its own TIMEOUT_S of 20s, which the old
# 10s here would have cut off mid-travel.
RPC_TIMEOUT_S = 60.0
# What run_task can cost, worst case, and why this is the number: the reward operator
# caps a whole task at its --timeout (90s, itself above the 15+20+25s of pick budgets
# plus a pause between each), and then pays one last policy unwind on the way out of
# 2x --policy-timeout (2x10s). 90 + 20 = 110, so 120 leaves a little slack. The pick
# budgets alone are NOT the total — that reading is what left this at 120 while the
# unwind allowance was 60s a side, i.e. an uncapped 400s+ of arm time the agent would
# abandon mid-motion. Move all three together (REWARD_ATTEMPT_BUDGETS,
# REWARD_TIMEOUT_S, this) or the agent gives up while the arm is still working.
RUN_TASK_TIMEOUT_S = 120.0

# Named waypoints mapped to raw positions the move service understands.
POSITIONS = {
    "candy shelf": 20,
    "drop zone": 70,
}

# The instruction strings SmolVLA was fine-tuned on, verbatim from the dataset's
# meta/tasks.parquet. The policy is language-conditioned, so a paraphrase — a
# capital letter, a missing article, a candy it never saw — is off-distribution
# and picks worse. Send these exactly; never build an instruction by f-string.
PICK_TASKS = {
    "kitkat": "pick up a kitkat",
    "nerd": "pick up a nerd",
    "twix": "pick up a twix",
    "snicker": "pick up a snicker",
}

# One fixed drop instruction, and it never names the candy — the policy was
# trained to put whatever it is holding into the black circle.
DROP_TASK = "drop candy into the black circle"

# Served through LiveKit Inference (no per-provider API keys required).
STT_MODEL = "deepgram/nova-3"
LLM_MODEL = "google/gemma-4-31b-it"
TTS_MODEL = "fishaudio/s2.1-pro"
TTS_VOICE = "9a9cf47702da476aa4629e2506d4a857"
