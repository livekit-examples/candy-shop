"""Shared configuration for the candy shop agent."""

ROBOT_IDENTITY = "robot"
VIDEO_TRACK_NAME = "overhead_camera"

MOVE_TO_IDENTITY = "move-to-operator"
# The reward operator wraps the policy: its run_task drives the policy and watches
# SARM for completion, so the agent talks to it rather than the policy directly.
REWARD_IDENTITY = "reward-operator"
# Teardown only, never to start work. Stopping reward already cascades here, but
# that misses a policy started directly from the Playground console. Safe to send
# while idle: `pick()` clears its stop flag on entry.
POLICY_IDENTITY = "policy-operator"
RPC_TIMEOUT_S = 10.0
# run_task retries a failed pick on escalating budgets (10+15+20s by default = 45s
# of picking), and each attempt still has to stop the policy and unwind; give the
# RPC headroom over that total so it returns its summary rather than being aborted
# from this side. Raise this in step with the reward operator's --attempt-budgets.
RUN_TASK_TIMEOUT_S = 60.0

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
LLM_MODEL = "google/gemini-3.6-flash"
TTS_MODEL = "fishaudio/s2.1-pro"
TTS_VOICE = "9a9cf47702da476aa4629e2506d4a857"