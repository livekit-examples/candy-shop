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
# run_task caps a task at the reward operator's 30s safety timeout, then still has
# to stop the policy and unwind; give the RPC a little headroom over that 30s so it
# returns its summary rather than being aborted from this side.
RUN_TASK_TIMEOUT_S = 35.0

# Named waypoints mapped to raw positions the move service understands.
POSITIONS = {
    "candy shelf": 20,
    "drop zone": 70,
}

# Served through LiveKit Inference (no per-provider API keys required).
STT_MODEL = "deepgram/nova-3"
LLM_MODEL = "google/gemma-4-31b-it"
TTS_MODEL = "inworld/inworld-tts-2"
TTS_VOICE = "Ashley"
