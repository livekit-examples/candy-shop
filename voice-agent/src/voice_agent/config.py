"""Shared configuration for the candy shop agent."""

# Participant / track identities in the LiveKit room.
ROBOT_IDENTITY = "robot"
VIDEO_TRACK_NAME = "overhead_camera"

# Identities of the robot control services we drive over RPC.
MOVE_TO_IDENTITY = "move-to-operator"
RUN_POLICY_IDENTITY = "policy-operator"
RPC_TIMEOUT_S = 10.0

# Named waypoints mapped to the raw positions the move service understands.
POSITIONS = {
    "candy shelf": 30,
    "drop zone": 80,
}

# Models, served through LiveKit Inference (no per-provider API keys required).
STT_MODEL = "deepgram/nova-3"
LLM_MODEL = "google/gemma-4-31b-it"
TTS_MODEL = "inworld/inworld-tts-2"
TTS_VOICE = "Ashley"
