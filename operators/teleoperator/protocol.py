"""The recorder's RPC surface — the contract between the two processes.

Both sides import this module so a rename can't drift. Payloads and replies are
JSON objects as strings; every reply carries ``ok``, plus ``error`` when false.

Two caps shape it. A reply must fit one LiveKit RPC payload (~15 KiB), so
``METHOD_EPISODES`` is paginated. And a reply must beat the caller's response
timeout (~10 s), so relabel/delete — whole-corpus rewrites taking seconds to
minutes — are *jobs*: the handler validates, schedules, and returns, and the UI
watches ``busy``/``error``/``revision`` in ``METHOD_STATUS``.

``ATTR_ROLE`` lets the UI find the recorder without being told an identity. Its
prefix is deliberately not ``vla_demo.*``: that is portal-playground's
operator-discovery contract, and a recorder must not look claimable to visitors.
"""
from __future__ import annotations

ATTR_ROLE = "operators.teleoperator.role"
ROLE_RECORDER = "recorder"

# --- read ---------------------------------------------------------------------
# {} -> the full status dict below.
METHOD_STATUS = "recorder_status"
# {"offset": int, "limit": int} -> {"total": int, "offset": int, "episodes": [...]}
METHOD_EPISODES = "recorder_episodes"
# {} -> Portal's own counters: rtt, sync, per-track jitter/evictions. Separate
# from status so the 4 Hz status poll stays lean; the UI asks for these at 1 Hz.
METHOD_METRICS = "recorder_metrics"
# {"episode": int} -> {"videos": {camera: {path, from, to}}, "length": int}.
# On demand rather than in METHOD_EPISODES: per-episode offsets for every camera
# would roughly double a page and push it at the payload cap.
METHOD_EPISODE_VIDEO = "recorder_episode_video"

# --- setup (before anything is open) -----------------------------------------
# {} -> {"ports": [{device, description}], "datasets": [{repo_id, root, episodes}],
#        "defaults": {port, repo_id, task, lerobot_home, local_root}}
# The recorder enumerates these because it is the process with the serial bus and
# the disk; the UI may not even be on the same host.
METHOD_SETUP_OPTIONS = "recorder_setup_options"
# {"port": str, "repo_id": str, "root": str, "task": str} -> {"ok": true}
# Opens the leader arm and the dataset. Returns as soon as the attempt is
# scheduled — opening a serial bus can block for seconds, and lerobot may stop to
# run calibration — so watch `configured`/`opening`/`open_error` in the status.
METHOD_OPEN = "recorder_open"

# --- session control (immediate) ----------------------------------------------
# {} -> {"episode": int}
METHOD_START = "recorder_start"
METHOD_STOP = "recorder_stop"
METHOD_DISCARD = "recorder_discard"
# {"task": str} -> {"task": str}. The default label stamped on new episodes.
METHOD_SET_TASK = "recorder_set_task"
# {} -> {"active": str|None}. Claim/release the robot's active-operator pointer;
# the arm ignores this peer's actions until it holds it.
METHOD_CLAIM = "recorder_claim"
METHOD_RELEASE = "recorder_release"

# --- corpus mutation (scheduled as a job) -------------------------------------
# {"episodes": {"<index>": "new task", ...}} -> {"job": "relabel"}
METHOD_RELABEL = "recorder_relabel"
# {"episodes": [int, ...]} -> {"job": "delete"}
METHOD_DELETE = "recorder_delete"

# ~40 rows of {index, length, seconds, task} fits the payload cap even with long
# task strings; the UI pages through for more.
EPISODE_PAGE_LIMIT = 40

STATUS_KEYS = (
    "identity",        # this recorder's participant identity
    "ready",           # the dataset is open and can record
    "recording",       # an episode is in flight
    "saving",          # a finished episode is still encoding
    "busy",            # human-readable text while a job runs; "" when idle
    "error",           # last job failure; "" when the last job succeeded
    "task",            # default label for new episodes
    "episodes",        # saved episode count
    "rows",            # rows in the in-flight episode
    "dropped",         # rows dropped this episode (stale obs / write error)
    "revision",        # bumps when the episode list changes; drives UI refetch
    "repo_id",
    "root",
    "fps",
    "cameras",         # track names, in contract order
    "robot",           # the robot's identity, or None if it hasn't joined
    "active_operator", # who the robot currently obeys, or None
)
