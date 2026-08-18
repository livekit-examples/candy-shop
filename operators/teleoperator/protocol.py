"""The recorder's RPC surface — the contract between the two processes.

Both sides import this module so a rename can't drift. Payloads/replies are JSON
strings; every reply carries ``ok`` (plus ``error`` when false). Two caps shape
it: a reply must fit one LiveKit RPC payload (~15 KiB), so ``METHOD_EPISODES`` is
paginated; and a reply must beat the ~10 s response timeout, so relabel/delete
run as jobs the UI watches via ``busy``/``error``/``revision`` in the status.
"""
from __future__ import annotations

# Lets the UI find the recorder without an identity. Not ``vla_demo.*``: that is
# portal-playground's discovery contract, and a recorder must not look claimable.
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
# {} -> {"active": str, "suspended": [str]}. Take the arm: preempt every peer
# operator first (an operator whose loop is still alive would re-claim on its next
# tick), remember what was preempted for METHOD_RESUME, then point the robot here.
METHOD_CLAIM = "recorder_claim"
# {} -> {"active": None}. Drop the pointer without restarting anything.
METHOD_RELEASE = "recorder_release"
# {} -> {"resumed": [str]}. Hand the arm back: re-issue the runs the claim
# preempted, with their original payloads, and stop asserting the claim.
METHOD_RESUME = "recorder_resume"

# --- peer operators (the rest of the room) ------------------------------------
# {"identity": str, "payload": str} -> {"identity": str}. Fire that operator's own
# run RPC (`run_task`, `run_policy`, `move_to`) and watch it in the background: the
# call outlives this reply by design — `run_policy` returns only when preempted —
# so progress shows up in `peers`, never in the reply.
METHOD_PEER_RUN = "recorder_peer_run"
# {"identity": str} -> {"identity": str}. That operator's stop RPC.
# {} (no identity) -> stop every declared peer, orchestrators first, then fold the
# arm with the robot's own reset. The panic path.
METHOD_PEER_STOP = "recorder_peer_stop"

# --- the leader arm -----------------------------------------------------------
# {"enabled": bool} -> {"mimic": {...}}. Drive the leader arm from the follower's
# observed pose, so the human feels what the policy is doing and can take the arm
# from a matched pose instead of snapping it. Refused until the leader is open.
METHOD_MIMIC = "recorder_mimic"
# {} -> {"mimic": {...}}. Switch mimic off and drop the leader's torque outright.
# The escape hatch: a half-failed engage leaves the arm stiff with mimic reporting
# `error`, and there the toggle has nothing to switch off.
METHOD_RELAX = "recorder_relax"
# {} -> {"leader": {...}}. Re-open the leader's serial bus without touching the
# open dataset. The recorder retries a dropped link on its own; this jumps the
# backoff once the cable is back in. Also the way out of a failed first open.
METHOD_RECONNECT = "recorder_reconnect"

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
    "dropped",         # rows dropped this episode, all causes
    "drop_causes",     # {unpaired, error, backlog} -> count, this episode
    "resized",         # frames rescaled to the dataset's resolution this episode
    "obs_fps",         # observed observation rate
    "queue_depth",     # rows waiting on the writer thread
    "pairing_age_ms",  # [last, worst] action-to-observation gap this episode
    "revision",        # bumps when the episode list changes; drives UI refetch
    "repo_id",
    "root",
    "fps",
    "cameras",         # track names, in contract order
    "robot",           # the robot's identity, or None if it hasn't joined
    "active_operator", # who the robot currently obeys, or None
    "claiming",        # this teleoperator is holding (and re-asserting) the arm
    "peers",           # peer operators; see PEER_KEYS
    "suspended",       # identities a claim preempted, waiting on METHOD_RESUME
    "mimic",           # the mimic toggle's own state; see MIMIC_KEYS
    "leader",          # the leader arm's link; see LEADER_KEYS
)

# One entry per peer operator the room can offer: everything `shared.operators`
# declares, plus any live operator it doesn't (which has no RPCs to drive, so the UI
# shows it as presence only). The descriptors themselves are not on the wire — both
# processes import `shared.operators` — so this carries only what changes.
PEER_KEYS = (
    "identity",
    "online",          # in the room right now
    "declared",        # `shared.operators` knows how to drive it
    "active",          # the robot is obeying this one
    "running",         # a run RPC we issued is still open
    "payload",         # what we last asked it to do
    "error",           # why its last run failed; "" if it didn't
    "result",          # one-line summary of its last completed run
    "elapsed_s",       # how long the open run has been going
)

# The leader's serial link, which outlives any one handle on it: a dropped bus is a
# reconnect, not a re-setup, so `configured` stays true across one and the corpus stays
# open. `state` is what the window reads to decide between a badge and a banner.
LEADER_KEYS = (
    "connected",       # the bus is open and answering right now
    "port",            # serial port this session opened
    "state",           # open | reconnecting | down
    "detail",          # why it dropped, or what the retry is doing
    "attempts",        # reconnect attempts since the link dropped
)

MIMIC_KEYS = (
    "enabled",         # the toggle
    "state",           # off | waiting | aligning | tracking | holding | yielded | error
    "detail",          # why it is waiting, or what went wrong
)
