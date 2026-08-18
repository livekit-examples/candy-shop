"""Mimic: drive the leader arm from the follower's pose while something else has the arm.

Normally the SO-101 leader is dead weight in the human's hand — torque off, positions
read, nothing written back. This module inverts that for as long as another operator is
driving: torque on, and every tick the leader's six joints are commanded to wherever the
follower actually is. Two things fall out of it:

* You feel the policy. The leader moves through the same trajectory as the arm, which is
  the difference between watching a pick and supervising one.
* **Takeover stops being a jump.** Claiming the arm hands the robot whatever pose the
  leader is in; from a matched pose that is a no-op, and from a leader lying on the desk
  it is a full-speed slew to it. Mimic is what makes intervening mid-policy safe.

Taking over is always explicit — the claim hotkey, the window's button, the ``claim``
RPC. Mimic reads no intent off the leader. It used to: a gap between where the leader was
and where it had been told to be, held past a threshold, was treated as a hand asking for
the arm. But a servo chasing a moving goal runs behind it by itself, so a fast policy
motion looked like a hand and took the arm mid-pick — and every attempt to tell the two
apart (judging against an older goal, ignoring joints that are catching up) is a
heuristic standing between a policy and a human's actual intent. A button says it
without guessing.

Both arms report normalized positions in the same units (degrees per joint, 0-100 on the
gripper, each against its own calibration), which is what makes the follower's state
usable as a leader goal — the same mapping ordinary teleop already relies on, pointed the
other way.

**Taking the arm frees the leader**, because every takeover from here is a human
stepping into a policy run and a torqued leader cannot be flown — it fights the hand, and
the robot follows the fight. Torque therefore drops on a claim, on the toggle going off,
and on `release` (the escape hatch a human asks for by name); after a claim mimic stays
off the leader until the arm goes back to a peer, since re-engaging would torque it up
under the hand that just took it.

The cost is the hazard that buys: an SO-101 leader falls under gravity, and while this
teleoperator holds the arm the follower goes down with it. Claiming means a hand on the
leader.

Every entry point that talks to the bus can find it gone: a leader is one USB cable, and
unplugging it (or lerobot's own ESC handler) closes the port under this. Those failures
park mimic in `error` with the cause on `failure`, which the caller reads as the leader
link being down — see `run.lost_leader`. `forget` is how it comes back: state
dropped without a single write to a port that is no longer there.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# The six leader joints. `<joint>.pos` on the wire, bare names on the leader's bus.
JOINTS: tuple[str, ...] = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
)
# 0-100 travel, not degrees, so it does not share the arm's alignment reference.
GRIPPER = "gripper"

# Alignment budget for the worst case: the leader can be parked a long way from wherever
# the arm is, and torque coming on is the one moment it moves on its own. Sized like the
# policy's start ramp — a cap on slew rate, so a leader already near the arm aligns in a
# fraction of it.
ALIGN_REFERENCE_DEG = 60.0

# Time constant of the goal filter. The follower's state crosses a network: it jitters,
# and repeats a value whenever a tick outruns the stream. Writing that straight through
# is what makes the leader step rather than move. It costs lag, which nothing reads any
# more now that the leader's position is not evidence of anything.
SMOOTH_S = 0.08


class MimicController:
    """Leader-follows-follower, and the release that hands the arm to a human.

    Owns nothing it did not open: the leader belongs to the recorder's runtime, so every
    entry point tolerates it being absent (setup not finished) or dead (bus error).
    """

    def __init__(
        self,
        *,
        fps: int,
        align_s: float = 1.5,
        smooth_s: float = SMOOTH_S,
        torque_limit: int = 0,
    ) -> None:
        self._fps = fps
        self._align_s = max(align_s, 0.0)
        # One-pole low-pass on the goal, as a per-tick coefficient.
        self._smooth = (1.0 - math.exp(-1.0 / (fps * smooth_s))) if smooth_s > 0 else 1.0
        # 0..1000 (per mille of the motor's rated torque); 0 leaves the motor's own limit
        # alone. A lower limit is what keeps the leader yielding to a hand resting on it
        # instead of fighting back, at the cost of tracking the arm less crisply.
        self._torque_limit = max(min(int(torque_limit), 1000), 0)

        self.enabled = False
        self._engaged = False      # torque is on
        self._holding = False      # engaged but frozen: the arm turned out to be ours
        self._yielded = False      # freed for a human; stay off the leader until the
                                   # claim they were freed for has actually landed
        self._origin: dict[str, float] = {}   # leader pose when torque came on
        self._align_ticks = 0
        self._tick = 0             # ticks written since engaging
        self._goal: dict[str, float] = {}     # last goal written, the filter's history
        self._detail = ""
        self._failed = ""

    # --- view ---------------------------------------------------------------

    @property
    def engaged(self) -> bool:
        """Torque is on and the leader is ours to command."""
        return self._engaged

    @property
    def holding(self) -> bool:
        """Engaged but frozen: the arm is ours and the leader is parked at a pose."""
        return self._holding

    @property
    def failure(self) -> str:
        """Why mimic parked, or "". Only ever a bus error, so the caller treats it as
        the leader link being down rather than as something mimic can retry."""
        return self._failed

    def snapshot(self) -> dict:
        if self._failed:
            state, detail = "error", self._failed
        elif not self.enabled:
            state, detail = "off", ""
        elif self._holding:
            state, detail = "holding", "this teleoperator has the arm and the leader is " \
                                       "torqued — switch mimic off to fly"
        elif self._yielded:
            state, detail = "yielded", "leader is yours — waiting for the claim to land"
        elif not self._engaged:
            state, detail = "waiting", self._detail
        elif self._tick < self._align_ticks:
            state, detail = "aligning", "moving the leader onto the arm's pose"
        else:
            state, detail = "tracking", "leader is following the arm — Take arm to fly it"
        return {
            "enabled": self.enabled,
            "state": state,
            "detail": detail,
        }

    # --- control ------------------------------------------------------------

    def set_enabled(self, enabled: bool, leader) -> None:
        """The toggle. Switching it off frees the leader: the human asked for it, so a
        hand is on it."""
        self.enabled = bool(enabled)
        self._failed = ""
        if not self.enabled:
            self.free(leader)

    def yield_to_human(self, leader) -> None:
        """Hand the leader to the human who is taking the arm: torque off, and stay off.

        Called as a claim *starts*, which is seconds before it lands — every peer has to
        answer its stop first. `update` would see the arm still belonging to a peer in
        that gap and torque the leader straight back up under the hand, so freeing alone
        is not enough; the flag is what makes it stick until the claim shows up.
        """
        self.free(leader)
        self._yielded = True

    def hold(self) -> None:
        """Freeze the goal, keep the torque. No-op when nothing is engaged.

        The fallback for the arm becoming ours without a yield: a claim through
        `yield_to_human` has already freed the leader, but a leader that both drives the
        follower and is driven from it is a feedback loop, so something has to stop
        writing goals either way.
        """
        if not self._engaged or self._holding:
            return
        self._holding = True
        logger.info("[mimic] leader held: the arm is this teleoperator's")

    def release(self, leader) -> None:
        """Switch mimic off *and* drop the leader's torque, whatever mimic believed
        about it. The escape hatch for a stiff leader.

        The toggle covers the ordinary case, but not the one where it is needed: an
        engage that failed after the torque write leaves the arm stiff with mimic
        reporting `error`, and there is nothing there for a toggle to switch off. Safe
        because a human is what asks for this, and a human asking for a loose leader has
        their hand on it.
        """
        self.enabled = False
        self._failed = ""
        self.free(leader)

    def forget(self) -> None:
        """Drop every belief about a leader that is gone, without touching a bus.

        For a link that dropped: the port is closed, so a torque write would only raise,
        and motors on a dead bus hold nothing anyway. Mimic comes back *off* — re-arming
        it torques the leader, and that is the human's call once the arm is back in
        their hand.
        """
        self.enabled = False
        self._engaged = self._holding = False
        self._failed = ""
        self._reset()

    def free(self, leader) -> None:
        """Drop torque and stop commanding. The leader is loose the moment this returns.

        The write goes out even when mimic believes nothing is engaged: an engage that
        failed after `enable_torque` leaves a stiff leader mimic has no record of, and
        one redundant packet is cheaper than that. An earlier failure keeps its message —
        it is the one that says what actually went wrong.
        """
        self._engaged = self._holding = False
        self._reset()
        if leader is None:
            return
        try:
            leader.bus.disable_torque()
        except Exception as exc:
            self._failed = self._failed or f"could not release the leader: {exc}"
            logger.exception("[mimic] disabling leader torque failed")
        else:
            logger.info("[mimic] leader released")

    def update(self, leader, follower_state: Optional[dict[str, float]],
               *, allowed: bool) -> None:
        """One tick: engage if needed, then ramp onto the arm's pose and track it.

        `allowed` is the caller's rule on whether mimicking makes sense right now —
        false while this teleoperator drives the arm itself, since the leader cannot both
        be driven and be doing the driving. A leader already engaged is held rather than
        released, because releasing is what drops it.
        """
        if leader is None or not self.enabled or self._failed:
            return
        if not allowed:
            # The claim landed, so a yield has served its purpose.
            self._yielded = False
            self.hold()
            self._detail = "this teleoperator is driving; mimic waits for a handover"
            return
        if self._yielded:
            return
        if self._holding:
            # The handover is over — the arm went back to somebody else, so track again
            # from wherever the leader was left.
            self._holding = False
            self._engaged = False

        target = self._targets(follower_state)
        if target is None:
            # Losing the arm's state mid-track holds the last goal rather than releasing:
            # releasing drops a torqued arm out of the sky.
            self._detail = ("no state from the robot" if self._engaged
                            else "waiting for the robot's state")
            return

        if not self._engaged and not self._engage(leader, target):
            return

        goal = self._next_goal(target)
        try:
            leader.bus.sync_write("Goal_Position", goal)
        except Exception as exc:
            self._failed = f"could not drive the leader: {exc}"
            logger.exception("[mimic] writing the leader's goal failed")
            self.free(leader)
            return
        self._goal = goal
        self._tick += 1

    # --- internals ----------------------------------------------------------

    def _next_goal(self, target: dict[str, float]) -> dict[str, float]:
        """Where to command the leader this tick: the alignment ramp, then the arm's
        pose, both through one low-pass.

        The ramp is eased rather than linear — a linear one arrives at full alignment
        still moving and then stops dead, which is felt in the hand as a knock. The
        filter is what makes tracking continuous: see SMOOTH_S.
        """
        if self._tick >= self._align_ticks:
            aim = target
        else:
            # Cosine ease: 0 at the start, 1 at the end, zero slope at both ends.
            alpha = 0.5 - 0.5 * math.cos(math.pi * self._tick / self._align_ticks)
            aim = {joint: self._origin[joint] + (target[joint] - self._origin[joint]) * alpha
                   for joint in target}
        previous = self._goal or self._origin
        return {joint: previous[joint] + (aim[joint] - previous[joint]) * self._smooth
                for joint in aim}

    def _targets(self, state: Optional[dict[str, float]]) -> Optional[dict[str, float]]:
        """The follower's six joints as leader goals, or None if the state is partial."""
        if not state:
            return None
        target: dict[str, float] = {}
        for joint in JOINTS:
            value = state.get(f"{joint}.pos")
            if value is None:
                return None
            target[joint] = float(value)
        return target

    def _engage(self, leader, target: dict[str, float]) -> bool:
        """Take the leader: read where it is, torque on, size the ramp. False if the bus
        refused, which parks mimic in `error` rather than retrying every tick."""
        try:
            present = leader.bus.sync_read("Present_Position", list(JOINTS))
            leader.bus.enable_torque()
            if self._torque_limit:
                # After enabling torque, not before: the motor resets its running limit
                # to the configured maximum when torque comes on.
                leader.bus.sync_write("Torque_Limit", self._torque_limit, normalize=False)
        except Exception as exc:
            self._failed = f"could not take the leader: {exc}"
            logger.exception("[mimic] engaging the leader failed")
            # Torque may already be on with no goal behind it, which is a leader gone
            # stiff in the human's hand. Nothing drives the robot from it — mimic engages
            # only while somebody else has the arm — so dropping it here is safe.
            self.free(leader)
            return False

        self._origin = {joint: float(present[joint]) for joint in JOINTS}
        gap = max((abs(target[j] - self._origin[j]) for j in JOINTS if j != GRIPPER),
                  default=0.0)
        align_s = self._align_s * min(1.0, gap / ALIGN_REFERENCE_DEG) if gap else 0.0
        self._align_ticks = max(int(align_s * self._fps), 1)
        self._tick = 0
        self._engaged = True
        self._detail = ""
        logger.info("[mimic] leader engaged; aligning %.1f deg over %.2fs", gap, align_s)
        return True

    def _reset(self) -> None:
        self._origin, self._goal = {}, {}
        self._yielded = False
        self._tick, self._align_ticks = 0, 0
        self._detail = ""
