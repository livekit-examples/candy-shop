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

Intervention is a push. With torque on, forcing a joint opens a gap between where the
leader is and where it was told to be, and a gap held past a threshold is a human asking
for the arm — no button, no reaching for the keyboard mid-motion.

Both arms report normalized positions in the same units (degrees per joint, 0-100 on the
gripper, each against its own calibration), which is what makes the follower's state
usable as a leader goal — the same mapping ordinary teleop already relies on, pointed the
other way.

**The one hazard is a torqued leader nobody is holding.** Drop torque and an SO-101
leader falls under gravity; if this teleoperator holds the arm at that moment, the
follower goes down with it. So torque is dropped in exactly two places, both of which
mean a hand is already on the arm: a push (the hand is what caused it) and the toggle
going off. Claiming any other way *holds* the leader instead — torque on, goal frozen —
which freezes the robot at the pose it was in and waits for the human to be ready.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# The six leader joints. `<joint>.pos` on the wire, bare names on the leader's bus.
JOINTS: tuple[str, ...] = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
)
# 0-100 travel, not degrees, so it shares neither the arm's push threshold nor its
# alignment reference.
GRIPPER = "gripper"

# Alignment budget for the worst case: the leader can be parked a long way from wherever
# the arm is, and torque coming on is the one moment it moves on its own. Sized like the
# policy's start ramp — a cap on slew rate, so a leader already near the arm aligns in a
# fraction of it.
ALIGN_REFERENCE_DEG = 60.0


class MimicController:
    """Leader-follows-follower, the push that ends it, and the hold that makes a
    handover safe.

    Owns nothing it did not open: the leader belongs to the recorder's runtime, so every
    entry point tolerates it being absent (setup not finished) or dead (bus error).
    """

    def __init__(
        self,
        *,
        fps: int,
        align_s: float = 1.5,
        intervene_deg: float = 10.0,
        intervene_gripper: float = 20.0,
        intervene_hold_s: float = 0.2,
        torque_limit: int = 0,
    ) -> None:
        self._fps = fps
        self._align_s = max(align_s, 0.0)
        self._intervene_deg = max(intervene_deg, 0.0)
        self._intervene_gripper = max(intervene_gripper, 0.0)
        self._intervene_ticks = max(int(intervene_hold_s * fps), 1)
        # 0..1000 (per mille of the motor's rated torque); 0 leaves the motor's own limit
        # alone. A lower limit is what keeps the leader yielding to a hand instead of
        # fighting it, at the cost of tracking the arm less crisply.
        self._torque_limit = max(min(int(torque_limit), 1000), 0)

        self.enabled = False
        self._engaged = False      # torque is on
        self._holding = False      # engaged, but frozen: a handover is in progress
        self._origin: dict[str, float] = {}   # leader pose when torque came on
        self._align_ticks = 0
        self._tick = 0             # ticks written since engaging
        self._goal: dict[str, float] = {}     # what we last commanded
        self._over = 0             # consecutive ticks past the push threshold
        self._error_deg = 0.0
        self._detail = ""
        self._failed = ""

    # --- view ---------------------------------------------------------------

    @property
    def engaged(self) -> bool:
        """Torque is on and the leader is ours to command."""
        return self._engaged

    @property
    def holding(self) -> bool:
        """Frozen mid-handover: the leader is held at a pose, waiting for a hand."""
        return self._holding

    def snapshot(self) -> dict:
        if self._failed:
            state, detail = "error", self._failed
        elif not self.enabled:
            state, detail = "off", ""
        elif self._holding:
            state, detail = "holding", "leader held at the handover pose — hold it, then " \
                                       "switch mimic off to fly"
        elif not self._engaged:
            state, detail = "waiting", self._detail
        elif self._tick < self._align_ticks:
            state, detail = "aligning", "moving the leader onto the arm's pose"
        else:
            state, detail = "tracking", "push the leader to take the arm"
        return {
            "enabled": self.enabled,
            "state": state,
            "detail": detail,
            "error_deg": round(self._error_deg, 1),
            "intervene_deg": self._intervene_deg,
        }

    # --- control ------------------------------------------------------------

    def set_enabled(self, enabled: bool, leader) -> None:
        """The toggle. Switching it off frees the leader — the human asked for it, so
        this is one of the two places torque is allowed to drop."""
        self.enabled = bool(enabled)
        self._failed = ""
        if not self.enabled:
            self.free(leader)

    def handover(self, leader, *, free: bool) -> None:
        """Give the arm to the human.

        `free` drops leader torque, and is only correct when a hand is already on it (a
        push). Otherwise the leader is *held*: torque stays on and the goal stops
        following the robot, which freezes the arm at this pose instead of letting a
        falling leader drive it.
        """
        if free:
            self.free(leader)
        else:
            self.hold()

    def hold(self) -> None:
        """Freeze the goal, keep the torque. No-op when nothing is engaged."""
        if not self._engaged or self._holding:
            return
        self._holding = True
        self._over = 0
        logger.info("[mimic] leader held at the handover pose")

    def free(self, leader) -> None:
        """Drop torque and stop commanding. The leader is loose the moment this returns."""
        was = self._engaged
        self._engaged = self._holding = False
        self._reset()
        if leader is None or not was:
            return
        try:
            leader.bus.disable_torque()
        except Exception as exc:
            self._failed = f"could not release the leader: {exc}"
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
            self.hold()
            self._detail = "this teleoperator is driving; mimic waits for a handover"
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

        alpha = 1.0 if self._tick >= self._align_ticks else self._tick / self._align_ticks
        goal = {joint: self._origin[joint] + (target[joint] - self._origin[joint]) * alpha
                for joint in target}
        try:
            leader.bus.sync_write("Goal_Position", goal)
        except Exception as exc:
            self._failed = f"could not drive the leader: {exc}"
            logger.exception("[mimic] writing the leader's goal failed")
            self.free(leader)
            return
        self._goal = goal
        self._tick += 1

    def check_push(self, leader_action: dict[str, float],
                   on_intervene: Callable[[str], None]) -> None:
        """Compare where the leader *is* against what it was told, and call
        `on_intervene` once the gap has held long enough to be a hand and not a wobble.

        Fed the action the tick loop already read, so this costs no extra bus traffic.
        Only while tracking: during the ramp the gap *is* the ramp.
        """
        if (not self._engaged or self._holding or self._tick < self._align_ticks
                or not self._intervene_deg):
            self._over = 0
            return
        worst, worst_joint, worst_gap = 0.0, "", 0.0
        for joint, goal in self._goal.items():
            present = leader_action.get(f"{joint}.pos")
            if present is None:
                continue
            gap = abs(present - goal)
            # The gripper's 0-100 travel is scaled onto the arm's degrees so one
            # threshold ranks every joint; the gap itself is what gets reported.
            ranked = (gap * self._intervene_deg / self._intervene_gripper
                      if joint == GRIPPER and self._intervene_gripper else gap)
            if ranked > worst:
                worst, worst_joint, worst_gap = ranked, joint, gap
        self._error_deg = worst_gap
        if worst < self._intervene_deg:
            self._over = 0
            return
        self._over += 1
        if self._over < self._intervene_ticks:
            return
        self._over = 0
        logger.info("[mimic] push on %s (%.1f past %.1f) — taking the arm",
                    worst_joint, worst, self._intervene_deg)
        on_intervene(worst_joint)

    # --- internals ----------------------------------------------------------

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
            return False

        self._origin = {joint: float(present[joint]) for joint in JOINTS}
        gap = max((abs(target[j] - self._origin[j]) for j in JOINTS if j != GRIPPER),
                  default=0.0)
        align_s = self._align_s * min(1.0, gap / ALIGN_REFERENCE_DEG) if gap else 0.0
        self._align_ticks = max(int(align_s * self._fps), 1)
        self._tick = 0
        self._over = 0
        self._engaged = True
        self._detail = ""
        logger.info("[mimic] leader engaged; aligning %.1f deg over %.2fs", gap, align_s)
        return True

    def _reset(self) -> None:
        self._origin, self._goal = {}, {}
        self._tick, self._align_ticks, self._over = 0, 0, 0
        self._error_deg = 0.0
        self._detail = ""
