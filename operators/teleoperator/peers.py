"""The rest of the room, from the teleoperator's seat: discover the other operators,
drive them, preempt them, and hand the arm back.

Discovery is presence (Portal's operator list); the descriptors come from
``shared.operators``, because these peers advertise none of their own. A peer that is
live but undeclared still gets an entry — presence with no controls — so plugging in a
second teleoperator is visible rather than silent.

Two rules shape the class:

* **A stop travels orchestrators-first.** Preempting the policy while the reward
  operator still holds its retry loop just starts attempt two. ``STOP_ORDER`` is that
  order, and ``run`` refuses to drive an operator that another one is already driving.
* **A claim outlives the RPCs it preempted.** Every operator's run loop clears the
  robot's active-operator pointer on its way out (``set_active_operator(None)`` in a
  ``finally``), so an unwind that lands *after* we claim would silently drop the arm.
  The claim is therefore an intent this class re-asserts, not a one-shot write.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from shared import operators as roster

logger = logging.getLogger(__name__)

# A stop is the one call that must not be given up on: it crosses a relay to an
# operator that is mid-motion, and abandoning it leaves the arm moving with nobody
# watching the reply. Short enough that a dead peer doesn't hold the panic path.
STOP_TIMEOUT_S = 15.0
# The reset that folds the arm — the robot answers it immediately (it only swaps a
# target), so a long wait would only ever be waiting on a dead robot.
ZERO_TIMEOUT_S = 15.0
# How often the claim is re-asserted while the pointer sits somewhere else. Slower than
# the tick on purpose: `set_active_operator` is a network write, and one every frame
# would be a write storm during the second an operator takes to unwind.
REASSERT_INTERVAL_S = 0.4

# How long a resumed run keeps re-offering itself, and the pause between tries.
#
# A stopped operator is not free the instant it answers our stop: the reward operator
# stays busy for the whole unwind it owes the policy (its own `--policy-timeout`, 10 s by
# default), and refuses a fresh `run_task` with 1409 "already running a task" until that
# finishes. Resume was reliably landing inside that window.
RESUME_GRACE_S = 25.0
RETRY_PAUSE_S = 1.0
# A refusal comes back at once; a run that worked for a while and *then* failed is a real
# outcome, not a race, and re-issuing it would start the work twice.
REFUSAL_WINDOW_S = 5.0


@dataclass
class PeerState:
    """What we know about one peer operator. Everything here is our own view: runs the
    voice agent started are point-to-point RPCs we never see, so `running` means *we*
    started it, and `active` is the only evidence of anyone else's work."""

    identity: str
    running: bool = False
    payload: str = ""
    error: str = ""
    result: str = ""
    started_at: float = 0.0
    # Cleared by a stop, so a resume that is still re-offering its run doesn't hand the
    # operator work the human just cancelled.
    retry_allowed: bool = True
    task: Optional[asyncio.Task] = field(default=None, repr=False)


class PeerControl:
    """Drive the room's other operators, and hold the arm against them."""

    def __init__(self, op: Any) -> None:
        self._op = op
        self._states: dict[str, PeerState] = {}
        # Runs a claim preempted, in the order they were preempted, with the payload to
        # restart them on. Not a set: `resume` replays it.
        self._suspended: list[tuple[str, str]] = []
        self._claiming = False
        self._last_reassert = 0.0
        self._reasserts = 0

    # --- view ---------------------------------------------------------------

    @property
    def claiming(self) -> bool:
        """We hold the arm and keep re-asserting it."""
        return self._claiming

    @property
    def suspended(self) -> list[str]:
        return [identity for identity, _ in self._suspended]

    def online(self) -> list[str]:
        """Peer operators in the room, this teleoperator excluded (Portal's own list
        already excludes it)."""
        return list(self._op.operators())

    def snapshot(self) -> list[dict]:
        """One entry per operator worth showing: everything declared, plus whatever
        undeclared peer turned up. Declared-but-absent rows stay, so an operator that
        was never started reads as `offline` rather than vanishing."""
        online = set(self.online())
        active = self._op.active_operator()
        now = time.monotonic()

        rows: list[dict] = []
        for identity in (*roster.DISPLAY_ORDER,
                         *sorted(o for o in online if o not in roster.BY_IDENTITY)):
            state = self._states.get(identity)
            rows.append({
                "identity": identity,
                "online": identity in online,
                "declared": identity in roster.BY_IDENTITY,
                "active": identity == active,
                "running": bool(state and state.running),
                "payload": state.payload if state else "",
                "error": state.error if state else "",
                "result": state.result if state else "",
                "elapsed_s": (round(now - state.started_at, 1)
                              if state and state.running and state.started_at else 0.0),
            })
        return rows

    # --- driving ------------------------------------------------------------

    def run(self, identity: str, payload: str, *, grace_s: float = 0.0) -> Optional[str]:
        """Fire an operator's run RPC in the background. Returns a refusal, or None.

        Scheduled rather than awaited: `run_policy` returns only when something
        preempts it, so an RPC handler that waited for it would never answer.

        `grace_s` keeps re-offering the run while the operator refuses it — see
        `RESUME_GRACE_S`. Zero for a run a human just asked for: they are watching, and a
        refusal is better news than a spinner.
        """
        spec = roster.BY_IDENTITY.get(identity)
        if spec is None:
            return f"{identity} is not an operator this teleoperator knows how to drive"
        if identity not in self.online():
            return f"{identity} is not in the room"
        state = self._states.setdefault(identity, PeerState(identity))
        if state.running:
            return f"{spec.title} is already running — stop it first"
        for other in roster.SPECS:
            if identity in other.drives and (held := self._states.get(other.identity)):
                if held.running:
                    return (f"{other.title} is driving {spec.title}; stop {other.title} "
                            f"instead of racing it")
        # Taking the arm back from us: an operator claims the pointer itself, and our
        # own claim would keep stealing it back.
        self._claiming = False
        state.running, state.payload = True, payload
        state.error, state.result = "", ""
        state.retry_allowed = True
        state.started_at = time.monotonic()
        state.task = asyncio.create_task(
            self._run(spec, payload, grace_s), name=f"peer-run-{identity}")
        return None

    async def _run(self, spec: roster.OperatorSpec, payload: str, grace_s: float) -> None:
        state = self._states[spec.identity]
        logger.info("[teleoperator] %s(%r) -> %s", spec.run_rpc, payload, spec.identity)
        deadline = time.monotonic() + grace_s
        try:
            while True:
                attempted = time.monotonic()
                try:
                    raw = await self._op.perform_rpc(
                        spec.run_rpc, payload, destination=spec.identity,
                        response_timeout_ms=int(spec.run_timeout_s * 1000),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    refused = time.monotonic() - attempted < REFUSAL_WINDOW_S
                    if refused and state.retry_allowed and time.monotonic() < deadline:
                        # Still unwinding the run we preempted; offer it again shortly.
                        state.error = f"{spec.run_rpc}: {exc} — retrying"
                        logger.info("[teleoperator] %s on %s refused (%s); retrying",
                                    spec.run_rpc, spec.identity, exc)
                        await asyncio.sleep(RETRY_PAUSE_S)
                        continue
                    state.error = f"{spec.run_rpc}: {exc}"
                    logger.warning("[teleoperator] %s on %s failed: %s",
                                   spec.run_rpc, spec.identity, exc)
                else:
                    state.error = ""
                    state.result = _summarize(raw)
                    logger.info("[teleoperator] %s on %s returned %s",
                                spec.run_rpc, spec.identity, state.result)
                return
        finally:
            state.running = False
            state.task = None

    async def stop(self, identity: str) -> Optional[str]:
        """Preempt one operator. Returns a refusal/failure, or None."""
        spec = roster.BY_IDENTITY.get(identity)
        if spec is None:
            return f"{identity} declares no stop RPC"
        if state := self._states.get(identity):
            state.retry_allowed = False
        try:
            await self._op.perform_rpc(spec.stop_rpc, "", destination=identity,
                                       response_timeout_ms=int(STOP_TIMEOUT_S * 1000))
        except Exception as exc:
            # Marked on the card as well as logged: an operator that would not stop is
            # the case worth seeing.
            state = self._states.setdefault(identity, PeerState(identity))
            state.error = f"{spec.stop_rpc}: {exc}"
            logger.warning("[teleoperator] %s on %s failed: %s", spec.stop_rpc, identity, exc)
            return state.error
        logger.info("[teleoperator] stopped %s", identity)
        return None

    async def stop_all(self, *, remember: bool) -> list[str]:
        """Preempt every live declared operator, orchestrators first.

        `remember` records what *we* had running so `resume` can restart it. An
        orchestrator swallows the operators it drives: restarting the reward operator
        restarts the policy, and doing both would leave two things planning at once.
        """
        stopped: list[str] = []
        suspended: list[tuple[str, str]] = []
        online = set(self.online())
        active = self._op.active_operator()
        for identity in roster.STOP_ORDER:
            if identity not in online:
                continue
            state = self._states.get(identity)
            # `running` is our own view, and it drops the moment the run RPC gives up —
            # which for a policy is a wait long enough to expire under the human rather
            # than the work ending. Holding the robot's pointer says it is still going,
            # so a run of ours whose reply we lost is still worth remembering. Only ours:
            # a payload is the proof we started it, and resuming someone else's task on a
            # blank prompt would pick the wrong candy.
            if remember and state is not None and (
                    state.running or (identity == active and state.payload)):
                suspended.append((identity, state.payload))
            if await self.stop(identity) is None:
                stopped.append(identity)
        for identity in roster.STOP_ORDER:  # cancel our own pending waits
            if (state := self._states.get(identity)) and state.task is not None:
                state.task.cancel()
        if remember and suspended:
            # Only overwrite when this stop actually interrupted something: claiming a
            # second time (nothing of ours running by then) must not wipe what the
            # first claim recorded, or Resume quietly becomes Release.
            driven = {d for identity, _ in suspended
                      for d in roster.BY_IDENTITY[identity].drives}
            self._suspended = [entry for entry in suspended if entry[0] not in driven]
        return stopped

    async def fold_arm(self) -> Optional[str]:
        """The robot's own reset: park every joint and stop the slider.

        Last, never first — the reset self-claims the robot, so an operator whose loop
        is still alive would take the pointer straight back and carry on.
        """
        robot = self._op.robot_identity()
        if robot is None:
            return "the robot is not in the room"
        try:
            await self._op.perform_rpc(roster.ZERO_RPC, "", destination=robot,
                                       response_timeout_ms=int(ZERO_TIMEOUT_S * 1000))
        except Exception as exc:
            logger.warning("[teleoperator] %s on %s failed: %s", roster.ZERO_RPC, robot, exc)
            return f"{roster.ZERO_RPC}: {exc}"
        return None

    # --- the arm ------------------------------------------------------------

    async def claim(self, me: Optional[str]) -> list[str]:
        """Take the arm: preempt everyone, then hold the pointer. Returns what was
        preempted and will be restarted by `resume`."""
        await self.stop_all(remember=True)
        self._claiming = True
        self._reasserts = 0
        self._last_reassert = time.monotonic()
        await self._op.set_active_operator(me)
        logger.info("[teleoperator] arm claimed%s",
                    f"; suspended {', '.join(self.suspended)}" if self._suspended else "")
        return self.suspended

    async def release(self) -> None:
        """Drop the pointer and the claim. Whatever was suspended stays suspended —
        `resume` is the call that restarts it."""
        self._claiming = False
        await self._op.set_active_operator(None)

    async def resume(self) -> list[str]:
        """Hand the arm back to whatever the claim interrupted."""
        pending, self._suspended = self._suspended, []
        # Only let go of a pointer that is ours: resuming with nothing suspended must not
        # yank the arm out from under an operator that took it in the meantime.
        #
        # Decide before clearing the claim, and clear it before the await: `reassert` runs
        # every tick, and with the flag still up it re-takes the pointer the moment this
        # clears it — which is exactly what it saw and did.
        mine = self._claiming or self._op.active_operator() == self._op.local_identity()
        self._claiming = False
        if mine:
            await self._op.set_active_operator(None)
        restarted: list[str] = []
        for identity, payload in pending:
            # Each operator claims the pointer itself on the way in, which is why this
            # only has to re-issue the run — with a grace window, because the operator we
            # preempted may still be unwinding and would refuse it outright.
            if self.run(identity, payload, grace_s=RESUME_GRACE_S) is None:
                restarted.append(identity)
        logger.info("[teleoperator] resumed %s", ", ".join(restarted) or "nothing")
        return restarted

    def reassert(self, me: Optional[str]) -> Optional[asyncio.Task]:
        """Re-take the pointer if something else grabbed it while we hold the arm.

        Called from the tick loop. Returns the write's task, or None when there is
        nothing to do — a preempted operator's `finally` clears the pointer up to a
        second after we claimed, and every operator self-claims on the way in, so
        without this the arm goes quiet (or worse, obeys someone else) mid-teleop.
        """
        if not self._claiming or me is None:
            return None
        if self._op.active_operator() == me:
            return None
        now = time.monotonic()
        if now - self._last_reassert < REASSERT_INTERVAL_S:
            return None
        self._last_reassert = now
        self._reasserts += 1
        # Loud only the first few times: a contested pointer (an agent starting a task
        # while a human holds the arm) would otherwise flood the log.
        log = logger.info if self._reasserts <= 3 else logger.debug
        log("[teleoperator] pointer drifted to %s; re-claiming",
            self._op.active_operator())
        return asyncio.create_task(self._op.set_active_operator(me), name="reclaim")


def _summarize(raw: str) -> str:
    """A run's JSON reply as one line. The operators answer with a dict of outcome
    fields (`reason`, `done`, `progress`, `elapsed_s`); anything else passes through."""
    try:
        reply = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return (raw or "").strip()[:120]
    if not isinstance(reply, dict):
        return str(reply)[:120]
    interesting = ("reason", "done", "progress", "position", "attempt_count", "elapsed_s")
    parts = [f"{key}={_short(reply[key])}" for key in interesting if key in reply]
    return "  ".join(parts) if parts else json.dumps(reply)[:120]


def _short(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, float) else str(value)
