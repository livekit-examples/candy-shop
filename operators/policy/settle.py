"""Settle gate: decide when a fresh observation has caught up to the last
commanded pose, so the policy replans from a state the arm actually reached.

MolmoAct2 emits 30-step chunks: ``select_action`` pops one step per tick and
only runs the model when the chunk drains. The gate belongs at that **replan
boundary** -- the new chunk is conditioned on one observation, and the camera
frame paired with it cannot be extrapolated forward. Mid-chunk pops are not
gated: waiting there would turn `fps`-rate execution into one step per wait.

It is a predicate, not a wait: the caller skips the tick and keeps looping, so
observations keep arriving while the arm catches up. Bounded three ways --

  * ``min_wait_s``  don't accept an observation until the command has had time
    to land, so a pre-command obs that happens to sit near the pose isn't taken;
  * a stillness fallback -- once the arm stops moving (per-joint delta under
    ``still_tolerance`` on two consecutive checks) it has caught up as far as it
    will; the residual is servo steady-state error, and waiting cannot improve
    the observation. Without this, a tolerance tighter than the servo's tracking
    error parks every replan on the timeout;
  * ``timeout_s``  replan anyway if a joint never settles (held load, stalled
    against an obstacle), so the loop cannot stall.

Compares in wire units: ``record`` the action you sent (same space as the
follower's reported ``.pos``), then gate on ``ready(state)``.
"""
from __future__ import annotations

import time
from typing import Mapping

# Per-joint movement under this (wire units, between consecutive checks ~one
# tick apart) counts as stopped; the command needs at least this long to land.
STILL_TOLERANCE = 0.4
MIN_WAIT_S = 0.05


class SettleGate:
    """Gates the replan boundary on the arm reaching the last commanded pose."""

    def __init__(self, *, keys: tuple[str, ...], tolerance: float, timeout_s: float,
                 still_tolerance: float = STILL_TOLERANCE, min_wait_s: float = MIN_WAIT_S) -> None:
        self._keys = keys
        self._tolerance = tolerance  # max per-joint error to call it "reached"; <=0 disables
        self._timeout_s = timeout_s
        self._still_tolerance = still_tolerance
        self._min_wait_s = min_wait_s
        self._last: dict[str, float] | None = None
        self._recorded_at = 0.0
        self._prev_state: dict[str, float] | None = None
        self._still_checks = 0

    @property
    def enabled(self) -> bool:
        return self._tolerance > 0

    @property
    def timeout_s(self) -> float:
        """How long we are willing to wait for the arm to reach a command."""
        return self._timeout_s

    def reset(self) -> None:
        """Forget the last command (call at the start of each run)."""
        self._last = None
        self._recorded_at = 0.0
        self._prev_state = None
        self._still_checks = 0

    def record(self, action: Mapping[str, float]) -> None:
        """Remember the pose we just commanded -- the target to settle to."""
        self._last = {key: float(action[key]) for key in self._keys}
        self._recorded_at = time.monotonic()
        self._prev_state = None
        self._still_checks = 0

    def error(self, state: Mapping[str, float]) -> float | None:
        """Largest per-joint gap between the observed state and the last command."""
        if self._last is None:
            return None
        gaps = [abs(state[key] - value) for key, value in self._last.items() if key in state]
        return max(gaps) if gaps else None

    def _still(self, state: Mapping[str, float]) -> bool:
        """True once the arm has stopped moving. Call at most once per tick."""
        prev, self._prev_state = self._prev_state, {
            key: float(state[key]) for key in self._keys if key in state
        }
        if prev is None or not self._prev_state:
            self._still_checks = 0
            return False
        moved = max(abs(value - prev[key]) for key, value in self._prev_state.items() if key in prev)
        self._still_checks = self._still_checks + 1 if moved <= self._still_tolerance else 0
        return self._still_checks >= 2

    def ready(self, state: Mapping[str, float]) -> bool:
        """True if it is OK to plan now: gate disabled, nothing commanded yet,
        the command landed, the arm stopped moving, or we waited past the timeout."""
        if not self.enabled or self._last is None:
            return True
        waited = time.monotonic() - self._recorded_at
        if waited < self._min_wait_s:
            return False
        if waited >= self._timeout_s:
            return True
        error = self.error(state)
        if error is None or error <= self._tolerance:
            return True
        return self._still(state)
