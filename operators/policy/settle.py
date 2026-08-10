"""Settle gate: wait for the arm to reach the last commanded pose.

MolmoAct2 infers on a single observation, so it should see a stationary arm. The
gate blocks until the observed arm is within ``tolerance`` of the last command
(or ``timeout_s`` elapses), sending nothing meanwhile so the robot keeps driving
toward that command.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Mapping, Optional

from utilities.common import pace

logger = logging.getLogger(__name__)


class SettleGate:
    """Blocks until observed joints reach the last recorded command."""

    def __init__(self, *, keys: tuple[str, ...], tolerance: float, timeout_s: float, fps: int) -> None:
        self._keys = keys
        self._tolerance = tolerance  # max per-joint error to call it "reached"; <=0 disables
        self._timeout_s = timeout_s
        self._fps = fps
        self._last: dict[str, float] | None = None

    @property
    def enabled(self) -> bool:
        return self._tolerance > 0

    def reset(self) -> None:
        """Forget the last command (call at the start of each run)."""
        self._last = None

    def record(self, action: Mapping[str, float]) -> None:
        """Remember the pose we just commanded, restricted to the gated joints."""
        self._last = {key: float(action[key]) for key in self._keys}

    def error(self, state: Mapping[str, float]) -> float:
        """Largest per-joint gap between the observed state and the last command."""
        return max(abs(state[key] - self._last[key]) for key in self._keys)

    def reached(self, state: Mapping[str, float]) -> bool:
        if self._last is None:
            return True
        if not all(key in state for key in self._keys):
            return False
        return self.error(state) <= self._tolerance

    async def wait(self, state_getter: Callable[[], Mapping[str, float]], stop: Optional["object"] = None) -> None:
        """Wait until the arm settles onto the last command (or times out).

        ``state_getter`` returns the latest observed state each poll; ``stop`` is
        an optional event with ``.is_set()`` to bail out early.
        """
        if self._last is None or not self.enabled:
            return
        deadline = time.monotonic() + self._timeout_s
        async for _ in pace(self._fps):
            if stop is not None and stop.is_set():
                return
            state = state_getter()
            if self.reached(state):
                return
            if time.monotonic() > deadline:
                logger.info("[policy] settle timeout (max joint err %.2f)", self.error(state))
                return
