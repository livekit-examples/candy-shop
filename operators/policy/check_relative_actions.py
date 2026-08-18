"""Verify the relative/absolute action pair recovers the pose it started from.

Run before trusting a relative-action checkpoint on the arm::

    uv run python -m operators.policy.check_relative_actions

The pair is stateful and order-sensitive, which is how it went wrong on pi0.5:
``RelativeActionsProcessorStep`` caches the state it subtracted -- on *every* call,
including ones carrying no action -- and ``AbsoluteActionsProcessorStep`` adds back
whatever is cached at the moment it runs. So the chunk must be converted back while
the cache still holds the pose it was planned from.

``run.py._infer`` satisfies that: one ``_pre`` per chunk, then ``_post`` on every action
before returning, so the whole chunk converts against one pose. Deferring the conversion
-- draining a queue of *relative* actions across later ticks, each of which re-runs
``_pre`` -- silently reintroduces the bug, and the symptom is subtle: the arm moves
plausibly, just wrong by however far it drifted since the chunk was planned.

Case B below is that mistake, kept executable so the failure is visible rather than
remembered.
"""
from __future__ import annotations

import torch

from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
)
from lerobot.utils.constants import OBS_STATE

from operators.policy.train import RELATIVE_EXCLUDE
from shared.rest_pose import ALL_ACTION_KEYS

# Far enough to be unmistakable in the output, and to make "error == drift" legible.
DRIFT = 7.5
HORIZON = 32


def _pair() -> tuple[RelativeActionsProcessorStep, AbsoluteActionsProcessorStep]:
    relative = RelativeActionsProcessorStep(
        enabled=True, exclude_joints=list(RELATIVE_EXCLUDE), action_names=list(ALL_ACTION_KEYS)
    )
    return relative, AbsoluteActionsProcessorStep(enabled=True, relative_step=relative)


def worst_error(state: torch.Tensor, actions: torch.Tensor, *, defer: bool) -> float:
    relative, absolute = _pair()
    deltas = relative(
        {TransitionKey.OBSERVATION: {OBS_STATE: state}, TransitionKey.ACTION: actions}
    )[TransitionKey.ACTION]

    if defer:
        # A later tick re-runs the preprocessor before the queue drains, which is all it
        # takes: the cache now holds a pose the chunk was never planned against.
        relative({TransitionKey.OBSERVATION: {OBS_STATE: state + DRIFT}, TransitionKey.ACTION: None})

    worst = 0.0
    for step in range(actions.shape[1]):
        recovered = absolute({TransitionKey.ACTION: deltas[:, step, :]})[TransitionKey.ACTION]
        worst = max(worst, (recovered - actions[:, step, :]).abs().max().item())
    return worst


def main() -> None:
    torch.manual_seed(0)
    dim = len(ALL_ACTION_KEYS)
    state = torch.randn(1, dim) * 20
    actions = torch.randn(1, HORIZON, dim) * 15 + state.unsqueeze(1)

    at_generation = worst_error(state, actions, defer=False)
    deferred = worst_error(state, actions, defer=True)

    print(f"convert at generation (run.py._infer) : {at_generation:.6f}")
    print(f"convert after a later _pre           : {deferred:.6f}  (state drift was {DRIFT})")

    relative, _ = _pair()
    deltas = relative(
        {TransitionKey.OBSERVATION: {OBS_STATE: state}, TransitionKey.ACTION: actions}
    )[TransitionKey.ACTION]
    for name in RELATIVE_EXCLUDE:
        index = list(ALL_ACTION_KEYS).index(name)
        kept = torch.allclose(deltas[0, :, index], actions[0, :, index])
        print(f"{name:<16} kept absolute       : {kept}")
        assert kept, f"{name} should not have been converted to a delta"

    assert at_generation < 1e-4, f"serving's own call pattern does not round-trip ({at_generation})"
    assert abs(deferred - DRIFT) < 1e-4, (
        "deferred conversion no longer fails by exactly the drift; the caching contract "
        "in lerobot.processor changed and run.py's assumptions need rechecking"
    )
    print("\nOK: the chunk round-trips when converted at generation, and only then.")


if __name__ == "__main__":
    main()
