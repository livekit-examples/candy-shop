"""The room's operator roster: who else drives the arm, and the RPC that drives them.

Every operator here owns its identity string; each one imports it from this module so a
rename can't drift between the peer that answers an RPC and the peers that call it.

The teleoperator *discovers* peers at runtime (Portal's operator list), but discovery
alone doesn't say what an operator accepts — these peers advertise no descriptor of
their own — so this table carries the form: which RPC starts work, which stops it, and
what the payload looks like. It mirrors the `operators` block of `web/demos.json` in
livekit-actuate, which drives the same RPCs from the browser: renaming a method here
is a rename there too.

Pure stdlib on purpose. The review UI imports this and must not pull `livekit.portal`
into its process (see `teleoperator.common.contract_camera_names`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

ROBOT = "robot"
MOVE_TO = "move-to-operator"
POLICY = "policy-operator"
REWARD = "reward-operator"

# Robot-side RPC that folds the arm and stops the slider. It self-claims the robot as
# active operator, so it is the last step of a stop, never the first.
ZERO_RPC = "reset_to_zero_position"


@dataclass(frozen=True)
class Field:
    """The single argument an operator's run RPC takes.

    Both Python parsers accept a bare value — a number for `move_to`, a sentence
    otherwise — so the payload is the value itself, not a JSON envelope.
    """

    label: str
    kind: str                       # "text" | "number"
    default: str = ""
    minimum: float = 0.0            # `number` only
    maximum: float = 100.0
    step: float = 1.0
    # (label, value) shortcuts. For prompts these are the instruction strings the
    # policy was fine-tuned on, verbatim from the dataset's `meta/tasks.parquet`: a
    # paraphrase is off-distribution and picks worse. The voice agent keeps its own
    # copy in `voice-agent/src/voice_agent/config.py` — change both together.
    presets: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class OperatorSpec:
    identity: str
    title: str
    kind: str
    run_rpc: str
    stop_rpc: str
    summary: str
    argument: Optional[Field] = None
    # Operators this one commands. Stopping it cascades, and restarting it restarts
    # them, so the teleoperator must not also drive them itself.
    drives: tuple[str, ...] = ()
    # How long to hold the run RPC open. These are not latencies but the length of the
    # work the call covers: an operator that returns only when preempted needs a wait
    # longer than any session, or the call is reported failed while the arm still moves.
    run_timeout_s: float = 120.0


PROMPT_PRESETS: tuple[tuple[str, str], ...] = (
    ("kitkat", "pick up a kitkat"),
    ("nerd", "pick up a nerd"),
    ("twix", "pick up a twix"),
    ("snicker", "pick up a snicker"),
    ("drop", "drop candy into the black circle"),
)

# Ordered orchestrators-first, which is the order a stop has to travel in: stopping the
# policy while the reward operator still holds its retry loop just starts attempt two.
SPECS: tuple[OperatorSpec, ...] = (
    OperatorSpec(
        identity=REWARD,
        title="Reward",
        kind="orchestrator",
        run_rpc="run_task",
        stop_rpc="stop",
        summary="Drives the policy and watches SARM for the task to finish, retrying on "
                "escalating budgets.",
        argument=Field(label="Prompt", kind="text", default="pick up the candy",
                       presets=PROMPT_PRESETS),
        drives=(POLICY,),
        # The reward operator caps a whole task at 90 s and pays one policy unwind on
        # the way out; matches the voice agent's RUN_TASK_TIMEOUT_S.
        run_timeout_s=120.0,
    ),
    OperatorSpec(
        identity=POLICY,
        title="Policy",
        kind="policy",
        run_rpc="run_policy",
        stop_rpc="stop",
        summary="Serves the DiT checkpoint. Runs until it is stopped — nothing in it "
                "decides the pick is done.",
        argument=Field(label="Prompt", kind="text", default="pick up the candy",
                       presets=PROMPT_PRESETS),
        # `run_policy` returns only when preempted, so this bounds a whole picking
        # session rather than one call. Generous: on expiry we lose track of a policy
        # that is still driving the arm.
        run_timeout_s=900.0,
    ),
    OperatorSpec(
        identity=MOVE_TO,
        title="Move-to",
        kind="positioner",
        run_rpc="move_to",
        stop_rpc="stop",
        summary="Slides the arm along the rail to a position, closed-loop from the "
                "overhead marker.",
        argument=Field(label="Slider position", kind="number", default="20",
                       minimum=0.0, maximum=100.0, step=1.0,
                       presets=(("shelf", "20"), ("drop zone", "70"))),
        run_timeout_s=60.0,
    ),
)

BY_IDENTITY: dict[str, OperatorSpec] = {spec.identity: spec for spec in SPECS}

# Every identity this table declares, in stop order.
STOP_ORDER: tuple[str, ...] = tuple(spec.identity for spec in SPECS)

# Reading order, which is the pipeline's: position the arm, pick, watch the pick. The
# reverse of the order a stop travels in.
DISPLAY_ORDER: tuple[str, ...] = STOP_ORDER[::-1]


def title_for(identity: str) -> str:
    """`"move-to-operator"` -> `"Move-to"`, for peers the table doesn't declare."""
    if spec := BY_IDENTITY.get(identity):
        return spec.title
    base = identity.removesuffix("-operator").replace("-", " ").replace("_", " ").strip()
    return base[:1].upper() + base[1:] if base else identity
