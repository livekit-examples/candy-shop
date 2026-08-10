"""Robot control task: pick a candy off the shelf and hand it to the user."""

import logging

from livekit import rtc
from livekit.agents import AgentTask

from voice_agent.config import (
    MOVE_TO_IDENTITY,
    POSITIONS,
    RPC_TIMEOUT_S,
    RUN_POLICY_IDENTITY,
)

logger = logging.getLogger("agent")


class GiveCandy(AgentTask[bool]):
    """Picks one candy off the shelf and drops it in the drop zone.

    Runs as a self-contained task: on entry it takes over the session, executes the
    physical routine over RPC, and completes with ``True`` on success or ``False`` if
    any step fails. Awaiting the task yields that result.
    """

    def __init__(self, room: rtc.Room, candy_name: str, chat_ctx=None) -> None:
        self.room = room
        self.candy_name = candy_name

        super().__init__(
            instructions=f"You are picking up a {candy_name} and handing it to the user.",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        # Narrate each step so the user hears progress during the multi-second
        # routine. say() isn't awaited, so speech plays alongside the RPCs.
        try:
            self.session.say("Heading over to the candy shelf.")
            await self._move_to("candy shelf")

            self.session.say(f"Picking up your {self.candy_name}.")
            await self._run_policy(f"Pick up {self.candy_name}")

            self.session.say("Bringing it over to you.")
            await self._move_to("drop zone")

            self.session.say("And here you go!")
            await self._run_policy(f"Drop {self.candy_name}")
        except Exception:
            logger.exception("failed to give candy: %s", self.candy_name)
            self.complete(False)
            return

        self.complete(True)

    async def _move_to(self, position: str) -> None:
        """Drive the robot to a named waypoint."""
        if position not in POSITIONS:
            raise ValueError(
                f"Unknown position {position!r}; expected one of {list(POSITIONS)}."
            )

        await self.room.local_participant.perform_rpc(
            destination_identity=MOVE_TO_IDENTITY,
            method="move_to",
            payload=str(POSITIONS[position]),
            response_timeout=RPC_TIMEOUT_S,
        )

    async def _run_policy(self, task: str) -> None:
        """Ask the manipulation policy to execute a pick or drop."""
        await self.room.local_participant.perform_rpc(
            destination_identity=RUN_POLICY_IDENTITY,
            method="run_policy",
            payload=task,
            response_timeout=RPC_TIMEOUT_S,
        )
