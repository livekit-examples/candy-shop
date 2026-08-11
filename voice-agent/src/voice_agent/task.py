"""Robot control task: pick a candy off the shelf and hand it to the user."""

import asyncio
import contextlib
import logging

from livekit import rtc
from livekit.agents import AgentTask, RunContext, function_tool

from voice_agent.config import (
    MOVE_TO_IDENTITY,
    POLICY_IDENTITY,
    POSITIONS,
    REWARD_IDENTITY,
    ROBOT_IDENTITY,
    RPC_TIMEOUT_S,
    RUN_TASK_TIMEOUT_S,
)

logger = logging.getLogger("agent")


class GiveCandy(AgentTask[bool]):
    """Picks one candy off the shelf and drops it in the drop zone. Awaiting yields success/failure.

    The chain runs as a background task, not inline in `on_enter`, so that `stop`
    has something it can cancel.
    """

    def __init__(self, room: rtc.Room, candy_name: str, chat_ctx=None) -> None:
        self.room = room
        self.candy_name = candy_name
        self._chain: asyncio.Task | None = None

        super().__init__(
            instructions=f"""
                You are picking up a {candy_name} and handing it to the user.

                If the user asks you to stop, wait, or cancel — or says anything that
                means they want the robot to quit what it's doing — call `stop`
                immediately. Don't finish the current step first, and don't ask them
                to confirm.
                """,
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        self._chain = asyncio.create_task(self._run(), name=f"give-candy:{self.candy_name}")

    async def on_exit(self) -> None:
        # However the task ended, never leave the chain driving the arm.
        await self._abandon()

    @function_tool()
    async def stop(self, context: RunContext) -> str:
        """Stop the robot now and give up on the candy currently being fetched."""
        logger.info("stop requested mid-errand: %s", self.candy_name)
        await self._abandon()
        if not self.done():
            self.complete(False)
        return f"Stopped. The {self.candy_name} was not delivered."

    async def _run(self) -> None:
        """The errand itself. Completes the task; never raises to the event loop."""
        try:
            # say() isn't awaited, so speech plays alongside the RPCs.
            self.session.say("Heading over to the candy shelf.")
            await self._move_to("candy shelf")

            self.session.say(f"Picking up your {self.candy_name}.")
            await self._run_task(f"Pick up {self.candy_name}")

            self.session.say("Bringing it over to you.")
            await self._move_to("drop zone")

            self.session.say("And here you go!")
            await self._run_task(f"Drop {self.candy_name}")
        except asyncio.CancelledError:
            # Teardown is the canceller's job — it can await cleanly.
            raise
        except Exception:
            logger.exception("failed to give candy: %s", self.candy_name)
            await self._teardown()
            if not self.done():
                self.complete(False)
            return

        if not self.done():
            self.complete(True)

    async def _abandon(self) -> None:
        """Cancel the chain, then park the robot. Safe to call more than once."""
        task, self._chain = self._chain, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await self._teardown()

    async def _teardown(self) -> None:
        """Stop the operators, then park the arm.

        Order matters: `reset_to_zero_position` self-claims the robot, but an operator
        whose run loop is still alive would just re-claim and carry on.
        """
        await asyncio.gather(
            self._rpc(REWARD_IDENTITY, "stop"),
            self._rpc(POLICY_IDENTITY, "stop"),
            self._rpc(MOVE_TO_IDENTITY, "stop"),
            return_exceptions=True,
        )
        await self._rpc(ROBOT_IDENTITY, "reset_to_zero_position")

    async def _rpc(self, identity: str, method: str, payload: str = "") -> None:
        """Best-effort RPC: teardown must not fail because one peer is gone."""
        try:
            await self.room.local_participant.perform_rpc(
                destination_identity=identity,
                method=method,
                payload=payload,
                response_timeout=RPC_TIMEOUT_S,
            )
        except Exception as exc:
            logger.warning("teardown %s.%s failed: %s", identity, method, exc)

    async def _move_to(self, position: str) -> None:
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

    async def _run_task(self, task: str) -> None:
        """Run one manipulation task: the reward operator drives the policy and
        watches SARM for completion, returning once the task is done."""
        await self.room.local_participant.perform_rpc(
            destination_identity=REWARD_IDENTITY,
            method="run_task",
            payload=task,
            response_timeout=RUN_TASK_TIMEOUT_S,
        )
