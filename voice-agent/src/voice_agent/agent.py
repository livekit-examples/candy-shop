import asyncio
import logging
import pathlib
from typing import Literal

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    cli,
    function_tool,
    get_job_context,
    inference,
    room_io,
)
from livekit.agents.llm import ImageContent
from livekit.plugins import ai_coustics

from voice_agent.config import (
    LLM_MODEL,
    ROBOT_IDENTITY,
    RPC_TIMEOUT_S,
    STT_MODEL,
    TTS_MODEL,
    TTS_VOICE,
    VIDEO_TRACK_NAME,
)
from voice_agent.task import GiveCandy

logger = logging.getLogger("agent")

# Single source of config: the repo-root .env (.env.local overrides it).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_REPO_ROOT / ".env.local", override=True)


class CandyShopAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a friendly robot running a candy shop.

                You stock exactly four candies: KitKat, Nerds, Twix, and Snickers.
                That is the whole shelf. If someone asks for anything else — a
                lollipop, chocolate in general, "whatever you have" — say what you
                actually stock and let them pick one. Never call give_candy with a
                candy that isn't on that list; the arm has only been taught those four.

                You pick candy off the shelf and give it to the user on request. If the
                user asks for several (e.g. "a KitKat and a Twix"), give them one at a
                time, confirming each landed before starting the next.

                Tools:
                - give_candy: pick up one candy and drop it in the drop zone.
                - look: check the overhead camera to confirm the candy actually landed in
                  the drop zone. Use it after give_candy; if the candy isn't there, try again.
                - reset: return the robot to its home position if something goes wrong.
                """
        )

        self.latest_frame = None
        self.video_stream = None
        self._frame_task = None

    async def on_enter(self):
        self.room = get_job_context().room

        @self.room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if (
                track.kind == rtc.TrackKind.KIND_VIDEO
                and participant.identity == ROBOT_IDENTITY
                and publication.name == VIDEO_TRACK_NAME
            ):
                self.video_stream = rtc.VideoStream(track)
                # Keep a reference so the task isn't garbage-collected mid-run.
                self._frame_task = asyncio.create_task(self._read_frames())

    @function_tool()
    async def give_candy(
        self, context: RunContext, candy_name: Literal["kitkat", "nerd", "twix", "snicker"]
    ) -> str:
        """Pick up one candy from the shelf and hand it to the user.

        Args:
            candy_name: Which of the four candies on the shelf to fetch. These are
                the only ones the arm was taught, so pick the closest match to what
                the user asked for — or ask them to choose if nothing fits.
        """
        succeeded = await GiveCandy(
            room=self.room,
            candy_name=candy_name,
            chat_ctx=self.chat_ctx,
        )
        if succeeded:
            return f"Handed the {candy_name} to the user."
        # No cause claimed: a customer's "stop" takes this path too.
        return f"The {candy_name} was not delivered."

    @function_tool()
    async def look(self, context: RunContext) -> str:
        """Look at the overhead camera to check the drop zone."""
        if not self.latest_frame:
            return "No camera frame is available yet."

        await self.session.generate_reply(
            user_input=[
                ImageContent(image=self.latest_frame),
            ]
        )
        return "Finished analyzing the camera image."

    @function_tool()
    async def reset(self, context: RunContext) -> None:
        """Reset the robot to its home position after a failure or a finished order."""
        await self.room.local_participant.perform_rpc(
            destination_identity=ROBOT_IDENTITY,
            method="reset_to_zero_position",
            payload="",
            response_timeout=RPC_TIMEOUT_S,
        )

    async def _read_frames(self):
        async for event in self.video_stream:
            self.latest_frame = event.frame


server = AgentServer()


@server.rtc_session(agent_name="candy-shop-assistant")
async def voice_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=inference.STT(model=STT_MODEL, language="multi"),
        llm=inference.LLM(model=LLM_MODEL),
        tts=inference.TTS(
            model=TTS_MODEL,
            voice=TTS_VOICE,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
        ),
        expressive=True
    )

    await session.start(
        room=ctx.room,
        agent=CandyShopAssistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )

    await ctx.connect()


def main() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    main()
