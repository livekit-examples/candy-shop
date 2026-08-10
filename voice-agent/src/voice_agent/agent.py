import asyncio
import logging

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
    MOVE_TO_IDENTITY,
    PARTICIPANT_IDENTITY,
    RPC_TIMEOUT_S,
    STT_MODEL,
    TTS_MODEL,
    TTS_VOICE,
    VIDEO_TRACK_NAME,
)
from voice_agent.task import GiveCandy

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class CandyShopAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a friendly robot running a candy shop.

                You pick candy off a shelf and give it to the user on request. If the user
                asks for several candies (e.g. "I want a KitKat and a lollipop"), give them
                one at a time.

                Tools:
                - give_candy: pick up one candy and drop it in the drop zone.
                - look: check the overhead camera to confirm the candy actually landed in
                  the drop zone. Use it after give_candy; if the candy isn't there, try again.
                - reset: return the robot to its home position if something goes wrong.
                """
        )

        # For vision
        self.latest_frame = None
        self.video_stream = None
        self._frame_task = None

    async def on_enter(self):

        # Set up the video stream and the look tool
        self.room = get_job_context().room

        @self.room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if (
                track.kind == rtc.TrackKind.KIND_VIDEO
                and participant.identity == PARTICIPANT_IDENTITY
                and publication.name == VIDEO_TRACK_NAME
            ):
                self.video_stream = rtc.VideoStream(track)
                # Keep a reference so the task isn't garbage-collected mid-run.
                self._frame_task = asyncio.create_task(self._read_frames())

    @function_tool()
    async def give_candy(self, context: RunContext, candy_name: str) -> str:
        """Pick up one candy from the shelf and hand it to the user.

        Args:
            candy_name: The candy the user asked for, e.g. "KitKat" or "lollipop".
        """
        # Delegate the physical routine to a task and await its typed result.
        succeeded = await GiveCandy(
            room=self.room,
            candy_name=candy_name,
            chat_ctx=self.chat_ctx,
        )
        if succeeded:
            return f"Handed the {candy_name} to the user."
        return f"Something went wrong giving the {candy_name}."

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
            destination_identity=MOVE_TO_IDENTITY,
            method="reset_robot",
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
