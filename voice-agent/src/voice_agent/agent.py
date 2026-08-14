import asyncio
import logging
import pathlib
from contextlib import suppress
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
from livekit.agents.llm import ChatContext, ImageContent
from livekit.plugins import ai_coustics

from voice_agent import visitor
from voice_agent.config import (
    CONTROL_UNTIL_ATTR,
    LLM_MODEL,
    ROBOT_IDENTITY,
    RPC_TIMEOUT_S,
    STT_MODEL,
    TTS_MODEL,
    TTS_VOICE,
    VIDEO_TRACK_NAME,
    VISITOR_KIND_ATTR,
)
from voice_agent.task import GiveCandy

logger = logging.getLogger("agent")

# Shared by the opening greeting and by every handover to a new turn-holder, who
# arrives mid-session and has heard nothing.
GREETING = (
    "Greet the customer in one short sentence and tell them you have "
    "KitKat, Nerds, Twix, and Snickers."
)

# Single source of config: the repo-root .env (.env.local overrides it). The
# hosted deployment ships neither file — LiveKit Cloud injects LIVEKIT_URL and
# the keys as environment variables — and load_dotenv on a missing path is a
# no-op, so the same two lines cover the container.
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
        # The errand currently driving the arm, so a handover can abandon it.
        self._errand: GiveCandy | None = None

    async def on_enter(self):
        self.room = get_job_context().room

        def attach(track, publication, participant) -> None:
            if (
                track.kind == rtc.TrackKind.KIND_VIDEO
                and participant.identity == ROBOT_IDENTITY
                and publication.name == VIDEO_TRACK_NAME
            ):
                self.video_stream = rtc.VideoStream(track)
                # Keep a reference so the task isn't garbage-collected mid-run.
                self._frame_task = asyncio.create_task(self._read_frames())

        self.room.on("track_subscribed", attach)

        # The room is connected before the session starts, and the site holds our
        # dispatch back until the robot is in it — so its camera is usually already
        # subscribed and `track_subscribed` fired long ago. Sweep what's there, or
        # `look` never gets a frame.
        for participant in self.room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track is not None:
                    attach(publication.track, publication, participant)

        # No greeting here: on_enter fires once, when the session starts, which is
        # before the site has seated anyone — a hello into an empty room.
        # TurnFollower says it, at the moment a customer can actually hear it.

    async def end_turn(self, session: AgentSession) -> None:
        """Close out the current customer before the counter changes hands.

        Parks the arm, stops us mid-sentence, and wipes the conversation, so the
        next turn-holder walks up to an idle shop rather than into the middle of
        someone else's order. Never raises: a failed handover must not take the
        session down with it.
        """
        errand, self._errand = self._errand, None
        if errand is not None:
            logger.info("turn ended mid-errand; abandoning %s", errand.candy_name)
            with suppress(Exception):
                await errand.abort()

        with suppress(Exception):
            await session.interrupt()

        with suppress(Exception):
            await self.update_chat_ctx(ChatContext())

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
        errand = GiveCandy(
            room=self.room,
            candy_name=candy_name,
            chat_ctx=self.chat_ctx,
        )
        self._errand = errand
        try:
            succeeded = await errand
        finally:
            # Only clear our own; a handover already swapped in None and may have
            # started something else.
            if self._errand is errand:
                self._errand = None
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


class TurnFollower:
    """Keeps the session's ear on whichever visitor the site says is the customer.

    RoomIO links a participant exactly once and never again — on a disconnect it
    re-arms its internal future, but the task that awaited it finished long ago —
    so following the control queue is ours to do.

    Every roster or attribute change re-derives the customer from the room and
    re-points the session, rather than reacting to the specific event. That is
    what makes it self-healing: grants, renewals, direct handoffs, releases, a
    turn that rotates every 20 seconds, and a reload that comes back under a
    fresh `portal-web-<hex>` identity all reduce to the same question — who is
    the customer now?
    """

    def __init__(
        self, room: rtc.Room, session: AgentSession, assistant: CandyShopAssistant
    ) -> None:
        self._room = room
        self._session = session
        self._assistant = assistant
        # Who we last pointed the session at, and who we've already said hello to.
        # Tracked here rather than read back from RoomIO, whose `linked_participant`
        # future isn't updated the way we drive it.
        self._linked: str | None = None
        self._greeted: str | None = None
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="follow-turn")

    def start(self) -> None:
        self._room.on("participant_connected", self._touch)
        self._room.on("participant_disconnected", self._touch)
        self._room.on("participant_attributes_changed", self._on_attributes)
        # Someone may already hold the turn by the time we get here.
        self._touch()

    async def aclose(self) -> None:
        self._room.off("participant_connected", self._touch)
        self._room.off("participant_disconnected", self._touch)
        self._room.off("participant_attributes_changed", self._on_attributes)
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task

    def _touch(self, *_: object) -> None:
        self._wake.set()

    def _on_attributes(
        self, changed: dict[str, str], participant: rtc.Participant
    ) -> None:
        # The queue writes both keys in one update, and nothing else it writes can
        # change who the customer is.
        if CONTROL_UNTIL_ATTR in changed or VISITOR_KIND_ATTR in changed:
            self._wake.set()

    async def _run(self) -> None:
        """Serialize the re-scans onto one worker.

        A handover awaits an arm park and a context wipe, long enough that a burst
        of room events would otherwise overlap mid-teardown. The flag also
        coalesces that burst into a single pass, which is safe because `_sync`
        re-reads the room every time — a coalesced event is never a missed one.
        """
        while True:
            await self._wake.wait()
            self._wake.clear()
            try:
                await self._sync()
            except Exception:
                logger.exception("failed to follow the control queue")

    def _customer(self) -> rtc.RemoteParticipant | None:
        """Who to listen to: the turn-holder, else stay put, else any visitor.

        The middle rule is what keeps the link steady. A rotation is two separate
        writes — revoke the old holder, then grant the new one — so there are
        moments with nobody holding the turn at all, and a departure can be
        observed out of order. Falling straight through to "any visitor" in those
        gaps hops the link to an arbitrary spectator and fires a handover on the
        way past, parking the arm and wiping the conversation for nothing. Staying
        with the visitor we already have costs us nothing: if they've lost the turn
        they've lost the microphone too, so there is nothing to mishear.
        """
        if (turn_holder := visitor.holder(self._room)) is not None:
            return turn_holder
        linked = self._room.remote_participants.get(self._linked or "")
        if linked is not None and visitor.is_visitor(linked):
            return linked
        return visitor.any_visitor(self._room)

    async def _sync(self) -> None:
        customer = self._customer()
        identity = customer.identity if customer else None

        if identity != self._linked:
            previous, self._linked = self._linked, identity
            logger.info("customer: %s -> %s", previous, identity)
            if identity is None:
                self._session.room_io.unset_participant()
            else:
                self._session.room_io.set_participant(identity)
            # Nothing to close out on the session's first customer.
            if previous is not None:
                await self._assistant.end_turn(self._session)

        # Keyed on the customer, not on the link: a lone visitor gets linked while
        # still a mic-less spectator, and the grant that hands them a microphone a
        # moment later changes nothing about the link — but it is the first moment
        # they can hear us, so it's the moment to say hello.
        if (
            customer is not None
            and visitor.holds_turn(customer)
            and identity != self._greeted
        ):
            self._greeted = identity
            self._session.generate_reply(instructions=GREETING)


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

    await ctx.connect()

    assistant = CandyShopAssistant()
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_options=room_io.RoomOptions(
            # `participant_identity` is deliberately NOT set, even though RoomIO's
            # default picks a machine (the robot and all three operators are
            # STANDARD too). Pinning is the worse failure: RoomIO publishes our
            # audio track from the same task that waits for the pinned identity,
            # and every page load mints a fresh `portal-web-<hex>`, so pinning a
            # visitor who reloads leaves that task waiting on someone who will
            # never return — and the relink below swaps its future out from under
            # it, so the shop never gets a voice at all. Left unset, RoomIO links
            # whichever machine it finds, finishes wiring up, and TurnFollower
            # immediately re-points it at the real customer.
            #
            # The shop outlives any one customer: spectators come and go and the
            # turn rotates every 20s, so a departure must not end the session.
            close_on_disconnect=False,
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )

    follower = TurnFollower(ctx.room, session, assistant)
    follower.start()
    ctx.add_shutdown_callback(follower.aclose)


def main() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    main()
