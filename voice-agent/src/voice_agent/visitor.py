"""Which participant in the room is the customer.

The Playground room is not a two-party call. The robot, the move-to / policy /
reward operators, and every spectator sit in it alongside the one visitor who
holds the control turn — and all of them are plain `STANDARD` participants. So
RoomIO's default "link the first standard participant" picks a machine that
publishes no microphone, and the agent listens to silence forever. The site's
dispatch gate makes that a certainty rather than a race: it holds the agent back
until all three operators have joined, so the room is always full of machines
before the agent ever arrives.

The visitor is identified by the `vla_demo.*` attributes the site's token
endpoint and queue API set. That prefix is a wire contract shared with
`web/components/playground/livekit/control.ts` — rename it on either side and
the agent silently stops finding its customer.
"""

from livekit import rtc

from voice_agent.config import CONTROL_UNTIL_ATTR, VISITOR_KIND, VISITOR_KIND_ATTR


def is_visitor(participant: rtc.Participant) -> bool:
    """A browser visitor, as opposed to the robot or an operator."""
    return participant.attributes.get(VISITOR_KIND_ATTR) == VISITOR_KIND


def holds_turn(participant: rtc.Participant) -> bool:
    """Holds the control turn, so the site has granted it a microphone.

    Presence of the attribute is the whole test: the epoch-ms deadline it carries
    is deliberately NOT compared against our clock. The site's queue is
    serverless and has no timers, so it renews a turn lazily off the holder's own
    ~5s poll — and with 20-second turns (`web/lib/queue.ts`) the deadline sits a
    few seconds in the past for much of the turn while the holder still has
    `canPublish` and is still talking to us. Reading it as expiry made the agent
    disown its own customer several times a minute.

    The queue clears the attribute when it really does revoke, and a visitor who
    closes the tab takes their attributes with them, so "set" and "is the
    customer" agree without us second-guessing the deadline.
    """
    return is_visitor(participant) and bool(
        participant.attributes.get(CONTROL_UNTIL_ATTR)
    )


def holder(room: rtc.Room) -> rtc.RemoteParticipant | None:
    """The visitor the site has handed the microphone to, if anyone holds it."""
    return next((p for p in room.remote_participants.values() if holds_turn(p)), None)


def any_visitor(room: rtc.Room) -> rtc.RemoteParticipant | None:
    """Some browser visitor, turn or no turn.

    Worth linking one even before they're seated: the site claims the turn a moment
    *after* joining, so linking straight away means the audio path is already wired
    when the grant lands. Which one hardly matters — a visitor without a turn has
    no microphone for us to hear either way.
    """
    return next((p for p in room.remote_participants.values() if is_visitor(p)), None)
