"""Following the site's control queue: who the shop listens to, and when it speaks.

The web hands the microphone to one visitor at a time and rotates the turn on a
20-second timer, so the interesting cases are all handovers. Every test drives
the same room events the Playground's queue API really produces — a grant writes
`vla_demo.control_until`, a revoke writes the empty string — and asserts what the
session was pointed at.
"""

import asyncio

import pytest

from voice_agent.agent import TurnFollower
from voice_agent.config import CONTROL_UNTIL_ATTR, VISITOR_KIND, VISITOR_KIND_ATTR


class FakeParticipant:
    def __init__(self, identity: str, **attributes: str) -> None:
        self.identity = identity
        self.attributes = attributes


class FakeRoom:
    def __init__(self, *participants: FakeParticipant) -> None:
        self.remote_participants = {p.identity: p for p in participants}
        self._handlers: dict[str, list] = {}

    def on(self, event: str, handler) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler) -> None:
        self._handlers.get(event, []).remove(handler)

    def _emit(self, event: str, *args) -> None:
        for handler in list(self._handlers.get(event, [])):
            handler(*args)

    def join(self, participant: FakeParticipant) -> None:
        self.remote_participants[participant.identity] = participant
        self._emit("participant_connected", participant)

    def leave(self, participant: FakeParticipant) -> None:
        self.remote_participants.pop(participant.identity, None)
        self._emit("participant_disconnected", participant)

    def grant(self, participant: FakeParticipant, until: str = "1763000000000") -> None:
        participant.attributes[CONTROL_UNTIL_ATTR] = until
        self._emit(
            "participant_attributes_changed", {CONTROL_UNTIL_ATTR: until}, participant
        )

    def revoke(self, participant: FakeParticipant) -> None:
        participant.attributes[CONTROL_UNTIL_ATTR] = ""
        self._emit(
            "participant_attributes_changed", {CONTROL_UNTIL_ATTR: ""}, participant
        )


class FakeRoomIO:
    def __init__(self) -> None:
        # Whatever RoomIO's own default picked before we took over.
        self.linked: str | None = "robot"
        self.calls: list[str | None] = []

    def set_participant(self, identity: str) -> None:
        self.linked = identity
        self.calls.append(identity)

    def unset_participant(self) -> None:
        self.linked = None
        self.calls.append(None)


class FakeSession:
    def __init__(self) -> None:
        self.room_io = FakeRoomIO()
        self.greetings: list[str | None] = []

    def generate_reply(self, *, instructions: str) -> None:
        self.greetings.append(self.room_io.linked)


class FakeAssistant:
    def __init__(self) -> None:
        self.turns_ended = 0

    async def end_turn(self, session) -> None:
        self.turns_ended += 1


def machines() -> list[FakeParticipant]:
    """What the site's dispatch gate guarantees is already in the room."""
    return [
        FakeParticipant("robot"),
        FakeParticipant("move-to-operator"),
        FakeParticipant("policy-operator"),
        FakeParticipant("reward-operator"),
    ]


def viewer(identity: str) -> FakeParticipant:
    return FakeParticipant(identity, **{VISITOR_KIND_ATTR: VISITOR_KIND})


async def settle() -> None:
    """Let the follower's worker drain every event it has been handed."""
    for _ in range(20):
        await asyncio.sleep(0)


@pytest.fixture
async def shop():
    """A room of machines, with the follower already watching it."""
    room = FakeRoom(*machines())
    session = FakeSession()
    assistant = FakeAssistant()
    follower = TurnFollower(room, session, assistant)
    follower.start()
    await settle()
    yield room, session, assistant
    await follower.aclose()


async def test_a_room_of_machines_leaves_the_link_alone(shop):
    """Never unlink on an empty shop: that resets the future RoomIO is still using
    to publish our audio track, and the agent would come up voiceless."""
    _, session, assistant = shop
    assert session.room_io.calls == []
    assert session.room_io.linked == "robot"
    assert assistant.turns_ended == 0


async def test_the_lone_visitor_is_linked_on_join_and_greeted_on_the_grant(shop):
    """The site seats a visitor a beat AFTER they join, so the link and the hello
    are two separate moments — and the grant doesn't move the link at all."""
    room, session, assistant = shop
    alice = viewer("portal-web-aaaa")

    room.join(alice)
    await settle()
    assert session.room_io.linked == "portal-web-aaaa"
    assert session.greetings == []  # no microphone yet, nobody to hear it

    room.grant(alice)
    await settle()
    assert session.room_io.calls == ["portal-web-aaaa"]  # link unchanged
    assert session.greetings == ["portal-web-aaaa"]
    assert assistant.turns_ended == 0  # first customer, nothing to close out


async def test_a_renewal_is_not_a_new_customer(shop):
    """20-second turns renew off a 5s poll — several times a minute."""
    room, session, _ = shop
    alice = viewer("portal-web-aaaa")
    room.join(alice)
    room.grant(alice)
    await settle()

    for _ in range(3):
        room.grant(alice, until="1763000099999")
        await settle()

    assert session.greetings == ["portal-web-aaaa"]


async def test_a_stale_deadline_keeps_the_customer(shop):
    """A deadline in the past is the norm mid-turn, not a lost turn."""
    room, session, _ = shop
    alice = viewer("portal-web-aaaa")
    bob = viewer("portal-web-bbbb")
    room.join(alice)
    room.grant(alice, until="1")  # long expired by any clock
    room.join(bob)
    await settle()

    assert session.room_io.linked == "portal-web-aaaa"


async def test_handover_closes_out_the_old_customer_and_greets_the_new_one(shop):
    room, session, assistant = shop
    alice = viewer("portal-web-aaaa")
    bob = viewer("portal-web-bbbb")
    room.join(alice)
    room.grant(alice)
    room.join(bob)
    await settle()
    assert session.room_io.linked == "portal-web-aaaa"

    # What reconcile() does when the turn rotates: revoke, then promote.
    room.revoke(alice)
    room.grant(bob)
    await settle()

    assert session.room_io.linked == "portal-web-bbbb"
    assert assistant.turns_ended == 1
    assert session.greetings == ["portal-web-aaaa", "portal-web-bbbb"]


async def test_a_reload_comes_back_as_a_stranger(shop):
    """Every page load mints a fresh `portal-web-<hex>`, so a reload is a new
    participant — the whole reason the link can't be pinned at session start."""
    room, session, _ = shop
    before = viewer("portal-web-aaaa")
    room.join(before)
    room.grant(before)
    await settle()

    after = viewer("portal-web-cccc")
    room.join(after)
    room.leave(before)
    await settle()

    assert session.room_io.linked == "portal-web-cccc"

    room.grant(after)
    await settle()
    assert session.greetings[-1] == "portal-web-cccc"


async def test_the_last_visitor_leaving_closes_the_shop(shop):
    room, session, assistant = shop
    alice = viewer("portal-web-aaaa")
    room.join(alice)
    room.grant(alice)
    await settle()

    room.leave(alice)
    await settle()

    assert session.room_io.linked is None
    assert assistant.turns_ended == 1  # arm parked, conversation wiped


async def test_releasing_the_turn_does_not_re_greet_the_same_visitor(shop):
    """Alice passes her turn and takes it again. Nobody else spoke, so her
    conversation was never wiped — greeting her again would be nonsense."""
    room, session, assistant = shop
    alice = viewer("portal-web-aaaa")
    room.join(alice)
    room.grant(alice)
    await settle()

    room.revoke(alice)
    await settle()
    room.grant(alice)
    await settle()

    assert session.greetings == ["portal-web-aaaa"]
    assert assistant.turns_ended == 0


async def test_a_burst_of_events_collapses_into_one_handover(shop):
    """The queue writes several attributes per reconcile and the roster can churn
    in the same tick; the follower re-reads the room, so it lands once."""
    room, session, assistant = shop
    alice = viewer("portal-web-aaaa")
    bob = viewer("portal-web-bbbb")
    room.join(alice)
    room.grant(alice)
    await settle()

    room.join(bob)
    room.revoke(alice)
    room.grant(bob)
    room.leave(alice)
    await settle()

    assert session.room_io.linked == "portal-web-bbbb"
    assert assistant.turns_ended == 1


async def test_the_gap_between_revoke_and_grant_does_not_move_the_link(shop):
    """A rotation is two writes, so the room briefly has no turn-holder at all.

    Hopping to a spectator in that window would park the arm and wipe the
    conversation on the way past — twice per rotation, every 20 seconds.
    """
    room, session, assistant = shop
    alice = viewer("portal-web-aaaa")
    spectators = [viewer("portal-web-bbbb"), viewer("portal-web-cccc")]
    room.join(alice)
    room.grant(alice)
    for spectator in spectators:
        room.join(spectator)
    await settle()
    assert session.room_io.linked == "portal-web-aaaa"

    room.revoke(alice)  # ...and nothing granted yet
    await settle()
    assert session.room_io.linked == "portal-web-aaaa"
    assert assistant.turns_ended == 0

    room.grant(spectators[1])
    await settle()
    assert session.room_io.linked == "portal-web-cccc"
    assert assistant.turns_ended == 1


async def test_an_out_of_order_departure_does_not_bounce_through_a_spectator(shop):
    """Two tabs closing at once can be observed in either order; the outgoing
    spectator must not become the customer on the way to an empty shop."""
    room, session, assistant = shop
    alice = viewer("portal-web-aaaa")
    bob = viewer("portal-web-bbbb")
    room.join(alice)
    room.join(bob)
    room.grant(bob)
    await settle()
    assert session.room_io.linked == "portal-web-bbbb"

    room.leave(bob)  # the turn-holder goes first, alice is still listed
    await settle()
    assert session.room_io.linked == "portal-web-aaaa"  # only visitor left
    room.leave(alice)
    await settle()

    assert session.room_io.linked is None
    assert assistant.turns_ended == 2
