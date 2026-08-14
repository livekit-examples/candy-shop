"""The room is full of machines; only one participant is the customer."""

import time

from voice_agent import visitor
from voice_agent.config import CONTROL_UNTIL_ATTR, VISITOR_KIND, VISITOR_KIND_ATTR


class FakeParticipant:
    def __init__(self, identity: str, **attributes: str) -> None:
        self.identity = identity
        self.attributes = attributes


class FakeRoom:
    """Just the surface visitor.py touches: a participant map."""

    def __init__(self, *participants: FakeParticipant) -> None:
        self.remote_participants = {p.identity: p for p in participants}


def machines() -> list[FakeParticipant]:
    """What the site's dispatch gate guarantees is already in the room."""
    return [
        FakeParticipant("robot"),
        FakeParticipant("move-to-operator"),
        FakeParticipant("policy-operator"),
        FakeParticipant("reward-operator"),
    ]


def viewer(identity: str, control_until: str | None = None) -> FakeParticipant:
    attributes = {VISITOR_KIND_ATTR: VISITOR_KIND}
    if control_until is not None:
        attributes[CONTROL_UNTIL_ATTR] = control_until
    return FakeParticipant(identity, **attributes)


def in_a_minute() -> str:
    return str(int(time.time() * 1000) + 60_000)


def a_minute_ago() -> str:
    return str(int(time.time() * 1000) - 60_000)


def test_machines_are_not_visitors():
    """The bug: RoomIO's default linked one of these and heard silence forever."""
    room = FakeRoom(*machines())
    for participant in machines():
        assert not visitor.is_visitor(participant)
    assert visitor.holder(room) is None
    assert visitor.any_visitor(room) is None


def test_the_turn_holder_is_found_among_spectators_and_machines():
    holder = viewer("portal-web-aaaa", in_a_minute())
    room = FakeRoom(*machines(), viewer("portal-web-bbbb"), holder)
    assert visitor.holder(room) is holder


def test_a_visitor_who_has_not_claimed_the_turn_yet_is_still_a_visitor():
    """The site claims the turn a beat after joining; link them meanwhile."""
    spectator = viewer("portal-web-bbbb")
    room = FakeRoom(*machines(), spectator)
    assert visitor.holder(room) is None
    assert visitor.any_visitor(room) is spectator


def test_a_stale_deadline_is_still_the_turn():
    """The site renews lazily off a 5s poll, so a live 20s turn is often 'expired'.

    Treating that as a lost turn made the agent disown its own customer several
    times a minute; only the queue clearing the attribute ends a turn.
    """
    holder = viewer("portal-web-aaaa", a_minute_ago())
    assert visitor.holds_turn(holder)

    other = viewer("portal-web-bbbb")
    assert visitor.holder(FakeRoom(*machines(), other, holder)) is holder


def test_a_cleared_deadline_is_not_the_turn():
    """What a revoke actually writes: the empty string."""
    assert not visitor.holds_turn(viewer("portal-web-aaaa", ""))
    assert not visitor.holds_turn(viewer("portal-web-aaaa"))


def test_a_machine_carrying_a_deadline_is_still_not_the_customer():
    """Belt and braces: only `vla_demo.kind` decides who may hold a turn."""
    impostor = FakeParticipant("policy-operator", **{CONTROL_UNTIL_ATTR: in_a_minute()})
    assert not visitor.holds_turn(impostor)
    assert visitor.holder(FakeRoom(impostor)) is None
