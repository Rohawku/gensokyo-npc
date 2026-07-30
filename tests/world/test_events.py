import pytest
from pydantic import ValidationError

from gensokyo.world.events import Event, EventKind
from gensokyo.world.ids import EventId, LocationId


def test_event_is_frozen() -> None:
    ev = Event(
        id=EventId("e00001"),
        tick=1,
        kind=EventKind.PLAYER_UTTERANCE,
        actor="player",
        location=LocationId("hakurei_shrine"),
        payload={"text": "你好"},
    )
    with pytest.raises(ValidationError):
        ev.payload = {}  # type: ignore[misc]


def test_event_caused_by_defaults_to_none() -> None:
    ev = Event(
        id=EventId("e00002"),
        tick=1,
        kind=EventKind.NPC_ACTION,
        actor="reimu",
        location=LocationId("hakurei_shrine"),
    )
    assert ev.caused_by is None
    assert ev.payload == {}
