from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gensokyo.world.ids import EventId, LocationId


class EventKind(StrEnum):
    PLAYER_UTTERANCE = "player_utterance"
    PLAYER_ACTION = "player_action"
    NPC_UTTERANCE = "npc_utterance"
    NPC_ACTION = "npc_action"
    WORLD_CHANGE = "world_change"
    QUEST_ADVANCE = "quest_advance"


class Event(BaseModel):
    """不可变的世界事件。event_log 是 append-only 的唯一真相来源。"""

    model_config = ConfigDict(frozen=True)

    id: EventId
    tick: int
    kind: EventKind
    actor: str
    location: LocationId
    payload: dict[str, Any] = Field(default_factory=dict)
    caused_by: EventId | None = None
