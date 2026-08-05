from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gensokyo.world.ids import EventId, LocationId


class EventKind(StrEnum):
    PLAYER_UTTERANCE = "player_utterance"
    PLAYER_ACTION = "player_action"
    NPC_UTTERANCE = "npc_utterance"
    NPC_ACTION = "npc_action"
    TOPIC_TOUCHED = "topic_touched"
    """玩家第一次聊到某个 NPC 在意的话题。

    单独一种事件而不是复用 `PLAYER_UTTERANCE`：好感变化必须能从事件日志里
    追溯到原因，而「她为什么涨了 4 点好感」和「玩家说了句话」是两件事。
    """
    MEMORY_LOST = "memory_lost"
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
