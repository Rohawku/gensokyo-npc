from enum import IntEnum

from pydantic import BaseModel, Field

from gensokyo.world.defs import WorldDefs
from gensokyo.world.events import Event
from gensokyo.world.ids import FactId, ItemId, LocationId, NpcId
from gensokyo.world.tools import Action


class QuestStage(IntEnum):
    S0_UNAWARE = 0
    S1_ANOMALY = 1
    S2_CLUES = 2
    S3_SOURCE = 3
    S4_END = 4


class LocationState(BaseModel):
    id: LocationId
    items: dict[ItemId, int] = Field(default_factory=dict)


class PlayerState(BaseModel):
    location: LocationId
    inventory: dict[ItemId, int] = Field(default_factory=dict)
    known_facts: set[FactId] = Field(default_factory=set)
    oblivion_exposure: int = 0
    """在无缘塚连续行动的次数。离开花田即清零，累到阈值会丢掉一条线索。

    按动作数而非 tick 计数，是为了让它进动作日志、能被 replay 精确重现——
    挂在 tick 上会让存档读档时丢掉的线索凭空回来。"""


class NpcState(BaseModel):
    id: NpcId
    location: LocationId
    attitude: int = 0
    emotion_var: str
    emotion: float = 0.0
    mode: str
    inventory: dict[ItemId, int] = Field(default_factory=dict)
    holds_facts: set[FactId] = Field(default_factory=set)
    revealed_facts: set[FactId] = Field(default_factory=set)
    received_items: set[ItemId] = Field(default_factory=set)


class QuestState(BaseModel):
    stage: QuestStage = QuestStage.S0_UNAWARE
    clues_obtained: set[FactId] = Field(default_factory=set)
    ending: str | None = None


class WorldState(BaseModel):
    tick: int = 0
    seq: int = 0
    player: PlayerState
    npcs: dict[NpcId, NpcState]
    locations: dict[LocationId, LocationState]
    quest: QuestState = Field(default_factory=QuestState)
    event_log: list[Event] = Field(default_factory=list)
    action_log: list[Action] = Field(default_factory=list)


PLAYER_START = LocationId("hakurei_shrine")

# 玩家初始赛钱。灵梦的线索门槛是好感 24、芙兰是 12，而送一次礼物 +6，
# 所以 4 枚给灵梦、2 枚给芙兰，留 2 枚余量。没有初始赛钱游戏无法通关。
STARTING_COIN = ItemId("offering_coin")
STARTING_COIN_COUNT = 8


def build_initial_state(defs: WorldDefs) -> WorldState:
    from gensokyo.world.rules import resolve_mode

    npcs: dict[NpcId, NpcState] = {}
    for npc_id, card in defs.characters.items():
        npcs[npc_id] = NpcState(
            id=npc_id,
            location=card.home,
            emotion_var=card.emotion.variable,
            emotion=card.emotion.initial,
            mode=resolve_mode(card, card.emotion.initial),
            holds_facts=set(card.knowledge.holds_facts),
        )

    locations = {
        loc_id: LocationState(id=loc_id, items=dict(loc.items))
        for loc_id, loc in defs.locations.items()
    }

    return WorldState(
        player=PlayerState(location=PLAYER_START, inventory={STARTING_COIN: STARTING_COIN_COUNT}),
        npcs=npcs,
        locations=locations,
    )
