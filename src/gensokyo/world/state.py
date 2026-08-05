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
    gift_counts: dict[ItemId, int] = Field(default_factory=dict)
    """每样东西被送过几次，用于送礼的边际递减（`GIFT_ATTITUDE_STEPS`）。

    **按物品种类计数，不是按总次数**：换一样东西送应该重新算，否则「送第四件
    不同的礼物」和「第四次投同一枚币」会被当成同一件事，而后者才是要掐死的磨。
    """
    discussed_topics: set[str] = Field(default_factory=set)
    """玩家已经跟她聊过的话题，用于「同一话题只涨一次好感」。

    它是 `apply()` 的结果，所以自动能被动作日志回放重建——坑 #9 立的规矩：
    新机制先问「它是动作的结果，还是时间的结果」。"""


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

# 玩家初始赛钱。送礼有边际递减（6/3/1，`GIFT_ATTITUDE_STEPS`），所以同一枚
# 币投第四次起不再涨好感——8 枚的意义不是「够投八次」，而是「投够了还有余量
# 分给另一个人」。灵梦门槛 16、芙兰 12，两边都必须靠聊天补最后一截。
# 没有初始赛钱游戏无法通关。
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
