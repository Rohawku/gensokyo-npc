from pydantic import BaseModel, ConfigDict, Field

from gensokyo.world.ids import FactId, ItemId, LocationId, NpcId


class StrictModel(BaseModel):
    """静态定义的共同基类。

    YAML 里拼错字段必须立刻报错。角色差异全部由这些数据承载，
    静默退回默认值会表现成「情绪 gate 不生效」这类极难定位的行为 bug。
    """

    model_config = ConfigDict(extra="forbid")


class SpeechCfg(StrictModel):
    style: str
    forbidden_phrases: list[str] = Field(default_factory=list)
    quirks: list[str] = Field(default_factory=list)


class PersonaCfg(StrictModel):
    core: str
    speech: SpeechCfg


class MemoryCfg(StrictModel):
    """W1 载入但不使用，W2 的分层记忆会读它。"""

    lambda_decay: float
    salience_multipliers: dict[str, float] = Field(default_factory=dict)
    reflection_threshold: float


class EmotionMode(StrictModel):
    name: str
    range: tuple[float, float]
    tools_allow: list[str] = Field(default_factory=list)
    tools_deny: list[str] = Field(default_factory=list)
    speech_hint: str = ""

    def contains(self, value: float) -> bool:
        """左闭右开，保证模式区间不重叠。最高档的上界特殊处理为闭区间。"""
        low, high = self.range
        if high >= 1.0:
            return low <= value <= high
        return low <= value < high


class EmotionCfg(StrictModel):
    variable: str
    initial: float = 0.0
    decay_per_tick: float = 0.0
    modes: list[EmotionMode]


class ToolsCfg(StrictModel):
    deny_always: list[str] = Field(default_factory=list)


class DormantMemoryCfg(StrictModel):
    """W1 载入但不使用，W2 的沉睡记忆会读它。"""

    content_key: str
    trigger_keys: list[str]
    hint: str


class KnowledgeCfg(StrictModel):
    holds_facts: list[FactId] = Field(default_factory=list)
    forbidden_knowledge: list[str] = Field(default_factory=list)
    dormant_memories: list[DormantMemoryCfg] = Field(default_factory=list)


class CharacterCard(StrictModel):
    id: NpcId
    name: str
    home: LocationId
    persona: PersonaCfg
    memory: MemoryCfg
    emotion: EmotionCfg
    tools: ToolsCfg = Field(default_factory=ToolsCfg)
    knowledge: KnowledgeCfg = Field(default_factory=KnowledgeCfg)
    behavior_baseline: dict[str, dict[str, float]] = Field(default_factory=dict)


class RevealConditions(StrictModel):
    attitude_gte: int | None = None
    traded_item_in: list[ItemId] = Field(default_factory=list)


class FactDef(StrictModel):
    id: FactId
    holder: NpcId
    content: str
    reveal_conditions: RevealConditions = Field(default_factory=RevealConditions)
    is_clue: bool = False


class ItemDef(StrictModel):
    id: ItemId
    name: str
    description: str = ""


class LocationDef(StrictModel):
    id: LocationId
    name: str
    description: str = ""
    exits: list[LocationId] = Field(default_factory=list)
    items: dict[ItemId, int] = Field(default_factory=dict)


class WorldDefs(StrictModel):
    locations: dict[LocationId, LocationDef]
    items: dict[ItemId, ItemDef]
    facts: dict[FactId, FactDef]
    characters: dict[NpcId, CharacterCard]

    def clue_facts(self) -> set[FactId]:
        return {fid for fid, f in self.facts.items() if f.is_clue}
