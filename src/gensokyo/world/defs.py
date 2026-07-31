from pydantic import BaseModel, ConfigDict, Field, field_validator

from gensokyo.world.ids import FactId, ItemId, LocationId, NpcId

SALIENCE_BASELINE: dict[str, float] = {
    "player_gave_item": 0.5,
    "npc_took_item": 0.6,
    "player_talked": 0.1,
    "player_arrived": 0.3,
    "revealed_info": 0.7,
    "asked_player": 0.15,
    "spellcard_duel": 0.9,
    "item_broken": 0.8,
    "quest_advance": 0.4,
    "memory_lost": 0.6,
}
"""事件类型的记忆显著性基线。键是**规范键**——角色卡的
`salience_multipliers` 只能用这些，`MemoryCfg` 会校验。

放在这里而不是 `memory/`：`world/` 不许 import 项目内其他模块（取舍 #1），
而这张表要在加载角色卡时就用上。它和 `rules.py` 的 `ATTITUDE_DELTA` /
`EMOTION_DELTA` 是同一类东西——事件到数值的静态表。一处定义，
`memory/salience.py` 读它，不另抄一份。

**这张表只登记 W1 真的会产生、她真的感知得到、而且召回出来有用的事件。**
删过两条：

- `player_left`：移动事件的 `location` 是终点（`_emit` 取动作后的位置），
  所以原地点的 NPC 收不到「有人走了」，那个键永远命中不了。
- `npc_talked`（她自己说过的话）：实测它会把复读**喂回去**。越狱局里召回给
  她的第一条是「我说：『你到底想干啥？』（这样的事有 6 次）」，而同一个
  prompt 里的禁语清单说的是「这些一句都不许再说」——同一份内容出现两次、
  指令相反。她最近说过什么，12 轮原话窗口和禁语清单都已经覆盖了，记忆层
  在这件事上加不了任何信息。「我告诉过他哪条情报」是另一个键
  （`revealed_info`），那个留着，它防的是把同一条线索说两遍。
"""


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
    lambda_decay: float
    """时间衰减率，单位是**事件**而不是回合。芙兰大（忘得快），
    魔理沙小（记得谁欠她书），灵梦居中。参数从人设推导。"""
    salience_multipliers: dict[str, float] = Field(default_factory=dict)
    reflection_threshold: float

    @field_validator("salience_multipliers")
    @classmethod
    def _keys_must_be_known(cls, value: dict[str, float]) -> dict[str, float]:
        """键拼错必须加载失败。

        这三张角色卡原先写的是 `receive_gift`、`someone_plays_with_me`、
        `magic_theory`——**一个都对不上真实事件名**（真实的是
        `player_gave_item`）。dict 字段的键拼错不会被 `extra="forbid"` 拦住，
        系数于是静默退回 1.0，「芙兰对陪玩敏感」这个人设差异从来没生效过。
        表现是「三个 NPC 的记忆行为一模一样」而不是加载失败，正是坑 #5
        那类最难查的 bug。
        """
        unknown = sorted(set(value) - set(SALIENCE_BASELINE))
        if unknown:
            raise ValueError(
                f"salience_multipliers 含未知键：{'、'.join(unknown)}。"
                f"可用的键只有：{'、'.join(sorted(SALIENCE_BASELINE))}"
            )
        return value


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
    event_deltas: dict[str, float] = Field(default_factory=dict)
    """同一事件对不同角色的情绪影响方向不同：收到赛钱让灵梦「没那么烦」，
    却让芙兰「更兴奋」。所以增量必须按角色配置，不能用全局表。"""


class ToolsCfg(StrictModel):
    deny_always: list[str] = Field(default_factory=list)
    deny_reasons: dict[str, str] = Field(default_factory=dict)


class DormantMemoryCfg(StrictModel):
    """W1 载入但不使用，W2 的沉睡记忆会读它。"""

    content_key: str
    trigger_keys: list[str]
    hint: str


class KnowledgeCfg(StrictModel):
    holds_facts: list[FactId] = Field(default_factory=list)
    forbidden_knowledge: list[str] = Field(default_factory=list)
    blind_to_outside: bool = False
    """该角色不知道自己所在地点之外发生的事。信息隔离的机器开关，
    与 forbidden_knowledge 的中文散文分开——一个字符串不该同时当键和文案。"""
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


class PrologueDef(StrictModel):
    title: str
    text: str
    objective_hint: str = ""


class StageDef(StrictModel):
    id: str
    hint: str
    """给 NPC 看的进展说法，进 prompt。"""
    objective: str
    """给玩家看的当前目标。开局只给指令表、不说要干什么，玩家会一脸茫然。"""


class EndingDef(StrictModel):
    id: str
    by: NpcId | None = None
    """由哪个 NPC 收尾。None 表示失败结局，没人解决。"""
    title: str
    text: str


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
    endings: dict[str, EndingDef]
    stages: dict[str, StageDef]
    prologue: PrologueDef

    def ending_by(self, npc_id: NpcId) -> EndingDef | None:
        for ending in self.endings.values():
            if ending.by == npc_id:
                return ending
        return None

    def clue_facts(self) -> set[FactId]:
        return {fid for fid, f in self.facts.items() if f.is_clue}
