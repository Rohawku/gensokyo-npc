from pydantic import BaseModel, ConfigDict, Field, field_validator

from gensokyo.world.ids import FactId, ItemId, LocationId, NpcId

SALIENCE_BASELINE: dict[str, float] = {
    "player_gave_item": 0.5,
    "npc_took_item": 0.6,
    "player_talked": 0.1,
    "topic_touched": 0.45,
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
    topics_of_interest: list[str] = Field(default_factory=list)
    """她在意的话题关键词。玩家**第一次**聊到其中任一个就涨好感。

    **为什么这里用子串匹配是可靠的。** 坑 #33、#36 的教训是「一组标记词能证明
    话题，不能证明命题」——而这里要的恰好**就是话题**：「玩家这句话提到了赛钱」
    是一个话题级判断，命中即事实。判据的名字和它实际测的东西一致，所以那两条
    坑的陷阱在这里不成立（对比：「出戏承认」是命题，所以子串匹配在那里是错的）。

    误报代价也低：多涨几点好感，而不是让她去指出一个不存在的矛盾。

    去重由 `NpcState.discussed_topics` 负责——不去重的话玩家会重复说同一个词
    刷好感，那只是把「刷数值」换了个形式，对话仍然没有价值。
    """


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
    refusal: str = ""
    """非空表示她在这个情绪模式里**根本不搭话**，而这个字符串就是玩家看到的
    那一行。

    做成一个字段而不是「布尔开关 + 文案」两个：两个字段可以互相矛盾（开关
    开着而文案是空的，或者反过来），而坑 #4 的教训正是「两处声明同一件事
    时，真正把守的往往不是你以为的那一处」。

    这是 prompt 层禁令失效之后的引擎侧杠杆。实测对抗人格下她 87.5% 的复读
    是「对语义不同的问题塌缩成同一句敷衍」，而那句话当时就列在禁语清单里
    ——8B 模型在这件事上不听指令（工程日志坑 #27）。被烦到不想说话是灵梦
    `irritated` 的人设本身（角色卡写的是「可能直接赶人」），把它变成机制
    比继续加一句提示可靠。
    """
    approaching: str = ""
    """快要进入这个模式时给玩家的预警，必须含 `{turns}` 占位符。

    从「懒散」直接跳到「转身走开」是断崖式的，玩家不知道自己踩到了什么。
    引擎里已经有同一套做法的先例——无缘塚的遗忘机制会提前说「再在这里待
    2 步，你会忘掉一件事」。可预告的惩罚才是机制，不可预告的是陷阱。

    倒计时由引擎按「她的情绪每回合净涨多少」算出来，所以它不是拍脑袋的
    文案，改角色卡的 `player_talked` 或 `decay_per_tick` 它会跟着变。
    """

    @field_validator("approaching")
    @classmethod
    def _must_have_countdown(cls, value: str) -> str:
        """预警文案漏掉占位符会渲染成一句没有数字的话，而它的全部价值就是
        那个数字。空串表示这个模式不预警，是合法的。"""
        if value and "{turns}" not in value:
            raise ValueError("approaching 必须含 {turns} 占位符，否则预警里没有倒计时")
        return value

    def contains(self, value: float, margin: float = 0.0) -> bool:
        """左闭右开，保证模式区间不重叠。最高档的上界特殊处理为闭区间。

        `margin` 把区间两端各外扩一点，用于迟滞：留在当前模式的判定比
        进入它更宽松，于是值在阈值上下抖动时模式不会跟着抖。
        """
        low, high = self.range
        low -= margin
        high += margin
        if self.range[1] >= 1.0:
            return low <= value
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
    aliases: list[str] = Field(default_factory=list)
    """她和玩家实际会怎么说这件东西。**一处定义两处使用**：`resolve_item`
    认它（玩家打 `/give 钱` 很自然），记忆探针也认它——实测她答的是
    「你给的钱呢？」而不是「赛钱」，只认全名的话事实召回率恒为 0，
    而那不是「她记不住」，是尺子看不见。

    别名必须在整张物品表里唯一，有测试锁住。"""
    description: str = ""

    def surfaces(self) -> frozenset[str]:
        return frozenset({self.name, *self.aliases})


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
