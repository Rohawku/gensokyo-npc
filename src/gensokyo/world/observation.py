from pydantic import BaseModel, Field

from gensokyo.world.ids import FactId, LocationId, NpcId


class FactContext(BaseModel):
    fact_id: FactId
    content: str
    can_reveal_now: bool
    already_revealed: bool
    gate_hint: str


class Observation(BaseModel):
    """NPC 视角的世界快照。只包含她该知道的东西——
    信息可见性是世界规则，不能靠在 prompt 里叮嘱模型别乱说。"""

    tick: int
    npc_id: NpcId
    npc_name: str
    location_id: LocationId
    location_name: str
    location_description: str
    player_is_here: bool
    attitude: int
    emotion_var: str
    emotion: float
    mode: str
    mode_speech_hint: str
    own_inventory: dict[str, int] = Field(default_factory=dict)
    items_here: dict[str, int] = Field(default_factory=dict)
    """以中文物品名为键。Observation 只服务 prompt 组装，
    英文 id 泄漏进去就会被 NPC 说出口；面板数据走 PlayerView。"""
    received_from_player: list[str] = Field(default_factory=list)
    """来访者迄今给过她的东西，中文名，按名字排序。**这是引擎的原始记录，
    不是记忆。**

    实测她在被问「我给过你什么」时约三分之一的回答里会说出一件从没给过的
    东西（工程日志坑 #24、#25）。prompt 里已经写了「没列出来的就是你想不
    起来了——别编」，不够用。所以把引擎本来就知道的这份清单直接给她——
    延续坑 #2 的方法论：引擎已经算出来的结论，直告，别让小模型重新推导。

    **代价写在明面上**：这条一加，事实召回率就不再是「记忆层的指标」，
    而变成「她会不会用给她的东西」。记忆层剩下要衡量的是分层衰减与沉睡
    召回，见工程日志取舍 #11。
    """
    others_here: list[str] = Field(default_factory=list)
    facts: list[FactContext] = Field(default_factory=list)
    quest_hint: str | None = None
    """剧情进展的中文说法。为 None 表示该 NPC 被信息隔离，不知道外面的事。"""
    claim_check: str = ""
    """来访者刚才提到了一件她**从没收到过**的东西时，引擎给出的指令。

    这一条是「陈述事实不等于给指令」的直接产物。【来访者给过你的东西】那段
    已经写着「除这些之外他什么都没给过你」，而锚点探针实测她的否认率是
    **0.0% ± 0.0%**（n=40）、顺着编 67.5%——事实摆在 prompt 里，她照样顺着
    玩家说。坑 #2 早就给过答案：起作用的是【现在该做的事】那种**指令**，
    不是一段供她自己推导的状态描述。

    判定是纯字符串匹配（物品表的 `surfaces()` 对 `received_items`），确定性、
    可回放，不需要模型理解「他在骗我」。
    """
    suggestion: str = ""
    """引擎按当前状态算出的「现在该做什么」，只进决策阶段的 prompt。

    门槛开没开、线索齐没齐，引擎是知道的，不该指望一个小模型从情报
    清单里自己推出来——实测它只有 20~60% 的概率想到该调 reveal_info。"""


class NpcPanel(BaseModel):
    npc_id: NpcId
    name: str
    attitude: int
    emotion_var: str
    emotion: float
    mode: str
    mode_hint: str = ""
    will_talk: bool = False
    """她现在是否有线索可给。机器可读的门槛信号，供玩家模拟器与指标使用。"""
    mood_warning: str = ""
    """她快要不搭话时的预警，含倒计时。可预告的惩罚才是机制。"""
    refusal: str = ""
    """非空表示她这会儿不搭话，内容就是玩家看到的那一行。

    与 `will_talk` 是两件不同的事：那个说「她有情报可给」，这个说
    「她连话都不想跟你说」。"""


class PlayerView(BaseModel):
    """右栏面板数据。玩家必须能立刻看到自己行为的后果，
    否则记忆与情绪系统做得再好也感知不到。"""

    tick: int
    location_id: LocationId
    location_name: str
    location_description: str
    exits: list[str] = Field(default_factory=list)
    inventory: dict[str, int] = Field(default_factory=dict)
    items_here: dict[str, int] = Field(default_factory=dict)
    """以中文物品名为键。玩家屏幕上不该出现 offering_coin 这种 id。"""
    known_facts: list[str] = Field(default_factory=list)
    known_fact_ids: list[str] = Field(default_factory=list)
    """线索 id。按 NPC 统计线索产出率需要它——只有内容字符串的话，
    两个 NPC 同场时会把线索归错人。"""
    quest_stage: str = ""
    """阶段枚举名，供调试与测试。玩家可见的文本用 quest_hint。"""
    quest_hint: str = ""
    objective: str = ""
    """当前该干什么。玩家不该靠猜。"""
    oblivion_warning: str = ""
    """记忆开始流失的提示。空串表示暂时安全。"""
    ending_title: str = ""
    ending_text: str = ""
    npcs_here: list[NpcPanel] = Field(default_factory=list)
