from enum import StrEnum

from pydantic import BaseModel, Field

from gensokyo.world.ids import EventId, NpcId


class Tier(StrEnum):
    """遗忘做成降级而不是删除。

    删除会让「她忘了」和「这件事没发生过」在数据上无法区分，于是负例探针
    （问从未发生的事）测不出任何东西。降级保留了源事件指针，所以永远能
    回答「这件事发生过，但她想不起来了」。
    """

    ACTIVE = "active"
    """可检索到原文。"""
    COMPRESSED = "compressed"
    """同类合并成一条摘要，细节丢失。即「记得大概但记不清具体」。"""
    DORMANT = "dormant"
    """不进常规检索，只能被强线索召回。芙兰那条线索的机制基础。"""


class MemoryItem(BaseModel):
    id: str
    npc_id: NpcId
    seq: int
    """事件序号，衰减的时间轴。**刻意不是 tick。**

    `replay` 不重放 `tick()`（取舍 #2），所以回放后 `Event.tick` 全是 0——
    时间衰减若读 tick，读档会让所有记忆的新鲜度归零，而且无声。事件 id
    由 `_emit` 顺序产生、回放逐字复现，已有测试锁住。

    坑 #9 立的规矩：新机制先问「它是动作的结果，还是时间的结果」。
    """
    content: str
    source_event_id: EventId | None
    """指回 event_log，可验证可追溯。

    这个字段把**「忘了」和「记错了」拆成两种可分别归因的失败**：条目衰减出
    检索范围是遗忘（调检索），条目还在但 content 与源事件不符是幻觉（调写入
    与生成约束）。两者修法完全不同，混在一个指标里就都没法修。

    `None` 表示这条记忆**没有源事件**——她入场前就有的往事（芙兰 495 年前
    那段）。这类条目必须排除在记忆评估之外：模型答得像芙兰可能只是因为
    预训练见过东方，不是因为记忆系统起作用。混进去会得出「加记忆没提升」
    的错误结论。判据就是这个字段是不是 None，不需要另设标记。
    """
    kind: str
    salience: float
    tier: Tier = Tier.ACTIVE
    trigger_keys: tuple[str, ...] = ()
    """沉睡记忆的强线索：命中其中任一个就召回。

    只有 `DORMANT` 条目用得上。匹配是精确的键匹配而不是相似度——这条路径
    直接决定芙兰那条线索能不能拿到，而可通关性不该依赖一个相似度阈值
    （坑 #6 是「游戏做出来不可通关」）。
    """
    access_count: int = 0
    last_access_seq: int = 0


class MemoryStore(BaseModel):
    """单个 NPC 的记忆库。每个 NPC 一份，互不可见。

    互不可见是设计而非疏漏：三个 NPC 各自只知道自己经历过的事，玩家因此
    可以对不同人说不同的话。世界层的近期事件能把她们串起来（同场时看得见
    对方做了什么），记忆层不能。
    """

    npc_id: NpcId
    items: list[MemoryItem] = Field(default_factory=list)
    ingested_events: set[EventId] = Field(default_factory=set)
    """已摄入的事件 id。重复摄入同一事件会让同一件事在记忆库里出现两遍，
    而检索按分数排序——两条一模一样的条目会挤掉本该被召回的第二名。"""

    def active(self) -> list[MemoryItem]:
        return [i for i in self.items if i.tier is not Tier.DORMANT]

    def dormant(self) -> list[MemoryItem]:
        return [i for i in self.items if i.tier is Tier.DORMANT]
