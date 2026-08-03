"""锚点探针：固定状态 + 单次提问 + 独立重复采样。

**这个模块存在的唯一理由是分辨率。** 整局评测的探针在同一局里问 16 次，而她
整局要么认真回答要么整局敷衍——16 次不是 16 个独立样本（坑 #25）。实测局间
σ = 0.250，每侧 10 局只能检出 **31%** 的变化，而所有值得做的干预都在 10 个
百分点级（坑 #28）。

锚点探针把「一局 40 回合问 16 次」换成「一个全新世界问 1 次，重复 N 遍」：

| | 整局探针 | 锚点探针 |
|---|---|---|
| 一个样本花多少次模型调用 | 约 5（摊到每次提问） | 1 |
| 样本之间独立吗 | 不（局内相关） | 是（每次全新世界与记忆） |
| 同样分辨率要多久 | 7.3 小时 | 约 10 分钟 |

**样本独立性来自「每个样本一个全新世界」**，而这件事做得到全靠世界与记忆两层
都能从动作日志精确重建（取舍 #2、#7）——没有那个性质，「同一个固定状态」
每次摆出来都会不一样。

只调说话阶段、不调决策阶段：被测的是**她说出口的话**，而决策阶段那次调用
既翻倍成本又引入一层无关的方差（她这次决定调不调工具）。所以 `thought`
按锚点写死。
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from gensokyo.agent.prompt import build_speak_messages
from gensokyo.llm.client import LlmClient
from gensokyo.memory.pipeline import absorb, new_stores, now_seq
from gensokyo.memory.query import MEMORY_TOP_K, build_focus, build_query
from gensokyo.memory.render import render_recall
from gensokyo.memory.retrieve import retrieve
from gensokyo.world.defs import WorldDefs
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action

REPO_ROOT = Path(__file__).resolve().parents[3]

SAMPLE_TEMPERATURE = 0.8
"""和真实对局同一个温度。

这里刻意**不**调高：锚点探针要回答的是「真实玩法下她会怎么答」，温度一改，
量出来的就是另一个分布了。造偏好数据是另一回事，那边要的是分布两端
（`training/harvest.py` 用 1.0）。
"""


@dataclass(frozen=True)
class Anchor:
    """一个固定探针场景。

    `setup` 只含**玩家动作**，不含任何模型调用——摆状态不该花钱，也不该引入
    方差。`thought` 写死是同一个理由：决策阶段那次调用会让「她这次想不想调
    工具」混进来。
    """

    id: str
    npc_id: str
    question: str
    setup: tuple[Action, ...] = ()
    thought: str = "……"
    history: tuple[str, ...] = ()
    already_said: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    """假装决策阶段刚做过这些事，结果如实喂给说话阶段。

    没有它，锚点只能测「她主动说什么」，测不到**线索揭示**这条路——
    `reveal_info` 发生在决策阶段，情报内容是靠工具结果进入台词的。第一版漏了
    这个字段，于是 `dormant_awake` 那个锚点的 note 声称自己在测「记忆系统端到端
    可见的通路」，实际只测到了「她会不会主动提起」。
    """
    note: str = ""


class Sample(BaseModel):
    anchor_id: str
    npc_id: str
    question: str
    utterance: str


class Rate(BaseModel):
    """带置信区间的比率。

    **锚点探针的产出必须带区间。** 裸比率是这个项目栽过最多次的地方——
    坑 #18、#25、#26、#28 都是「一个看起来精确的比率，实际分辨不出任何东西」。
    样本独立之后区间才算得出来，而这正是换掉整局探针的目的。
    """

    hits: int
    total: int

    @property
    def rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def half_width(self) -> float:
        """Wald 半宽（95%）。样本独立才成立，所以只在锚点探针上用。"""
        if self.total == 0:
            return 0.0
        p = self.rate
        return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / self.total)

    def __str__(self) -> str:
        return f"{self.rate:.1%} ± {self.half_width:.1%}（n={self.total}）"

    def separated_from(self, other: "Rate") -> bool:
        """两个比率的区间是否不重叠——**能不能下「有差异」这个结论**。

        不重叠不等于统计显著，但重叠一定不显著。做成这个方向是因为这个项目
        真正需要防的是「报出一个假的改进」（坑 #26），而不是漏掉一个真的。
        """
        return abs(self.rate - other.rate) > self.half_width + other.half_width


@dataclass
class AnchorRun:
    samples: list[Sample] = field(default_factory=list)

    def of(self, anchor_id: str) -> list[Sample]:
        return [s for s in self.samples if s.anchor_id == anchor_id]


def _load(defs: WorldDefs | None = None) -> WorldDefs:
    return defs or load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")


def stage(anchor: Anchor, defs: WorldDefs) -> tuple[WorldEngine, list[str]]:
    """摆出锚点声明的那个状态，返回引擎和渲染好的召回。

    **全程只用动作。** 直接给 state 赋值不进动作日志，于是记忆库重建不出来
    （坑 #9 那个陷阱，这个项目里踩过两次）——而锚点的价值恰恰在于「每次摆出来
    的都是同一个状态，且它是可回放的」。
    """
    engine = WorldEngine(build_initial_state(defs), defs)
    stores = new_stores(defs)
    for action in anchor.setup:
        result = engine.apply(action)
        if not result.ok:
            raise ValueError(f"锚点 {anchor.id} 的 setup 失败：{action.tool} —— {result.error}")
        engine.tick()
        absorb(engine, stores)

    npc_id = NpcId(anchor.npc_id)
    card = defs.characters[npc_id]
    obs = engine.observe(npc_id)
    scored = retrieve(
        stores[npc_id],
        build_query(obs, anchor.question),
        card,
        now_seq(engine),
        build_focus(obs),
        k=MEMORY_TOP_K,
    )
    return engine, render_recall(scored)


def ask(anchor: Anchor, llm: LlmClient, defs: WorldDefs | None = None) -> Sample:
    """摆好状态、问一句、取她的回答。一次模型调用。

    每次调用都重新摆状态：这就是样本独立的来源，也是它比整局探针贵不了多少
    的原因——摆状态只用引擎动作，不花模型调用。
    """
    world = _load(defs)
    engine, recalled = stage(anchor, world)
    npc_id = NpcId(anchor.npc_id)
    card = world.characters[npc_id]

    engine.apply(Action(actor="player", tool="say", args={"text": anchor.question}))
    messages = build_speak_messages(
        card,
        engine.observe(npc_id),
        [*anchor.history, f"玩家：{anchor.question}"],
        anchor.thought,
        list(anchor.outcomes),
        list(anchor.already_said),
        recalled,
    )
    try:
        text = llm.complete(messages, temperature=SAMPLE_TEMPERATURE).strip()
    except Exception:  # noqa: BLE001
        # 批量采集里单次失败不该让整轮白跑。空回答会被判据算成缺陷，
        # 而那会把端点抽风记成模型变差——所以显式丢弃。
        text = ""
    return Sample(
        anchor_id=anchor.id, npc_id=anchor.npc_id, question=anchor.question, utterance=text
    )


def run(
    anchors: Sequence[Anchor],
    llm: LlmClient,
    repeats: int,
    defs: WorldDefs | None = None,
) -> AnchorRun:
    """每个锚点独立采样 `repeats` 次。丢掉空回答（端点抽风不算数据）。"""
    world = _load(defs)
    out = AnchorRun()
    for anchor in anchors:
        for _ in range(repeats):
            sample = ask(anchor, llm, world)
            if sample.utterance:
                out.samples.append(sample)
    return out
