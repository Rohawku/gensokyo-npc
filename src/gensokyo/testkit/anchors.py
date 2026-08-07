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
from typing import cast

from pydantic import BaseModel

from gensokyo.agent.prompt import build_speak_messages
from gensokyo.agent.schema import normalize_utterance
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

Z_95 = 1.96
"""95% 双侧正态分位点。"""

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
    revealed: str = ""
    """这一回合刚成功揭示的情报正文，喂给说话阶段那条「必须把它说出来」的指令。

    **和 `outcomes` 分开声明，因为它驱动的是一条硬性指令，不能靠从 outcomes 里
    认前缀。** 第一版忘了在 `ask()` 里传它，于是「干预后」那次测量实际测的是干预前
    ——13.3%，而我据此差点写下「这个干预在锚点上无效」。坑 #28 的形态，
    这次骗到的是结论本身。
    """
    variant_of: str = ""
    """这个锚点是另一个锚点的变体，判据完全复用那一个的。

    存在的理由是坑 #30 那条教训的推广：**一个比率可能是提问条件的性质，而不是
    她的性质。** 整局探针那次是问句自带答案，而在锚点上的风险是「换个说法、
    或者换个状态，数字就变了」。

    判据一律只在本体上声明——给变体也写一份就是重复定义，而重复定义会漂移。
    """
    varies: str = "phrasing"
    """这个变体改的是**哪一维**：`phrasing`（问法）/ `attitude`（好感档位）/
    `emotion`（情绪档位）。

    **必须分开，因为「区间不重叠」在两类变体上的含义正好相反：**

    - 换问法不重叠是**警告**——同一个状态问两种说法得到两个数，那个数就是问法
      的性质，不能单独引用（坑 #30）。
    - 换状态不重叠是**发现**——她在好感 0 和好感 18 下表现不同，这本来就该不同；
      真正要警觉的是「引用那个数字时没说它属于哪个状态」。反过来，重叠说明这个
      数字对状态不敏感，那才是可以裸引用的。

    混成一节报的话，第二类会被读成第一类，于是每一条真实的状态依赖都会被
    标成「这个指标不可靠」。
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

    区间用 **Wilson 得分区间**，不是 Wald。第一版用 Wald，于是 40 个样本全中
    被印成「100.0% ± 0.0%」——Wald 半宽在 p=0 和 p=1 处恒等于 0（坑 #31）。
    那是这份日志花了四条坑警告的同一种假精确，而且它会让 `separated_from`
    开始报假的改进。Wilson 在边界上给出的是不对称但非零的区间。
    """

    hits: int
    total: int

    @property
    def rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def bounds(self) -> tuple[float, float]:
        """Wilson 得分区间（95%）。样本独立才成立，所以只在锚点探针上用。

        0 个样本时返回 `(0.0, 1.0)`——没有信息。返回一个零宽区间才是假的，
        它会让 `separated_from` 在完全没有数据的情况下开始下结论。
        """
        if self.total == 0:
            return 0.0, 1.0
        z2 = Z_95**2
        denom = 1.0 + z2 / self.total
        center = (self.rate + z2 / (2 * self.total)) / denom
        spread = (
            Z_95
            / denom
            * math.sqrt(self.rate * (1 - self.rate) / self.total + z2 / (4 * self.total**2))
        )
        return max(0.0, center - spread), min(1.0, center + spread)

    @property
    def lower(self) -> float:
        return self.bounds[0]

    @property
    def upper(self) -> float:
        return self.bounds[1]

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def __str__(self) -> str:
        low, high = self.bounds
        return f"{self.rate:.1%}（95% CI {low:.1%}–{high:.1%}，n={self.total}）"

    def separated_from(self, other: "Rate") -> bool:
        """两个比率的区间是否不重叠——**能不能下「有差异」这个结论**。

        不重叠不等于统计显著，但重叠一定不显著。做成这个方向是因为这个项目
        真正需要防的是「报出一个假的改进」（坑 #26），而不是漏掉一个真的。
        """
        return self.lower > other.upper or other.lower > self.upper


@dataclass
class AnchorRun:
    samples: list[Sample] = field(default_factory=list)

    def of(self, anchor_id: str) -> list[Sample]:
        return [s for s in self.samples if s.anchor_id == anchor_id]


def _load(defs: WorldDefs | None = None) -> WorldDefs:
    return defs or load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")


class Confounded(AssertionError):
    """两臂差了不止一个变量。消融的两臂只许差一处，否则区间分离也归不了因。"""


def assert_one_variable_apart(
    left: Anchor,
    right: Anchor,
    defs: WorldDefs,
    *,
    expect: str | set[str],
) -> None:
    """断言两个锚点在被观察到的状态上**只差 `expect` 声明的那些维度**。

    维度：`attitude` / `emotion` / `mode` / `recall` / `question` / `history`。

    **`expect` 允许多个维度，是因为这个引擎里有些变量无法独立操纵**——把它写成
    一个集合是「显式承认这处混淆」，而不是「放宽检查」：情绪只能靠送礼或说话推，
    两者都会改写召回块，所以「只改情绪」的锚点变体做不出来（同理：好感只能靠
    送礼和话题涨，所以高好感必然伴随相应的记忆条目）。声明成
    `expect={"emotion", "recall"}` 之后，**哪天有了别的推情绪的手段，这条断言
    会因为「差得比声明的少」而红**，提醒你去收紧。

    **这个函数是为一次真实的归因失败补的。** 我写了个一次性消融脚本比较
    「她记忆里有那段往事」和「没有」，测出 100.0% 对 33.3%、区间完全分离，
    据此做了修法——而修法零效果（坑 #54）。回头看那两臂：「有音乐盒」那臂是
    `calm`，「无音乐盒」那臂被垫话推成了 `destructive`。**两个变量混在一起了。**

    锚点变体那条纪律（坑 #49：「新的 = 旧的 + 增量」）有测试守着，而随手写的
    消融脚本没有——于是同一个错在没有测试的地方又犯了一次。所以把它做成一个
    **可调用的断言**：消融脚本自己 assert 一次，就不用靠人记得。
    """
    dims: dict[str, object] = {}
    for name, anchor in (("left", left), ("right", right)):
        engine, recalled = stage(anchor, defs)
        npc = engine.state.npcs[NpcId(anchor.npc_id)]
        dims[name] = {
            "attitude": npc.attitude,
            "emotion": round(npc.emotion, 2),
            "mode": npc.mode,
            "recall": tuple(recalled),
            "question": anchor.question,
            "history": tuple(anchor.history),
        }

    a = cast(dict[str, object], dims["left"])
    b = cast(dict[str, object], dims["right"])
    differ = sorted(k for k in a if a[k] != b[k])
    # 情绪变了必然带着模式变，反过来不成立——把它们算成同一处差别。
    if {"emotion", "mode"} <= set(differ):
        differ = [k for k in differ if k != "mode"]
    wanted = {expect} if isinstance(expect, str) else set(expect)
    if set(differ) != wanted:
        raise Confounded(f"两臂本该只差 {sorted(wanted)}，实际差了 {differ}；左 {a}，右 {b}")


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
        anchor.question,
        anchor.revealed,
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


def collapse_pairs(samples: Sequence[Sample]) -> dict[tuple[str, str], Rate]:
    """两个锚点之间「同一句话」的概率，按锚点对报出。

    **单锚点的判据看不见坑 #27。** 那一条实测 87.5% 的复读是「对不同问题塌缩成
    同一句」，而每个锚点各自只被问了一个问题——那一句在它自己的上下文里完全
    合理，任何单锚点判据都判不出问题。要看见它必须横着比。

    配对方式是**按序号对齐**：锚点 A 的第 k 个样本配 B 的第 k 个样本。两边都是
    独立抽样，所以每个比较就是「问两个不同问题，她给出同一句话」这个事件的一次
    独立观测，n 个比较是 n 个独立伯努利试验，区间成立。换成「这句有没有在
    别的锚点出现过」会让每个样本参与多次比较，指标之间互相依赖，区间随之失效
    ——这个项目在假分母上栽过四次（坑 #18、#25、#26、#28）。
    """
    by_key: dict[tuple[str, str], list[str]] = {}
    for sample in samples:
        by_key.setdefault((sample.npc_id, sample.anchor_id), []).append(
            normalize_utterance(sample.utterance)
        )

    out: dict[tuple[str, str], Rate] = {}
    keys = sorted(by_key)
    for i, (npc, left) in enumerate(keys):
        for other_npc, right in keys[i + 1 :]:
            if other_npc != npc:
                continue
            pairs = list(zip(by_key[npc, left], by_key[npc, right], strict=False))
            out[left, right] = Rate(hits=sum(a == b for a, b in pairs), total=len(pairs))
    return out
