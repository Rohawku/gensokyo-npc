"""记忆指标：真值全部来自轨迹本身，零人工标注。

三类探针（`MemoryProbePlayer` 发问）：

1. **事实召回** —— 「我给过你什么东西？」比对轨迹里真实发生过的赠予
2. **幻觉** —— 她说出一件从未给过的东西
3. **负例否认** —— 问一件从未发生的事，她该否认或表示不知道

**第三类必须有。** 只测召回会奖励「什么都说记得」的模型——那正是坑 #19 证明
过的退化方向（她塌缩成两三句固定应答，指标反而好看）。precision 与 recall
要一起看。

真值取自 `player_input`：赠予是 `/give <中文名>` 指令，成功与否记在
`command_ok` 上。**不去读 event_log 或记忆库**——那样就变成「用系统自己的
状态验证系统」，而这三个指标要回答的是「玩家问她，她答得对不对」。
"""

from collections.abc import Sequence

from pydantic import BaseModel

from gensokyo.session.commands import ALIASES
from gensokyo.testkit.metrics.safety import hits
from gensokyo.testkit.personas import FLIP_LINES, PROBE_BY_QUESTION
from gensokyo.testkit.trajectory import Trajectory
from gensokyo.world.defs import WorldDefs

GIVE_HEADS: frozenset[str] = frozenset(k for k, v in ALIASES.items() if v == "give")
"""哪些指令头算「玩家交出了东西」，**直接从别名表推导**。

第一版这里另写了一份 `{"give", "送", "给"}`——「送」「给」不在别名表里
（那两个是死条目），而真实存在的 `pay` 反倒漏了。一处定义两处使用，
和助手腔词库取自角色卡 `forbidden_phrases` 是同一个理由。
"""

DENIAL_WORDS: tuple[str, ...] = (
    "没有",
    "没给",
    "不记得",
    "没印象",
    "想不起",
    "记不",
    "什么时候",
    "哪来的",
    "没见过",
    "没收到",
    "别瞎说",
    "胡说",
    "编",
    "才没",
    "没这回事",
    "不知道",
)
"""否认标记词。**这一项是近似指标**，报告里按近似标注。

硬化它需要让 NPC 输出结构化的「我不记得」信号，而那会把元层概念塞进台词
（坑 #10 清了四轮同类问题）。所以这里接受关键词近似，代价写明。
"""


FLAG_WORDS: tuple[str, ...] = (
    "刚才",
    "刚说",
    "刚刚",
    "不是说",
    "你不是",
    "怎么又",
    "又变了",
    "前后",
    "矛盾",
    "撒谎",
    "说谎",
    "骗",
    "改口",
    "自相",
    "到底哪句",
    "哪句是真",
)
"""指出矛盾的标记词。**近似指标**，报告里按近似标注。

**这一项的「正确答案」因角色而异**，所以比率必须按角色看：灵梦该直接怼；
芙兰大概率根本没记住前一句，而没记住在她身上不算失败——她的 λ 是全场
最大的，忘得快是设定而不是缺陷。一个跨角色聚合的矛盾检出率会把「符合人设
的遗忘」和「被绕晕了」加在一起，正是坑 #18 那个错误。
"""


class MemoryMetrics(BaseModel):
    recall_probes: int
    probe_episodes: int
    """跑出过探针的对局数。**这才是探针类比率的有效分母。**

    同一局里的探针结果高度相关——她整局要么认真回答、要么整局敷衍，实测
    每局召回率在 0.00 和 0.76 之间。16 次探针不是 16 个独立样本，是 1 个
    样本重复观测 16 次。按 166 报分母等于把置信区间凭空缩小 4 倍
    （工程日志坑 #25）。"""
    fact_recall_rate: float
    """召回探针里，她说出了**至少一件真实给过**的东西的比例。硬指标。"""
    fact_hallucination_rate: float
    """召回探针里，她说出了**从未给过**的东西的比例。硬指标，与召回率一起看。"""
    negative_probes: int
    false_affirmation_rate: float
    """负例探针里，她**没有**否认那件从未发生的事的比例。近似指标。"""
    contradiction_probes: int
    contradiction_episodes: int
    contradiction_flag_rate: dict[str, float]
    """npc_id -> 她指出玩家改口的比例。**按角色拆开**，见 `FLAG_WORDS`。"""
    recalled_per_turn: float
    """平均每个 NPC 回合召回到的记忆条目数。不是质量指标，是「检索通路
    有没有在工作」的体检项——零召回说明记忆层根本没接上。"""
    zero_recall_turns: int


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _surface_to_item(defs: WorldDefs) -> dict[str, str]:
    """她可能说出口的字面 → 物品 id。

    只认全名的话事实召回率恒为 0：实测 166 次召回探针里她一次都没说出
    「赛钱」三个字，说的是「你给的钱呢？」。那是尺子看不见，不是她记不住。
    """
    return {
        surface: str(item_id) for item_id, item in defs.items.items() for surface in item.surfaces()
    }


def _mentioned_items(text: str, surfaces: dict[str, str]) -> set[str]:
    return {item for surface, item in surfaces.items() if surface in text}


def _given_names(traj: Trajectory, defs: WorldDefs) -> set[str]:
    """这一局里玩家**真的**成功交出去过的物品 id。

    只认 `command_ok is True`：失败的 `/give`（身上没有那么多）不构成真值，
    而按 `player_input` 一律计入会让「她记得一件没送成的东西」被算成正确召回。
    """
    surfaces = _surface_to_item(defs)
    given: set[str] = set()
    for turn in traj.turns:
        if turn.kind != "command" or not turn.command_ok:
            continue
        head, _, arg = turn.player_input.partition(" ")
        if head.lstrip("/") not in GIVE_HEADS:
            continue
        item = surfaces.get(arg.strip())
        if item is not None:
            given.add(item)
    return given


def memory_metrics(trajectories: Sequence[Trajectory], defs: WorldDefs) -> MemoryMetrics:
    surfaces = _surface_to_item(defs)

    recall_probes = 0
    probe_episodes = 0
    recall_hit = 0
    recall_halluc = 0
    negative_probes = 0
    false_affirm = 0
    flips: dict[str, int] = {}
    flagged: dict[str, int] = {}
    contradiction_episodes = 0
    recalled_total = 0
    npc_turns = 0
    zero_recall = 0

    for traj in trajectories:
        given = _given_names(traj, defs)
        if any(turn.player_input in PROBE_BY_QUESTION for turn in traj.turns):
            probe_episodes += 1
        if any(turn.player_input in FLIP_LINES for turn in traj.turns):
            contradiction_episodes += 1
        for turn in traj.turns:
            if turn.npc_id is not None:
                npc_turns += 1
                recalled_total += len(turn.retrieved_memory_ids)
                if not turn.retrieved_memory_ids:
                    zero_recall += 1

            if turn.player_input in FLIP_LINES and turn.npc_id and turn.utterance:
                flips[turn.npc_id] = flips.get(turn.npc_id, 0) + 1
                if hits(turn.utterance, FLAG_WORDS):
                    flagged[turn.npc_id] = flagged.get(turn.npc_id, 0) + 1

            probe = PROBE_BY_QUESTION.get(turn.player_input)
            if probe is None or turn.npc_id is None or not turn.utterance:
                continue

            if probe.kind == "recall":
                recall_probes += 1
                said = _mentioned_items(turn.utterance, surfaces)
                if said & given:
                    recall_hit += 1
                if said - given:
                    recall_halluc += 1
            else:
                negative_probes += 1
                # 她提到了那个从未给过的东西，且没有任何否认标记 —— 顺着编了。
                mentioned = probe.subject_item in _mentioned_items(turn.utterance, surfaces)
                denied = bool(hits(turn.utterance, DENIAL_WORDS))
                if mentioned and not denied:
                    false_affirm += 1

    return MemoryMetrics(
        recall_probes=recall_probes,
        probe_episodes=probe_episodes,
        fact_recall_rate=_rate(recall_hit, recall_probes),
        fact_hallucination_rate=_rate(recall_halluc, recall_probes),
        negative_probes=negative_probes,
        false_affirmation_rate=_rate(false_affirm, negative_probes),
        contradiction_probes=sum(flips.values()),
        contradiction_episodes=contradiction_episodes,
        contradiction_flag_rate={
            npc: _rate(flagged.get(npc, 0), n) for npc, n in sorted(flips.items())
        },
        recalled_per_turn=_rate(recalled_total, npc_turns),
        zero_recall_turns=zero_recall,
    )
