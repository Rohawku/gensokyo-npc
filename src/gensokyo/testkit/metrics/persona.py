"""角色一致性指标。

四项里三项是硬指标：助手腔用角色卡自己的禁语清单，行为偏离用 JS 散度对
角色卡的 `behavior_baseline`，复读率是标准化后的字符串相等。只有越界知识一项
是关键词近似，且它用的**不是**角色卡数据——原因写在 `OUT_OF_BOUNDS_WORDS` 上。
"""

import math
from collections.abc import Iterable, Sequence

from pydantic import BaseModel

from gensokyo.agent.schema import normalize_utterance
from gensokyo.testkit.metrics.safety import hits
from gensokyo.testkit.trajectory import Trajectory
from gensokyo.world.defs import WorldDefs
from gensokyo.world.ids import NpcId

BASELINE_KEY = "tool_frequency"

PLOT_TOOLS = frozenset({"reveal_info", "travel_to", "use_spellcard"})
"""剧情主线工具，不计入行为偏离度。

这个指标衡量的是**性格表达**——她爱不爱拿东西、爱不爱问话。而剧情工具的
调用由任务状态驱动、与性格无关：线索该给的时候谁都得 reveal_info。把它们
算进去会给散度一个与人设无关的结构性下限，实测让芙兰虚高到 1.000、魔理沙
0.884，把「走了剧情」误读成「人设崩了」。
"""

OUT_OF_BOUNDS_WORDS: tuple[str, ...] = (
    "手机",
    "电脑",
    "计算机",
    "互联网",
    "网络",
    "电视",
    "飞机",
    "汽车",
    "火车",
    "电话",
    "照相机",
    "冰箱",
    "空调",
    "卫星",
    "火箭",
    "软件",
    "数据库",
    "wifi",
)
"""越界知识（现代科技）的独立词表。

**这里和角色卡不一致，得说清楚。** 角色卡的 `knowledge.forbidden_knowledge`
现在是中文散文（「外界的科技」「地下室外面发生的事」），那是给模型看的提示
文本，拿它做子串匹配等于问「她有没有把『外界的科技』这五个字念出来」——
永远命中不了，又是一个空转的词表（工程日志坑 #10、#11）。

所以本项目在这里**没有复用角色卡数据**，用的是一份独立的现代科技词表。
代价是词表和角色卡可能漂移：改了 `forbidden_knowledge` 不会自动改这里。这是
一处已知的、刻意的不一致，不是「用了角色卡数据」——报告里也按近似指标标注。
（另一条硬化路径是给角色卡加机器可读的越界词字段，但那要动数据契约。
`blind_to_outside` 那个开关已经是这条路走通的先例。）
"""


class PersonaMetrics(BaseModel):
    utterances: int
    assistant_tone_rate: float
    assistant_tone_hits: dict[str, int]
    behavior_divergence: dict[str, float]
    """npc_id -> 与角色卡基线的 JS 散度，取值 [0,1]。没有基线、或整批一次
    工具都没调的角色**不出现在这里**——见 `behavior_divergence` 的计算注释。"""
    behavior_observed: dict[str, dict[str, float]]
    out_of_bounds_rate: float
    repetition_rate: float


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _without_plot_tools(weights: dict[str, float]) -> dict[str, float]:
    """期望与实际两侧都要排除剧情工具。只排一侧会凭空造出散度——
    魔理沙的基线里本来就列了 use_spellcard。"""
    return {k: v for k, v in weights.items() if k not in PLOT_TOOLS}


def _normalized(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in weights.items()}


def js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    """Jensen-Shannon 散度，底数 2，取值落在 [0,1]。

    自己实现而不引入 scipy：这是评测里唯一需要的数值函数，为它多一个
    重依赖不划算，而且自己写的版本能把「两侧工具集合不同」这件事按本项目
    的语义处理——先取并集补零再各自归一化。补零是必须的：基线里有
    `break_item` 而实际一次没调，那正是最该被算成偏离的情况，把它当成
    「这一维不存在」会把偏离抹掉。

    完全相同的分布返回 0，完全不重叠返回 1。两侧都为空返回 0（没有基线也
    没有行为，谈不上偏离）；只有一侧为空返回 1。
    """
    p = _normalized(left)
    q = _normalized(right)
    if not p and not q:
        return 0.0
    if not p or not q:
        return 1.0

    total = 0.0
    for key in p.keys() | q.keys():
        pi = p.get(key, 0.0)
        qi = q.get(key, 0.0)
        mi = (pi + qi) / 2
        if pi > 0:
            total += 0.5 * pi * math.log2(pi / mi)
        if qi > 0:
            total += 0.5 * qi * math.log2(qi / mi)
    # 浮点误差可能让完全不重叠的两个分布算出 1.0000000000000002。
    return min(1.0, max(0.0, total))


def _forbidden_phrases(defs: WorldDefs, npc_id: str) -> tuple[str, ...]:
    """助手腔词库**直接取角色卡的 `persona.speech.forbidden_phrases`**。

    一处定义两处使用：同一份清单既进系统提示约束生成，又在这里用于检测。
    这是刻意的——检测词库若和生成约束各写一份，两份很快就会漂移，于是
    「污染率下降」可能只是因为检测的那份忘了跟着更新。
    """
    card = defs.characters.get(NpcId(npc_id))
    if card is None:
        return ()
    return tuple(card.persona.speech.forbidden_phrases)


def _observed_tool_counts(trajectories: Sequence[Trajectory]) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, float]] = {}
    for traj in trajectories:
        for turn in traj.turns:
            if turn.npc_id is None:
                continue
            per_npc = counts.setdefault(turn.npc_id, {})
            for callinfo in turn.tool_calls:
                tool = str(callinfo.get("tool", ""))
                per_npc[tool] = per_npc.get(tool, 0.0) + 1.0
    return counts


def _repetitions(trajectories: Sequence[Trajectory]) -> tuple[int, int]:
    """(重复的台词数, 台词总数)。

    「重复」= 同一 NPC 在**同一局内**说出过同一句话，第二次起计数。比较走
    `normalize_utterance`，只差标点或空格的两句算同一句——第一份基线里
    「你到底想干啥？」19 次和「你到底想干啥。」12 次被算成两句不同的话，
    于是测出来的复读率**低于**真实值。这条口径改动让 43.1% / 56.7% 那组
    数字不可与之后的数字直接比较。

    跨局重复不算：两局之间没有共享历史，说同一句话是采样巧合而不是复读。

    工程日志坑 #2 实测过这件事的严重性：她说过一次「你管的太多了」，这句话
    进了对话历史，模型看到这个模式就继续敷衍，`reveal_info` 命中率从 3/5
    掉到 1/5。**复读是自我强化的，而且它同时压低工具调用率**——不只是台词
    单调，是玩法直接卡死。这个数是那件事的量化。
    """
    repeats = 0
    total = 0
    for traj in trajectories:
        seen: set[tuple[str, str]] = set()
        for turn in traj.turns:
            if turn.npc_id is None or not turn.utterance:
                continue
            total += 1
            key = (turn.npc_id, normalize_utterance(turn.utterance))
            if key in seen:
                repeats += 1
            seen.add(key)
    return repeats, total


def _count_hits(text: str, words: Iterable[str], into: dict[str, int]) -> bool:
    found = hits(text, words)
    for word in found:
        into[word] = into.get(word, 0) + 1
    return bool(found)


def persona_metrics(trajectories: Sequence[Trajectory], defs: WorldDefs) -> PersonaMetrics:
    utterances = 0
    assistant_tone = 0
    out_of_bounds = 0
    tone_hits: dict[str, int] = {}

    for traj in trajectories:
        for turn in traj.turns:
            if turn.npc_id is None or not turn.utterance:
                continue
            utterances += 1
            if _count_hits(turn.utterance, _forbidden_phrases(defs, turn.npc_id), tone_hits):
                assistant_tone += 1
            if hits(turn.utterance, OUT_OF_BOUNDS_WORDS):
                out_of_bounds += 1

    observed_counts = _observed_tool_counts(trajectories)
    observed = {
        npc: _normalized(_without_plot_tools(counts)) for npc, counts in observed_counts.items()
    }

    divergence: dict[str, float] = {}
    for npc_id, actual in observed.items():
        card = defs.characters.get(NpcId(npc_id))
        expected = (
            _without_plot_tools(dict(card.behavior_baseline.get(BASELINE_KEY, {}))) if card else {}
        )
        if not expected or not actual:
            # 没有基线就没有「期望」可言；一次工具都没调则是没有数据。两种
            # 情况都不给数字，因为 0.0 会被读成「完全符合基线」，而 1.0 会被
            # 读成「行为完全跑偏」——两个都是在没有依据时下结论。报告里这一
            # 格印「—」并注明原因。
            continue
        divergence[npc_id] = js_divergence(expected, actual)

    repeats, repeat_total = _repetitions(trajectories)

    return PersonaMetrics(
        utterances=utterances,
        assistant_tone_rate=_rate(assistant_tone, utterances),
        assistant_tone_hits=tone_hits,
        behavior_divergence=divergence,
        behavior_observed=observed,
        out_of_bounds_rate=_rate(out_of_bounds, utterances),
        repetition_rate=_rate(repeats, repeat_total),
    )


def expected_baselines(defs: WorldDefs) -> dict[str, dict[str, float]]:
    """各角色卡的期望工具分布，归一化。报告要把期望和实际并排印出来——
    只给一个散度值的话，读的人无法判断偏在哪个工具上。"""
    return {
        str(npc_id): _normalized(
            _without_plot_tools(dict(card.behavior_baseline.get(BASELINE_KEY, {})))
        )
        for npc_id, card in defs.characters.items()
    }


def persona_library_sizes() -> dict[str, int]:
    return {
        "out_of_bounds": len(OUT_OF_BOUNDS_WORDS),
        "plot_tools_excluded": len(PLOT_TOOLS),
    }
