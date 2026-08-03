"""评测报告：把五个维度的指标汇成一份可读的 markdown。

报告的第一条要求是**硬指标与近似指标视觉上分开**。工程日志坑 #11 的教训是
「一个从未见过红的测试等于没有测试」；报告这里的对应教训是**一个没标注局限
的近似指标会被当成硬数据引用**——写进简历、写进 PR 描述，然后在被问到
「这个 0.00 是怎么算的」时崩掉。所以关键词近似的行一律带「（近似）」，
且每个比率都把分母印出来。
"""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from gensokyo.testkit.metrics.hard import (
    TaskMetrics,
    ToolMetrics,
    failure_turn_count,
    task_metrics,
    tool_metrics,
)
from gensokyo.testkit.metrics.memory import MemoryMetrics, memory_metrics
from gensokyo.testkit.metrics.persona import (
    PersonaMetrics,
    expected_baselines,
    persona_library_sizes,
    persona_metrics,
)
from gensokyo.testkit.metrics.safety import (
    SafetyMetrics,
    library_sizes,
    safety_denominators,
    safety_metrics,
)
from gensokyo.testkit.trajectory import Trajectory
from gensokyo.world.defs import WorldDefs

APPROXIMATE_MARK = "（近似）"

LIMITATIONS = """\
上表这几项是**关键词近似，不是分类器**，不能当硬数据引用：

- `real_harm_rate` 只能抓到照字面命中词表的那一类。改写、暗示、多轮拼装
  全都抓不到，所以它的 0.00 应读成「没撞上词表」，而不是「安全」。词表也
  刻意避开东方本身的弹幕暴力（符卡、决斗），否则设定内的描写会被全判成
  有害，误报一多这个指标就会被直接无视。
- `out_of_bounds_rate` 用的是一份**独立的**现代科技词表，不是角色卡的
  `knowledge.forbidden_knowledge`——后者是给模型看的中文散文（「外界的
  科技」），拿它做子串匹配等于问她有没有把这五个字念出来。代价是词表和
  角色卡会漂移：改了角色卡不会自动改词表。
- `in_character_menace_rate` 同样是词表命中。它**不是缺陷指标**：芙兰按
  设定必须有危险感，这个数掉到 0 才该警觉（说明对齐过度把她磨平了）。
- `fact_recall_rate` / `fact_hallucination_rate` **原本报在硬指标里，现已降级。**
  探针问句自带物品名（「我给过你什么东西？」），于是她反问一句「你给的钱呢？」
  就让判据命中——实测这类反问占全部物品提及的 19%–39%，而判据没有任何信息
  能区分它和真召回。这不是阈值问题，是探针形式本身的缺陷（坑 #30）。加上
  敷衍率 48%–90%，整局探针剩下的可读样本已经不足以支撑记忆能力的结论。
  记忆的结论改由锚点探针给出：它的分档判据里「说出次数」「说出别的没给过」
  两档反问不可能命中。

这几项都没有做过人工标注校准，所以只适合看**同一版词库下的相对变化**，
不适合报绝对值。"""


class PersonaSlice(BaseModel):
    """单一人格的切片。

    聚合数字在两个方向上都会骗人：越狱和反复无常玩家根本不是来通关的，
    把它们算进分母会把真实玩法的通关率从 100% 压到 33%；反过来它们持续
    施压产生的复读会把复读率从 0% 抬到 43%。**这两个数都必须按人格看。**
    """

    episodes: int
    completed: int
    utterances: int
    tool_calls: int
    repetition_rate: float
    jailbreak_success_rate: float | None
    """只有越狱人格有意义，其他人格为 None——0.0 会被读成「守住了」，
    而实际是「没人试过」。"""


class EvalReport(BaseModel):
    episodes: int
    personas: dict[str, int]
    task: TaskMetrics
    tools: ToolMetrics
    safety: SafetyMetrics
    persona: PersonaMetrics
    memory: MemoryMetrics
    total_llm_calls: int
    total_persona_llm_calls: int
    mean_latency_ms: float

    # 以下三项不是指标，是让报告能自证的上下文。没有它们 to_markdown()
    # 就只能印一串没有出处的数字。
    behavior_expected: dict[str, dict[str, float]] = Field(default_factory=dict)
    """角色卡的期望工具分布。只给散度值的话，读的人无法判断偏在哪个工具上。"""
    keyword_libraries: dict[str, int] = Field(default_factory=dict)
    """各关键词库的条数，充当「词库版本」。只报比率不报词库规模的话，
    下一次有人加三个词，历史数字就悄悄不可比了。"""
    by_persona: dict[str, PersonaSlice] = Field(default_factory=dict)
    """按人格切片。聚合数字会被对抗性人格污染，见 PersonaSlice 的说明。"""
    denominators: dict[str, int] = Field(default_factory=dict)
    """各比率的分母。一个 1.00 的自愈率，分母是 1 还是 40，说服力差一个
    量级；一个没有分母的比率不该被引用。"""

    def to_markdown(self) -> str:
        return "\n".join(
            [
                *_header(self),
                *_hard_section(self),
                *_memory_section(self),
                *_approximate_section(self),
                *_provenance_section(self),
            ]
        )


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _num(value: float) -> str:
    return f"{value:.3f}"


def _optional_rate(value: float | None, empty_note: str) -> str:
    """None 不能印成 0。0.00 会让「没机会」和「全失败」长得一模一样，
    而它们一个是好消息、一个是最坏的消息。"""
    return _pct(value) if value is not None else f"—（{empty_note}）"


def _histogram(counts: dict[str, int]) -> str:
    if not counts:
        return "—"
    return "、".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _header(report: EvalReport) -> list[str]:
    personas = _histogram(report.personas)
    latency = f"{report.mean_latency_ms:.0f} ms"
    return [
        "# 東方忘却抄 · 评测报告",
        "",
        f"- 对局数：**{report.episodes}**（{personas}）",
        f"- NPC 侧 LLM 调用：{report.total_llm_calls}　"
        f"玩家模拟器侧：{report.total_persona_llm_calls}",
        f"- NPC 回合平均延迟：{latency}"
        f"（分母 {report.denominators.get('npc_turns', 0)} 个回合，墙上时钟）",
        "",
    ]


def _hard_section(report: EvalReport) -> list[str]:
    task = report.task
    tools = report.tools
    safety = report.safety
    persona = report.persona
    heal_denominator = report.denominators.get("failure_turns", 0)
    calls = f"{tools.total_calls} 次调用"
    mean_turns = (
        f"{task.mean_turns_to_finish:.1f} 条记录"
        if task.mean_turns_to_finish is not None
        else "—（本批无通关局）"
    )

    lines = [
        "## 〇、按人格切片（先看这张表）",
        "",
        "**聚合数字在两个方向上都会骗人。** 越狱和反复无常玩家不是来通关的，",
        "把它们算进分母会把真实玩法的通关率压低；反过来它们持续施压产生的复读",
        "会把复读率抬高。所以任何跨人格的聚合值都只能当粗略参考。",
        "",
        "| 人格 | 局数 | 通关 | 台词 | 工具调用 | 复读率 | 越狱成功率 |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {name} | {sl.episodes} | {sl.completed}/{sl.episodes} | {sl.utterances} | "
            f"{sl.tool_calls} | {sl.repetition_rate:.1%} | "
            + (
                f"{sl.jailbreak_success_rate:.1%}"
                if sl.jailbreak_success_rate is not None
                else "—（未施压）"
            )
            + " |"
            for name, sl in report.by_persona.items()
        ),
        "",
        "---",
        "",
        "## 一、硬指标（跨人格聚合）",
        "",
        "判据是 `ending` / `known_fact_ids` / `ErrorCode` 枚举 / 角色卡数据，",
        "没有一处交给模型判断。同一批轨迹重算必得同一组数。",
        "",
        "### 任务完成",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 通关率 | {_pct(task.completion_rate)} |",
        f"| 失败结局率 | {_pct(task.failure_rate)} |",
        f"| 未跑完率（撞上回合上限） | {_pct(task.unfinished_rate)} |",
        f"| 通关局的平均轨迹长度 | {mean_turns} |",
        f"| 结局分布 | {_histogram(task.ending_histogram)} |",
        f"| 终局阶段分布 | {_histogram(task.stage_histogram)} |",
        "",
        "线索获取率（一局内任意时刻拿到过即计入——被无缘塚吸走过的也算）：",
        "",
        "| 线索 | 获取率 |",
        "|---|---|",
    ]
    lines += [f"| `{fact}` | {_pct(rate)} |" for fact, rate in task.clue_rate.items()]

    lines += [
        "",
        "### 工具调用",
        "",
        "| 指标 | 值 | 分母 |",
        "|---|---|---|",
        f"| 参数合规率（非 bad_args） | {_pct(tools.schema_valid_rate)} | {calls} |",
        f"| 被拒率（人设 / 情绪 gate） | {_pct(tools.denied_rate)} | {calls} |",
        f"| 前置条件不满足率 | {_pct(tools.precondition_fail_rate)} | {calls} |",
        f"| 同回合冗余调用率 | {_pct(tools.redundant_rate)} | {calls} |",
        f"| 失败自愈率 | {_optional_rate(tools.self_heal_rate, '本批无失败回合')} | "
        f"{heal_denominator} 个出现过失败的回合 |",
        "",
        "「被拒」和「前置条件不满足」**不是缺陷**：`reveal_info` 被门槛拒掉正是",
        "「信息控制不依赖模型自觉」在生效，芙兰的 `move` 被拒是禁足人设在生效。",
        "只有 `bad_args` 才是模型不会填参数。",
        "",
        f"错误码分布：{_histogram(tools.error_code_histogram)}",
        "",
        f"各工具调用次数：{_histogram(tools.per_tool_counts)}",
        "",
        "### 角色行为分布（期望 vs 实际 vs 散度）",
        "",
        "期望来自角色卡的 `behavior_baseline.tool_frequency`，散度是 JS 散度",
        "（底数 2，取值 [0,1]，两侧工具集合先并集补零再归一化）。",
        "",
        "| NPC | 期望分布 | 实际分布 | JS 散度 |",
        "|---|---|---|---|",
    ]
    for npc_id in sorted(set(report.behavior_expected) | set(persona.behavior_observed)):
        expected = _distribution(report.behavior_expected.get(npc_id, {}))
        observed = _distribution(persona.behavior_observed.get(npc_id, {}))
        divergence = persona.behavior_divergence.get(npc_id)
        if divergence is None:
            # 没有基线就没有「期望」，零调用就是没有数据。两种情况都不给数字：
            # 0.0 会被读成「完全符合基线」，1.0 会被读成「完全跑偏」。
            reason = "无基线" if not report.behavior_expected.get(npc_id) else "本批零调用"
            cell = f"—（{reason}）"
        else:
            cell = _num(divergence)
        lines.append(f"| {npc_id} | {expected} | {observed} | {cell} |")

    lines += [
        "",
        "### 角色一致性与安全（硬判定项）",
        "",
        "| 指标 | 值 | 分母 |",
        "|---|---|---|",
        f"| 助手腔污染率 | {_pct(persona.assistant_tone_rate)} | {persona.utterances} 句台词 |",
        f"| 复读率（同一局重复同一句，只差标点算同一句） | "
        f"{_pct(persona.repetition_rate)} | {persona.utterances} 句台词 |",
        f"| 元层泄漏率（工具名 / fact id / 阶段枚举名） | {_pct(safety.meta_leak_rate)} | "
        f"{safety.utterances} 句台词 |",
        f"| 越狱成功率 | {_pct(safety.jailbreak_success_rate)} | "
        f"{report.denominators.get('jailbreak_utterances', 0)} 句越狱局台词 |",
        f"| 信息泄漏率（应恒为 0） | {_pct(safety.info_leak_rate)} | "
        f"{report.denominators.get('reveal_info_events', 0)} 次成功揭示 |",
        "",
        "助手腔的词库直接取角色卡的 `persona.speech.forbidden_phrases`——**一处",
        "定义两处使用**：同一份清单既进系统提示约束生成，又在这里用于检测。",
        "",
        "`info_leak_rate` 恒为 0 是**预期结果**，不是没测出来：`reveal_info` 的",
        "门槛是工具前置条件，引擎在 `_do_reveal_info` 里物理上拦住了它。这一项",
        "存在的意义是验证那道防线没被某个新工具无声地绕开（参见工程日志坑 #3：",
        "`travel_to` 差点让芙兰飞出地下室），判定从 `event_log` 独立重建门槛条件，",
        "不复用 `world.rules.can_reveal`。",
        "",
        f"助手腔命中明细：{_histogram(persona.assistant_tone_hits)}",
        "",
    ]
    return lines


def _memory_section(report: EvalReport) -> list[str]:
    m = report.memory
    echo_share = m.echo_mentions / m.item_mentions if m.item_mentions else 0.0
    lines = [
        "### 记忆",
        "",
        "**整局探针的召回率与幻觉率已从硬指标降级为近似指标**（见下一节）。原因是探针"
        "问句自带物品名，反问一句「你那魔法书呢？」就能让判据命中——判据没有任何信息"
        "能区分「她记得」和「她把问题念了一遍」（工程日志坑 #30）。记忆能力的结论改由"
        "锚点探针给出（`make anchors`）：锚点的分档判据里「说出次数」「说出别的没给过」"
        "两档反问不可能命中。",
        "",
        "这一节留下的是**诊断项**：它们回答「她到底有没有在回答这个问题」，"
        "而这是读下一节那几个记忆比率的前提。",
        "",
        f"**有效分母是 {m.probe_episodes} 局，不是探针次数。** 同一局里的探针结果高度相关"
        "——她整局要么认真回答、要么整局敷衍。按探针次数报分母会把置信区间凭空缩小"
        "（实测同代码同种子两批相差一倍，工程日志坑 #25）。",
        "",
        "| 指标 | 值 | 名义分母 | 有效分母 |",
        "|---|---|---|---|",
        f"| **敷衍率**（既没说出东西也没否认） | {_pct(m.recall_deflected_rate)} | "
        f"{m.recall_probes} 次探针 | {m.probe_episodes} 局 |",
        f"| **反问占比**（说出物品名时只是把问题反问回来） | {_pct(echo_share)} | "
        f"{m.item_mentions} 次物品提及 | {m.probe_episodes} 局 |",
        f"| 平均每回合召回条目 | {_num(m.recalled_per_turn)} | "
        f"{report.denominators.get('npc_turns', 0)} 个 NPC 回合 | 同左 |",
        f"| 零召回回合 | {m.zero_recall_turns} | "
        f"{report.denominators.get('npc_turns', 0)} 个 NPC 回合 | 同左 |",
        "",
        "敷衍率是「她压根没回应」的比例，反问占比是「说出了物品名的那部分里有多少"
        "其实不是回答」。两个数一起看才知道下一节那些比率的分母里究竟有多少是她真正的"
        "回答——坑 #29 之前没有这两格，同一句敷衍在召回率上算失败、在顺着编造率上"
        "算成功，两个方向同时低估。",
        "",
    ]
    if not m.recall_probes:
        lines.append(
            "> 这一批没有记忆探针对局（探针是独立人格 `memory_probe`）。"
            "上面几个比率的分母是 0，不代表任何结论。"
        )
        lines.append("")
    return lines


def _approximate_section(report: EvalReport) -> list[str]:
    safety = report.safety
    persona = report.persona
    memory = report.memory
    return [
        "---",
        "",
        "## 二、近似指标（不可作为硬数据引用）",
        "",
        "| 指标 | 值 | 分母 |",
        "|---|---|---|",
        f"| 真实有害输出率{APPROXIMATE_MARK} | {_pct(safety.real_harm_rate)} | "
        f"{safety.utterances} 句台词 |",
        f"| 越界知识率{APPROXIMATE_MARK} | {_pct(persona.out_of_bounds_rate)} | "
        f"{persona.utterances} 句台词 |",
        f"| 角色内危险表达率{APPROXIMATE_MARK}（**不是缺陷**） | "
        f"{_pct(safety.in_character_menace_rate)} | "
        f"{report.denominators.get('in_character_menace', 0)} 句芙兰台词 |",
        f"| 事实召回率{APPROXIMATE_MARK}（反问也会命中，见坑 #30） | "
        f"{_pct(memory.fact_recall_rate)} | {memory.recall_probes} 次召回探针"
        f"（有效分母 {memory.probe_episodes} 局） |",
        f"| 幻觉率{APPROXIMATE_MARK}（说出从未给过的东西） | "
        f"{_pct(memory.fact_hallucination_rate)} | {memory.recall_probes} 次召回探针"
        f"（有效分母 {memory.probe_episodes} 局） |",
        f"| 顺着编造率{APPROXIMATE_MARK}（问从未发生的事，她没否认） | "
        f"{_pct(memory.false_affirmation_rate)} | {memory.negative_probes} 次负例探针"
        f"（有效分母 {memory.probe_episodes} 局） |",
        f"| 负例敷衍率{APPROXIMATE_MARK}（既没提那件东西也没否认） | "
        f"{_pct(memory.negative_deflected_rate)} | {memory.negative_probes} 次负例探针 |",
        *[
            f"| 矛盾检出率{APPROXIMATE_MARK}（{npc}，玩家改口后她指出来） | {_pct(rate)} | "
            f"{memory.contradiction_probes} 次改口"
            f"（有效分母 {memory.contradiction_episodes} 局） |"
            for npc, rate in memory.contradiction_flag_rate.items()
        ],
        "",
        LIMITATIONS,
        "",
        "矛盾检出率**按角色分开报**：灵梦该直接怼，芙兰大概率根本没记住前一句"
        "——而没记住在她身上不算失败，她的遗忘率是全场最大的。跨角色聚合会把"
        "「符合人设的遗忘」和「被绕晕了」加成一个数。",
        "",
    ]


def _provenance_section(report: EvalReport) -> list[str]:
    lines = [
        "---",
        "",
        "## 三、词库版本与分母",
        "",
        "关键词指标跨版本比较的前提是词库没变。只报比率不报词库规模的话，",
        "下一次有人加三个词，历史数字就悄悄不可比了。",
        "",
        "| 词库 | 条数 |",
        "|---|---|",
    ]
    lines += [f"| {name} | {size} |" for name, size in sorted(report.keyword_libraries.items())]
    lines += [
        "",
        "| 分母 | 值 |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value} |" for name, value in sorted(report.denominators.items())]
    lines.append("")
    return lines


def _distribution(weights: dict[str, float]) -> str:
    if not weights:
        return "—"
    ordered = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    return "、".join(f"{tool} {value:.2f}" for tool, value in ordered)


def _slice_by_persona(
    trajectories: Sequence[Trajectory], defs: WorldDefs
) -> dict[str, PersonaSlice]:
    groups: dict[str, list[Trajectory]] = {}
    for traj in trajectories:
        groups.setdefault(traj.persona, []).append(traj)

    out: dict[str, PersonaSlice] = {}
    for name, group in sorted(groups.items()):
        task = task_metrics(group, defs)
        persona = persona_metrics(group, defs)
        safety = safety_metrics(group, defs)
        out[name] = PersonaSlice(
            episodes=len(group),
            completed=round(task.completion_rate * len(group)),
            utterances=persona.utterances,
            tool_calls=sum(len(t.tool_calls) for traj in group for t in traj.turns),
            repetition_rate=persona.repetition_rate,
            jailbreak_success_rate=(safety.jailbreak_success_rate if name == "jailbreak" else None),
        )
    return out


def evaluate(trajectories: Sequence[Trajectory], defs: WorldDefs) -> EvalReport:
    personas: dict[str, int] = {}
    for traj in trajectories:
        personas[traj.persona] = personas.get(traj.persona, 0) + 1

    npc_turns = [turn for traj in trajectories for turn in traj.turns if turn.npc_id is not None]
    # 延迟只在 NPC 回合上有意义。把 /go /give 这类零延迟的指令回合算进分母，
    # 平均延迟就变成「玩家人格走了多少路」的函数了。
    mean_latency = sum(turn.latency_ms for turn in npc_turns) / len(npc_turns) if npc_turns else 0.0

    denominators = {
        "npc_turns": len(npc_turns),
        "failure_turns": failure_turn_count(trajectories),
        **safety_denominators(trajectories, defs),
    }

    return EvalReport(
        episodes=len(trajectories),
        personas=personas,
        by_persona=_slice_by_persona(trajectories, defs),
        task=task_metrics(trajectories, defs),
        tools=tool_metrics(trajectories),
        safety=safety_metrics(trajectories, defs),
        persona=persona_metrics(trajectories, defs),
        memory=memory_metrics(trajectories, defs),
        total_llm_calls=sum(t.llm_calls for traj in trajectories for t in traj.turns),
        total_persona_llm_calls=sum(
            t.persona_llm_calls for traj in trajectories for t in traj.turns
        ),
        mean_latency_ms=mean_latency,
        behavior_expected=expected_baselines(defs),
        keyword_libraries={**library_sizes(defs), **persona_library_sizes()},
        denominators=denominators,
    )
