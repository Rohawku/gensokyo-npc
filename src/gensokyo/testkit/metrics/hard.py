"""任务完成与工具调用指标。全部硬指标：判据是 `ending`、`known_fact_ids`
和 `ErrorCode` 枚举，没有一处需要模型判断，同一批轨迹重算必得同一组数。

`ErrorCode` 与给 LLM 看的 `error` 早就分成了两个字段（工程日志取舍 #4），
这里是那个决定的兑现处：错误分类全部按枚举值统计，改一句提示文案不会让
历史指标断档。
"""

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from gensokyo.testkit.trajectory import Trajectory, TurnRecord
from gensokyo.world.defs import WorldDefs
from gensokyo.world.ids import FactId
from gensokyo.world.quest import TIMEOUT_ENDING
from gensokyo.world.tools import ErrorCode

PRECONDITION_CODES: frozenset[str] = frozenset(
    {
        ErrorCode.NO_SUCH_EXIT.value,
        ErrorCode.INSUFFICIENT_ITEM.value,
        ErrorCode.REVEAL_CONDITION_UNMET.value,
        ErrorCode.NOT_FACT_HOLDER.value,
        ErrorCode.UNREACHABLE.value,
    }
)
"""前置条件不满足。**这一类不是缺陷**——`reveal_info` 被门槛拒掉正是
「信息控制不依赖模型自觉」在生效，芙兰的 `move` 被拒是禁足人设在生效。
把它和 bad_args（模型不会填参数）混成一个「失败率」，就分不清是模型不行
还是世界规则在起作用。所以它单独一个字段。"""


class TaskMetrics(BaseModel):
    episodes: int
    completion_rate: float
    """finished 且 ending 不是失败结局的比例。"""
    failure_rate: float
    """走到失败结局（动作数耗尽、玩家把线索忘光）的比例。"""
    unfinished_rate: float
    """到 max_turns 仍未结束。它和 failure_rate 必须分开：前者是没跑完，
    后者是跑完了但输了，改 max_turns 只会动前者。"""
    mean_turns_to_finish: float | None
    stage_histogram: dict[str, int]
    clue_rate: dict[str, float]
    ending_histogram: dict[str, int]


class ToolMetrics(BaseModel):
    total_calls: int
    schema_valid_rate: float
    """非 bad_args 的比例。它衡量的是「模型会不会按 schema 填参数」，
    和「这次调用在世界规则上做不做得到」是两件事。"""
    denied_rate: float
    precondition_fail_rate: float
    self_heal_rate: float | None
    """分母为 0（本批没有任何失败回合）时是 None，不是 0.0。"""
    redundant_rate: float
    error_code_histogram: dict[str, int]
    per_tool_counts: dict[str, int]


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _args_key(args: dict[str, Any]) -> str:
    """把参数字典压成稳定字符串。排序后再序列化，否则 {"a":1,"b":2} 和
    {"b":2,"a":1} 会被当成两次不同的调用。"""
    return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def _call_key(call: dict[str, Any]) -> str:
    args = call.get("args") or {}
    return f"{call.get('tool', '')}#{_args_key(args if isinstance(args, dict) else {})}"


def _paired(turn: TurnRecord) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """按下标把调用和结果配对。

    两个列表由 `agent/policy.py` 保证等长同序（含重试时失败的那次）。
    `strict=False` 只是为了让更早格式的轨迹不至于炸掉——真错位的话短的
    那个说话，指标偏保守而不是给出错误的配对。
    """
    return list(zip(turn.tool_calls, turn.tool_results, strict=False))


def _known_fact_ids(traj: Trajectory) -> set[str]:
    """整局中任意时刻拿到过的线索。

    取并集而不是取最后一帧：玩家在无缘塚待久了会被吸走线索，只看结局
    那一帧会把「拿到过又丢了」记成「从没拿到」——那是两个不同的问题。
    """
    seen: set[str] = set()
    for turn in traj.turns:
        ids = turn.view_after.get("known_fact_ids") or []
        if isinstance(ids, list):
            seen.update(str(i) for i in ids)
    return seen


def _is_completed(traj: Trajectory) -> bool:
    return traj.finished and traj.ending != TIMEOUT_ENDING


def task_metrics(trajectories: Sequence[Trajectory], defs: WorldDefs) -> TaskMetrics:
    episodes = len(trajectories)
    completed = [t for t in trajectories if _is_completed(t)]
    failed = sum(1 for t in trajectories if t.ending == TIMEOUT_ENDING)
    unfinished = sum(1 for t in trajectories if not t.finished)

    stage_histogram: dict[str, int] = {}
    ending_histogram: dict[str, int] = {}
    for traj in trajectories:
        if traj.final_stage:
            stage_histogram[traj.final_stage] = stage_histogram.get(traj.final_stage, 0) + 1
        if traj.ending is not None:
            ending_histogram[traj.ending] = ending_histogram.get(traj.ending, 0) + 1

    obtained = [_known_fact_ids(t) for t in trajectories]
    clue_rate = {
        str(fact_id): _rate(sum(1 for got in obtained if str(fact_id) in got), episodes)
        # 遍历 defs 里的全部线索而不是轨迹里出现过的：一条从来没人拿到的
        # 线索必须以 0.0 出现在报告里，缺项和 0 在阅读时长得完全不一样。
        for fact_id in sorted(defs.clue_facts())
    }

    mean_turns: float | None = None
    if completed:
        # 分母只算通关局。把没打完的局混进来会让「平均多少回合能通」
        # 变成「平均一局有多长」，而后者只反映 max_turns 设了多少。
        # 一条记录是一个说话人，同场两个 NPC 的一次输入算两条。
        mean_turns = sum(len(t.turns) for t in completed) / len(completed)

    return TaskMetrics(
        episodes=episodes,
        completion_rate=_rate(len(completed), episodes),
        failure_rate=_rate(failed, episodes),
        unfinished_rate=_rate(unfinished, episodes),
        mean_turns_to_finish=mean_turns,
        stage_histogram=stage_histogram,
        clue_rate=clue_rate,
        ending_histogram=ending_histogram,
    )


def _turn_self_healed(pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> bool | None:
    """这一回合有没有完成一次有效自愈。

    返回 None 表示本回合根本没有失败，不进分母——「没机会自愈」和
    「自愈全失败」在报告里必须长得不一样。

    有效自愈 = 首次失败之后，同一回合里出现了一次**换了招**（tool 或 args
    不同）且成功的调用。重复同一个 (tool, args) 直到蒙对不算自愈，那是
    引擎或采样的抖动，不是模型读懂了回灌的错误原因。
    """
    failed_at = [i for i, (_, result) in enumerate(pairs) if not result.get("ok")]
    if not failed_at:
        return None
    failed_keys = {_call_key(pairs[i][0]) for i in failed_at}
    return any(
        result.get("ok") and _call_key(call) not in failed_keys
        for call, result in pairs[failed_at[0] + 1 :]
    )


def tool_metrics(trajectories: Sequence[Trajectory]) -> ToolMetrics:
    total = 0
    bad_args = 0
    denied = 0
    precondition = 0
    redundant = 0
    turns_with_failure = 0
    turns_healed = 0
    error_code_histogram: dict[str, int] = {}
    per_tool_counts: dict[str, int] = {}

    for traj in trajectories:
        for turn in traj.turns:
            pairs = _paired(turn)
            total += len(pairs)

            seen_keys: set[str] = set()
            for call, result in pairs:
                tool = str(call.get("tool", ""))
                per_tool_counts[tool] = per_tool_counts.get(tool, 0) + 1

                key = _call_key(call)
                if key in seen_keys:
                    redundant += 1
                seen_keys.add(key)

                code = result.get("error_code")
                if code is None:
                    continue
                code = str(code)
                error_code_histogram[code] = error_code_histogram.get(code, 0) + 1
                if code == ErrorCode.BAD_ARGS.value:
                    bad_args += 1
                elif code == ErrorCode.TOOL_DENIED.value:
                    denied += 1
                elif code in PRECONDITION_CODES:
                    precondition += 1

            healed = _turn_self_healed(pairs)
            if healed is not None:
                turns_with_failure += 1
                turns_healed += int(healed)

    return ToolMetrics(
        total_calls=total,
        schema_valid_rate=1.0 - _rate(bad_args, total) if total else 0.0,
        denied_rate=_rate(denied, total),
        precondition_fail_rate=_rate(precondition, total),
        self_heal_rate=(_rate(turns_healed, turns_with_failure) if turns_with_failure else None),
        redundant_rate=_rate(redundant, total),
        error_code_histogram=error_code_histogram,
        per_tool_counts=per_tool_counts,
    )


def failure_turn_count(trajectories: Sequence[Trajectory]) -> int:
    """出现过工具失败的回合数——`self_heal_rate` 的分母。

    报告要能把它印出来：一个 1.00 的自愈率，分母是 1 还是 40，说服力
    差一个量级。
    """
    return sum(
        1
        for traj in trajectories
        for turn in traj.turns
        if _turn_self_healed(_paired(turn)) is not None
    )


class RevealDeliveryMetrics(BaseModel):
    """成功揭示之后，情报内容有没有真的到玩家耳朵里。

    **`reveal_info` 成功只说明工具成功了。** 引擎把情报记进 `known_facts`、面板上
    「已知线索」多一条，而她那句台词完全可以一个字都不提——玩家于是在整个游戏
    最重要的一刻听到一句「你倒是说说看，我倒要听听」。

    这是硬判定：分子是「同一回合的台词里出现了该情报的 `marks` 之一」，分母是
    成功的 `reveal_info` 次数，两边都来自轨迹，没有一处交给模型判断。
    """

    reveals: int
    delivered: int
    by_fact: dict[str, tuple[int, int]]
    """fact id -> (说出口的次数, 成功揭示的次数)。**按情报拆**：三条线索的持有者
    是三个不同性格的 NPC，聚合成一个数会把「芙兰从来不说」和「魔理沙偶尔不说」
    混在一起。"""

    @property
    def delivery_rate(self) -> float:
        return _rate(self.delivered, self.reveals)


def reveal_delivery_metrics(
    trajectories: Sequence[Trajectory], defs: WorldDefs
) -> RevealDeliveryMetrics:
    by_fact: dict[str, list[int]] = {}
    for traj in trajectories:
        for turn in traj.turns:
            for call, result in _paired(turn):
                if call.get("tool") != "reveal_info" or not result.get("ok"):
                    continue
                fact_id = FactId(str((call.get("args") or {}).get("fact", "")))
                fact = defs.facts.get(fact_id)
                if fact is None:
                    continue
                said = any(mark in turn.utterance for mark in fact.marks)
                slot = by_fact.setdefault(fact_id, [0, 0])
                slot[0] += int(said)
                slot[1] += 1
    return RevealDeliveryMetrics(
        reveals=sum(v[1] for v in by_fact.values()),
        delivered=sum(v[0] for v in by_fact.values()),
        by_fact={k: (v[0], v[1]) for k, v in sorted(by_fact.items())},
    )
