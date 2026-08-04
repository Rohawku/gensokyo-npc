"""LLM-as-Judge：成对比较，双向交换去位置偏差。

**judge 不在这个模块里直连模型。** 它把待判的对子导出成 JSONL，读回判定，
算一致性。理由有三个：

1. **judge 必须 ≠ policy**（规格 5.6 的自我偏好一条）。policy 是本地
   `qwen3:8b`，而这个仓库只配了那一个端点，所以 judge 的执行者必然在外部。
2. **judge 刻意不全量调用**（design.md:185）。它只用于最终评估与人工抽查
   辅助，不进 reward——judge 最容易被 hack，模型会学会写讨好 judge 的文风。
3. **可审计**。谁判的、判了什么、什么时候判的，全在文件里。换一个执行者
   （另一个模型、或真人）不需要改代码。

**为什么是成对比较而不是 1–5 分**：规格里「分数聚集」那一条。绝对分数在 LLM
上会挤在 3~4 之间，方差小到分不出东西；成对偏好只要求「哪个更好」，人和模型
都答得稳，而且它和偏好数据的形态一致（同一个 prompt 下的两条候选）。
"""

import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel

from gensokyo.testkit.metrics.agreement import Verdict


class JudgeTask(BaseModel):
    """一个方向的比较任务。

    同一个语义对会产生**两个** task（`swapped` 一真一假），因为位置偏差的对策
    是双向交换取平均——只问一个方向的话，judge 偏好首位还是末位就无从得知。
    """

    pair_id: str
    """语义对的 id。两个方向共享它，读回时靠它把两次判定配起来。"""
    dimension: str
    swapped: bool
    """False 表示 left 是原始的第一条；True 表示两条已经交换过位置。"""
    context: str
    """玩家那句话与场景，judge 判「回应得合不合理」需要它。"""
    left: str
    right: str

    @property
    def task_id(self) -> str:
        """给 judge 看的**不透明** id。

        judge 不能知道「同一对出现了两次、只是左右调换」——那是在告诉被检测者
        检测手段。第一次校准时我在 prompt 里写明了 swapped 的含义，于是那个
        「自相矛盾 0/30」是被污染的（judge 会刻意保持一致）。

        所以给 judge 的表里只有 `task_id`，而它是 `pair_id + 方向`的哈希：
        稳定（同一批导出两次得同一个 id）、不可逆推（看不出哪两行同源）。
        """
        raw = f"{self.pair_id}|{self.swapped}".encode()
        return hashlib.sha1(raw).hexdigest()[:12]

    def sheet_row(self) -> dict[str, str]:
        """给 judge 的一行。**刻意只有这四个字段**——多一个 `pair_id` 就够它
        把同源的两行认出来。"""
        return {"task_id": self.task_id, "context": self.context, "A": self.left, "B": self.right}

    def resolve(self, said: Verdict) -> Verdict:
        """把这个方向上的判定翻译回**原始顺序**下的结论。

        交换过位置时 LEFT/RIGHT 的含义也跟着反过来——漏掉这一步会让双向交换
        变成「两次问同一个方向」，位置偏差一点没消掉，而 `flipped` 计数会
        虚高到接近全部。
        """
        if not self.swapped or said is Verdict.TIE:
            return said
        return Verdict.RIGHT if said is Verdict.LEFT else Verdict.LEFT


class JudgeVerdict(BaseModel):
    """执行者填回来的一条判定。

    **按 `task_id` 回填，不是 pair_id + swapped**——执行者手里那张表根本没有
    后两个字段（见 `JudgeTask.sheet_row`）。代码侧靠 `task_id` 映射回方向。
    """

    task_id: str
    said: Verdict
    reason: str = ""
    """为什么。**不参与计算，但必须留着**——一条说不出理由的判定没法审计，
    而 judge 的可信度全靠人能复核它（同 `PreferencePair.reason`）。"""


def tasks_for(
    pair_id: str, dimension: str, context: str, first: str, second: str
) -> list[JudgeTask]:
    """一个语义对 → 两个方向的任务。"""
    return [
        JudgeTask(
            pair_id=pair_id,
            dimension=dimension,
            swapped=False,
            context=context,
            left=first,
            right=second,
        ),
        JudgeTask(
            pair_id=pair_id,
            dimension=dimension,
            swapped=True,
            context=context,
            left=second,
            right=first,
        ),
    ]


def resolve_pair(tasks: Sequence[JudgeTask], verdicts: Sequence[JudgeVerdict]) -> Verdict | None:
    """两个方向的判定归并成一个结论。**不自洽就返回 None。**

    `None` 表示 judge 自己都不稳定：换个位置就改主意。那种对子被排除在 κ 之外
    ——和人工比一致性没有意义——但会计入 `judge_flipped` 报出来。**排除而不是
    随便取一个方向**：取一个方向等于把位置偏差当成判定，而那正是双向交换要
    消掉的东西。

    缺一个方向也返回 None：只判了一半的对子不构成有效观测。
    """
    said_by_task = {v.task_id: v.said for v in verdicts}
    answered = [t for t in tasks if t.task_id in said_by_task]
    if len({t.swapped for t in answered}) != 2:
        return None

    resolved = {t.resolve(said_by_task[t.task_id]) for t in answered}
    return resolved.pop() if len(resolved) == 1 else None


def write_tasks(tasks: Iterable[JudgeTask], path: Path) -> int:
    """导出**内部**任务文件（含 pair_id 与 swapped）。judge 不读这个。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(task.model_dump_json() + "\n")
            written += 1
    return written


def write_sheet(tasks: Sequence[JudgeTask], path: Path, seed: int = 0) -> int:
    """导出**给 judge 的**表：只有 task_id / context / A / B，且行序打乱。

    打乱是盲化的第二半。不打乱的话同源的两行紧挨着，judge 一眼就看出「这两行
    是同一对，只是左右调换了」——那等于把位置偏差的检测手段告诉被检测者。
    种子固定，所以同一批导出两次得同一个顺序（可复现）。
    """
    rows = [t.sheet_row() for t in tasks]
    random.Random(seed).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def read_verdicts(path: Path) -> list[JudgeVerdict]:
    """读回判定，跳过空行。

    格式不合的行**直接抛异常**而不是跳过：一个判定文件里有半数行填错格式，
    静默跳过会让 κ 的分母悄悄缩水，而报告照旧印一个数（坑 #25 那一类）。
    """
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [JudgeVerdict.model_validate_json(ln) for ln in lines]


def flatten(tasks: Sequence[JudgeTask]) -> dict[str, list[JudgeTask]]:
    """按 pair_id 分组，方便逐对归并。"""
    out: dict[str, list[JudgeTask]] = {}
    for task in tasks:
        out.setdefault(task.pair_id, []).append(task)
    return out


def judge_labels(
    tasks: Sequence[JudgeTask], verdicts: Sequence[JudgeVerdict]
) -> tuple[dict[str, Verdict], int]:
    """(pair_id -> 归并后的结论, judge 自相矛盾的对数)。"""
    by_pair = flatten(tasks)
    task_to_pair = {t.task_id: t.pair_id for t in tasks}
    grouped: dict[str, list[JudgeVerdict]] = {}
    for verdict in verdicts:
        pair = task_to_pair.get(verdict.task_id)
        if pair is not None:
            grouped.setdefault(pair, []).append(verdict)

    labels: dict[str, Verdict] = {}
    flipped = 0
    for pair_id, pair_tasks in by_pair.items():
        resolved = resolve_pair(pair_tasks, grouped.get(pair_id, []))
        if resolved is None:
            flipped += 1
        else:
            labels[pair_id] = resolved
    return labels, flipped


def load_json_lines(path: Path) -> list[dict[str, object]]:
    """给需要原样读回任务文件的地方用（比如把待判对子渲染给人看）。"""
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
