"""成对比较 judge 的测试。

重点在**双向交换的归并**：漏掉「交换后 LEFT/RIGHT 含义也反过来」这一步，
双向交换就退化成「两次问同一个方向」，位置偏差一点没消掉。
"""

import json
from pathlib import Path

import pytest

from gensokyo.testkit.judge import (
    JudgeTask,
    JudgeVerdict,
    judge_labels,
    read_verdicts,
    resolve_pair,
    tasks_for,
    write_sheet,
    write_tasks,
)
from gensokyo.testkit.metrics.agreement import Verdict

L, R, T = Verdict.LEFT, Verdict.RIGHT, Verdict.TIE
FIRST = "赛钱都不投就想问东问西。"
SECOND = "请问还有什么需要我帮忙的吗？"


def _tasks() -> list[JudgeTask]:
    return tasks_for("p1", "叙事合理性", "玩家问她无缘塚的事", FIRST, SECOND)


def _verdict(swapped: bool, said: Verdict) -> JudgeVerdict:
    """按 task_id 回填——执行者手里那张表没有 pair_id/swapped 字段。"""
    task = next(t for t in _tasks() if t.swapped is swapped)
    return JudgeVerdict(task_id=task.task_id, said=said, reason="因为")


# ---------------------------------------------------------------- 双向交换


def test_one_semantic_pair_becomes_two_directions() -> None:
    """只问一个方向的话，judge 偏好首位还是末位就无从得知——而那正是规格
    「位置偏差」那一条要消掉的东西。"""
    tasks = _tasks()

    assert len(tasks) == 2
    assert [t.swapped for t in tasks] == [False, True]
    assert (tasks[0].left, tasks[0].right) == (FIRST, SECOND)
    assert (tasks[1].left, tasks[1].right) == (SECOND, FIRST)


def test_a_swapped_task_translates_its_verdict_back_to_the_original_order() -> None:
    """交换过位置时 LEFT/RIGHT 的含义跟着反过来。**漏掉这一步是这个模块最容易
    出的错**：双向交换会退化成两次问同一个方向，而 `flipped` 计数虚高到接近全部。"""
    original, swapped = _tasks()

    assert original.resolve(L) is L
    assert swapped.resolve(L) is R
    assert swapped.resolve(R) is L


def test_tie_is_direction_independent() -> None:
    """平手不带方向，交换后仍是平手。"""
    _, swapped = _tasks()

    assert swapped.resolve(T) is T


def test_two_directions_that_agree_resolve_to_that_verdict() -> None:
    """第一次说「左边好」（= FIRST 好），交换后说「右边好」（= 仍然 FIRST 好）
    ——两次都指向 FIRST，所以结论是 FIRST 更好。"""
    tasks = _tasks()

    resolved = resolve_pair(tasks, [_verdict(False, L), _verdict(True, R)])

    assert resolved is L


def test_two_directions_that_both_favour_the_left_slot_are_a_contradiction() -> None:
    """两次都说「左边好」意味着第一次选 FIRST、第二次选 SECOND——judge 换个
    位置就改主意，那是位置偏差本身。这种对子必须排除，**而不是随便取一个方向**：
    取一个方向等于把位置偏差当成判定。"""
    tasks = _tasks()

    assert resolve_pair(tasks, [_verdict(False, L), _verdict(True, L)]) is None


def test_a_tie_in_one_direction_only_is_also_unstable() -> None:
    """一次平手一次不平手同样算不自洽。把它当成非平手那一侧会让「judge 犹豫」
    被记成一个确定判定。"""
    tasks = _tasks()

    assert resolve_pair(tasks, [_verdict(False, T), _verdict(True, R)]) is None


def test_a_pair_judged_in_only_one_direction_is_not_a_valid_observation() -> None:
    """执行者漏填一半时不能凑数——半个对子不构成有效观测。"""
    tasks = _tasks()

    assert resolve_pair(tasks, [_verdict(False, L)]) is None


def test_both_ties_resolve_to_tie() -> None:
    tasks = _tasks()

    assert resolve_pair(tasks, [_verdict(False, T), _verdict(True, T)]) is T


# ---------------------------------------------------------------- 汇总


def test_labels_and_flip_count_are_reported_together() -> None:
    """`judge_flipped` 大到某个程度说明 judge 在这个维度上不可用，那时高 κ
    只是剩下那些容易对子造成的假象。所以两个数必须一起出。"""
    stable = tasks_for("ok", "叙事合理性", "上下文", FIRST, SECOND)
    unstable = tasks_for("bad", "叙事合理性", "上下文", FIRST, SECOND)
    verdicts = [
        JudgeVerdict(task_id=stable[0].task_id, said=L),
        JudgeVerdict(task_id=stable[1].task_id, said=R),
        JudgeVerdict(task_id=unstable[0].task_id, said=L),
        JudgeVerdict(task_id=unstable[1].task_id, said=L),
    ]

    labels, flipped = judge_labels([*stable, *unstable], verdicts)

    assert labels == {"ok": L}
    assert flipped == 1


def test_a_pair_with_no_verdicts_counts_as_unresolved() -> None:
    """执行者整对漏掉时不能当成「没这个对子」——它得计入自相矛盾/无效，
    否则 κ 的分母会悄悄缩水而报告照旧印数（坑 #25 那一类）。"""
    labels, flipped = judge_labels(_tasks(), [])

    assert labels == {}
    assert flipped == 1


# ---------------------------------------------------------------- 文件往返


def test_tasks_round_trip_through_jsonl(tmp_path: Path) -> None:
    """执行者拿到的是这个文件，原样加一个 `said` 字段填回。换执行者
    （另一个模型、或真人）不需要改代码。"""
    path = tmp_path / "tasks.jsonl"

    written = write_tasks(_tasks(), path)

    assert written == 2
    assert "赛钱" in path.read_text(encoding="utf-8")


def test_verdicts_are_read_back_and_blank_lines_ignored(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.jsonl"
    path.write_text(
        _verdict(False, L).model_dump_json() + "\n\n" + _verdict(True, R).model_dump_json() + "\n",
        encoding="utf-8",
    )

    assert len(read_verdicts(path)) == 2


def test_a_malformed_verdict_line_is_loud(tmp_path: Path) -> None:
    """静默跳过格式错误的行会让 κ 的分母悄悄缩水。fail-loud 的成本在写代码时，
    fail-silent 的成本在查 bug 时（坑 #5）。"""
    path = tmp_path / "verdicts.jsonl"
    path.write_text('{"task_id": "abc123"}\n', encoding="utf-8")

    with pytest.raises(Exception, match="said"):
        read_verdicts(path)


# ---------------------------------------------------------------- 盲化


def test_the_sheet_given_to_the_judge_hides_the_pairing() -> None:
    """**judge 不能知道「同一对出现了两次、只是左右调换」**——那是把检测手段
    告诉被检测者。第一次校准时我在 prompt 里写明了 swapped 的含义，于是那个
    「自相矛盾 0/30」是被污染的（judge 会刻意保持一致）。

    所以给 judge 的行里只有 task_id / context / A / B。"""
    rows = [t.sheet_row() for t in _tasks()]

    assert all(set(r) == {"task_id", "context", "A", "B"} for r in rows)
    assert rows[0]["task_id"] != rows[1]["task_id"]


def test_the_task_id_is_stable_but_does_not_leak_the_pair() -> None:
    """稳定：同一批导出两次得同一个 id（可复现）。不泄漏：看不出哪两行同源
    ——所以它不能是 `pair_id` 加个后缀。"""
    first, second = _tasks()

    assert first.task_id == _tasks()[0].task_id
    assert "p1" not in first.task_id
    assert "p1" not in second.task_id


def test_the_sheet_row_order_is_shuffled_but_reproducible(tmp_path: Path) -> None:
    """不打乱的话同源两行紧挨着，judge 一眼就看出来。种子固定所以可复现。"""
    many = [t for i in range(12) for t in tasks_for(f"p{i}", "叙事合理性", "上下文", FIRST, SECOND)]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"

    assert write_sheet(many, a, seed=0) == len(many)
    write_sheet(many, b, seed=0)

    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
    # 打乱之后，同源两行不再总是相邻。
    ids = [json.loads(ln)["task_id"] for ln in a.read_text(encoding="utf-8").splitlines()]
    original = [t.task_id for t in many]
    assert ids != original


def test_verdicts_for_unknown_task_ids_are_ignored_not_crashed() -> None:
    """执行者可能把行搞乱或多填一行。未知 task_id 应当被忽略，而对应的对子
    因为缺一个方向自然计入无效——不能让一行脏数据崩掉整轮校准。"""
    labels, flipped = judge_labels(
        _tasks(), [_verdict(False, L), JudgeVerdict(task_id="deadbeef", said=R)]
    )

    assert labels == {}
    assert flipped == 1
