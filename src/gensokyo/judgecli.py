"""judge 校准入口：导出待判 → judge 判 → 人工标注 → 算 κ。

三步，中间两步在这个进程之外发生（judge 是外部模型，人是人）：

    # 1. 导出。judge 判全部对子，人工只标一半带建议、一半盲标
    python -m gensokyo.judgecli export --pairs reports/pairs.jsonl --out reports/judge

    # 2. 让 judge 填 reports/judge/judge_verdicts.jsonl（外部模型，≠ policy）
    #    再由人填 human_primed.jsonl 与 human_blind.jsonl 里的 said 字段

    # 3. 算一致性与准入
    python -m gensokyo.judgecli score --out reports/judge

**为什么人工要分两组**：这个仓库只配了一个端点（policy 是本地 `qwen3:8b`），
而规格要求 judge ≠ policy，所以 judge 与「预标注」只能是同一个外部模型。那样
人工复核后的标签就带着 judge 的先验——锚定偏差会让 κ 虚高。盲标那一组是准入
依据，两组的 κ 之差就是锚定偏差的量。见 `metrics/agreement.py`。
"""

import argparse
import json
import sys
from pathlib import Path

from gensokyo.cli import REPO_ROOT
from gensokyo.testkit.judge import (
    JudgeTask,
    judge_labels,
    read_verdicts,
    tasks_for,
    write_sheet,
    write_tasks,
)
from gensokyo.testkit.metrics.agreement import (
    KAPPA_ADMISSION,
    Agreement,
    AnchoringCheck,
    Verdict,
    cohens_kappa,
    correction_rate,
)

DIMENSION = "叙事合理性"
"""当前校准的维度。judge 只做规格里那两个软维度，硬判据能判的不交给它。"""

TASKS = "judge_tasks.jsonl"
SHEET = "judge_sheet.jsonl"
JUDGE = "judge_verdicts.jsonl"
PRIMED = "human_primed.jsonl"
BLIND = "human_blind.jsonl"


def split_groups(pair_ids: list[str]) -> tuple[list[str], list[str]]:
    """交替分配成预标注组与盲标组。

    **交替而不是前半/后半**：对子是按局产出的，前半后半可能整体来自不同人格的
    对局，那样两组的样本特征就不可比，而 κ 之差会把「样本不同」读成「锚定偏差」。
    """
    ordered = sorted(pair_ids)
    return ordered[0::2], ordered[1::2]


def _pairs_from_jsonl(path: Path) -> list[dict[str, str]]:
    """从偏好对文件里取出成对候选与玩家那句话。

    `context` 只取**玩家最后那句话**：`prompt` 里「玩家：」之后紧跟着
    【你刚才的想法】【你还记得的事】那些段落，整段喂给 judge 会让它去评价
    prompt 本身而不是那两条候选。
    """
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [
        {
            "pair_id": r["source"],
            "context": r["prompt"].rsplit("玩家：", 1)[-1].split("\n", 1)[0].strip(),
            "first": r["chosen"],
            "second": r["rejected"],
        }
        for r in rows
    ]


def _blank_sheet(tasks: list[JudgeTask], suggestion: dict[str, Verdict] | None) -> list[str]:
    """人工标注表。每行一个**语义对**（不是方向）——让人判两遍同一对毫无意义，
    位置偏差是 judge 的问题，不是人的。
    """
    seen: set[str] = set()
    lines: list[str] = []
    for task in tasks:
        if task.swapped or task.pair_id in seen:
            continue
        seen.add(task.pair_id)
        row: dict[str, object] = {
            "pair_id": task.pair_id,
            "context": task.context,
            "A": task.left,
            "B": task.right,
            "said": "",
        }
        if suggestion is not None:
            hint = suggestion.get(task.pair_id)
            row["judge_suggests"] = hint.value if hint else "unresolved"
        lines.append(json.dumps(row, ensure_ascii=False))
    return lines


def do_export(pairs_path: Path, out: Path, size: int) -> int:
    if not pairs_path.is_file():
        print(f"{pairs_path} 不存在。先跑 `make harvest` 产出候选对。", file=sys.stderr)
        return 2

    pairs = _pairs_from_jsonl(pairs_path)[:size]
    if not pairs:
        print(f"{pairs_path} 里没有可用的对子。", file=sys.stderr)
        return 2

    tasks = [
        t
        for p in pairs
        for t in tasks_for(p["pair_id"], DIMENSION, p["context"], p["first"], p["second"])
    ]
    out.mkdir(parents=True, exist_ok=True)
    write_tasks(tasks, out / TASKS)
    # 给 judge 的是盲化过的那一份：只有 task_id/context/A/B，且行序打乱。
    write_sheet(tasks, out / SHEET)

    primed_ids, blind_ids = split_groups([p["pair_id"] for p in pairs])
    (out / BLIND).write_text(
        "\n".join(_blank_sheet([t for t in tasks if t.pair_id in set(blind_ids)], None)) + "\n",
        encoding="utf-8",
    )

    print(f"内部任务 {len(tasks)} 条（{len(pairs)} 对 × 2 方向）→ {out / TASKS}")
    print(f"**给 judge 的盲化表** → {out / SHEET}（只有 task_id/context/A/B，行序已打乱）")
    print(f"盲标表 {len(blind_ids)} 对 → {out / BLIND}")
    print(f"\n下一步：让**外部模型**（≠ policy 的 qwen3:8b）读 {SHEET}，")
    print(f"每行补一个 said 字段（left/right/tie）写成 {JUDGE}；")
    print("然后跑 `judgecli primed` 生成带建议的那一半。")
    return 0


def do_primed(out: Path) -> int:
    """judge 判完之后才能生成预标注表——建议就是 judge 的判定。"""
    tasks = [JudgeTask.model_validate_json(ln) for ln in _lines(out / TASKS)]
    if not (out / JUDGE).is_file():
        print(f"{out / JUDGE} 还不存在——先让 judge 判。", file=sys.stderr)
        return 2

    labels, flipped = judge_labels(tasks, read_verdicts(out / JUDGE))
    primed_ids, _ = split_groups(sorted({t.pair_id for t in tasks}))
    sheet = _blank_sheet([t for t in tasks if t.pair_id in set(primed_ids)], labels)
    (out / PRIMED).write_text("\n".join(sheet) + "\n", encoding="utf-8")

    print(f"judge 有效判定 {len(labels)} 对，自相矛盾 {flipped} 对")
    print(f"预标注表 {len(sheet)} 对 → {out / PRIMED}（`said` 留空，`judge_suggests` 是建议）")
    return 0


def _lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _human_labels(path: Path) -> tuple[dict[str, Verdict], dict[str, Verdict]]:
    """(pair_id -> 人工标签, pair_id -> judge 建议)。`said` 为空的行跳过。"""
    labels: dict[str, Verdict] = {}
    suggested: dict[str, Verdict] = {}
    for row in (json.loads(ln) for ln in _lines(path)):
        said = str(row.get("said", "")).strip()
        if not said:
            continue
        labels[row["pair_id"]] = Verdict(said)
        hint = str(row.get("judge_suggests", "")).strip()
        if hint in {v.value for v in Verdict}:
            suggested[row["pair_id"]] = Verdict(hint)
    return labels, suggested


def _agreement(judge: dict[str, Verdict], human: dict[str, Verdict], flipped: int) -> Agreement:
    shared = sorted(set(judge) & set(human))
    left = [judge[k] for k in shared]
    right = [human[k] for k in shared]
    return Agreement(
        dimension=DIMENSION,
        pairs=len(shared),
        agreed=sum(1 for a, b in zip(left, right, strict=True) if a == b),
        kappa=cohens_kappa(left, right),
        judge_flipped=flipped,
    )


def do_score(out: Path) -> int:
    for name in (TASKS, JUDGE):
        if not (out / name).is_file():
            print(f"{out / name} 不存在。", file=sys.stderr)
            return 2

    tasks = [JudgeTask.model_validate_json(ln) for ln in _lines(out / TASKS)]
    judge, flipped = judge_labels(tasks, read_verdicts(out / JUDGE))

    primed_human: dict[str, Verdict] = {}
    suggestion: dict[str, Verdict] = {}
    if (out / PRIMED).is_file():
        primed_human, suggestion = _human_labels(out / PRIMED)
    blind_human: dict[str, Verdict] = {}
    if (out / BLIND).is_file():
        blind_human, _ = _human_labels(out / BLIND)

    shared = sorted(set(suggestion) & set(primed_human))
    check = AnchoringCheck(
        primed=_agreement(judge, primed_human, flipped),
        blind=_agreement(judge, blind_human, flipped),
        correction_rate=correction_rate(
            [suggestion[k] for k in shared], [primed_human[k] for k in shared]
        ),
    )

    print(f"judge 有效判定 {len(judge)} 对、自相矛盾 {flipped} 对（位置偏差的直接观测）\n")
    print(f"预标注组：{check.primed}")
    print(f"盲标组　：{check.blind}")
    print(f"\n{check}")
    if not check.admissible:
        print(
            f"\n**{DIMENSION} 的数字不得用于下结论**"
            f"（盲标 κ 未过 {KAPPA_ADMISSION}，或还没有盲标数据）。"
        )
    (out / "agreement.json").write_text(check.model_dump_json(indent=2), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gensokyo.judgecli", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    export = sub.add_parser("export", help="导出待判任务与盲标表")
    export.add_argument("--pairs", type=Path, default=REPO_ROOT / "reports" / "pairs.jsonl")
    export.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "judge")
    export.add_argument("--size", type=int, default=40, help="取多少对")

    primed = sub.add_parser("primed", help="judge 判完后生成带建议的标注表")
    primed.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "judge")

    score = sub.add_parser("score", help="算 κ、修正率与锚定偏差")
    score.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "judge")

    args = parser.parse_args(argv)
    if args.cmd == "export":
        return do_export(args.pairs, args.out, args.size)
    if args.cmd == "primed":
        return do_primed(args.out)
    return do_score(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
