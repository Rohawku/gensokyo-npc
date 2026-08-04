"""锚点探针的批量入口。

    python -m gensokyo.anchorcli --repeats 30

每个锚点独立采样 `repeats` 次并按分级判据报带区间的比率。**产出必须带区间**：
裸比率是这个项目栽过最多次的地方（坑 #18、#25、#26、#28）。
"""

import argparse
import json
from pathlib import Path

from gensokyo.cli import load_dotenv
from gensokyo.llm.client import OpenAiCompatibleClient
from gensokyo.testkit.anchor_set import ANCHORS, BY_ID, families, grade
from gensokyo.testkit.anchors import Sample, collapse_pairs, run

REPO_ROOT = Path(__file__).resolve().parents[2]

COLLAPSE_TOP = 8
"""塌缩那一节只印最严重的几对。全印是 21 对（灵梦一个人就 7 个锚点），
而这一节要回答的是「有没有一对塌了」，不是列出全矩阵。"""


def _phrasing_section(samples: list[Sample], ran: set[str]) -> list[str]:
    """同一件事的几种问法横着比。

    坑 #30 的教训是「一个比率可能是问法的性质，而不是她的性质」。那一次是
    整局探针的问句自带答案；在锚点上同样的风险是「换个说法数字就变了」。
    **两种问法的区间不重叠时，那个数字不能单独引用。**
    """
    lines = [
        "## 换一种问法",
        "",
        "同一个状态、同一套判据，只换问法。**区间不重叠的那一档说明这个数字是"
        "问法的性质，不是她的性质**——坑 #30 那条教训的推广。",
        "",
    ]
    shaky: list[str] = []
    for base, ids in sorted(families().items()):
        present = [i for i in ids if i in ran and grade(samples, i)]
        if len(present) < 2:
            continue
        graded = {i: grade(samples, i) for i in present}
        lines += [f"### {base}", "", "| 档位 | " + " | ".join(present) + " |"]
        lines.append("|" + "---|" * (len(present) + 1))
        for label in graded[base]:
            rates = [graded[i][label] for i in present]
            split = any(a.separated_from(b) for a in rates for b in rates)
            mark = " ⚠️" if split else ""
            lines.append(f"| {label}{mark} | " + " | ".join(str(r) for r in rates) + " |")
            if split:
                shaky.append(f"{base} · {label}")
        lines += ["", *(f"- `{i}`：{BY_ID[i].question}" for i in present), ""]

    if shaky:
        lines += [
            "**这些档位对问法敏感，不能单独引用**：" + "、".join(shaky) + "。",
            "",
        ]
    else:
        lines += ["所有档位在各种问法下区间都重叠——这些数字可以按「她的性质」读。", ""]
    return lines


def _collapse_section(samples: list[Sample]) -> list[str]:
    """跨锚点塌缩：**问两个不同问题，她给出同一句话的概率。**

    单锚点的判据看不见这件事——那一句在它自己的上下文里完全合理。坑 #27
    实测 87.5% 的复读正是这个形态，而它在按锚点分开报的表里一个格子都不会红。
    """
    pairs = sorted(collapse_pairs(samples).items(), key=lambda kv: -kv[1].rate)
    hit = [(k, v) for k, v in pairs if v.hits]
    for (a, b), rate in hit[:3]:
        print(f"── 塌缩 {a} × {b}   {rate}")
    lines = [
        "## 跨锚点塌缩",
        "",
        "**问两个不同问题，她给出同一句话的概率。** 单个锚点的判据看不见这件事——"
        "那一句在它自己的上下文里完全合理。坑 #27 实测 87.5% 的复读是这个形态。",
        "",
        f"配对按序号对齐，每个样本只用一次，所以区间成立。共 {len(pairs)} 对，"
        f"其中 {len(hit)} 对出现过塌缩。",
        "",
    ]
    if not hit:
        lines += ["没有任何一对锚点撞出同一句话。", ""]
        return lines
    lines += ["| 锚点对 | 同句率 |", "|---|---|"]
    lines += [f"| {a} × {b} | **{rate}** |" for (a, b), rate in hit[:COLLAPSE_TOP]]
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="锚点探针")
    parser.add_argument("--repeats", type=int, default=30, help="每个锚点采样多少次")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "anchors")
    parser.add_argument("--only", default="", help="逗号分隔的锚点 id，留空跑全部")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    wanted = [a for a in ANCHORS if not args.only or a.id in args.only.split(",")]
    print(f"锚点 {len(wanted)} 个 × {args.repeats} 次 = {len(wanted) * args.repeats} 次模型调用\n")

    result = run(wanted, OpenAiCompatibleClient(), repeats=args.repeats)

    lines: list[str] = ["# 锚点探针报告", ""]
    for anchor in wanted:
        samples = result.of(anchor.id)
        print(f"── {anchor.id}（n={len(samples)}）")
        lines += [f"## {anchor.id}", "", f"> {anchor.note}", "", f"问：{anchor.question}", ""]
        for label, rate in grade(result.samples, anchor.id).items():
            print(f"   {label:<24} {rate}")
            lines.append(f"- {label}：**{rate}**")
        lines += ["", "抽样看到的回答：", ""]
        for sample in samples[:5]:
            lines.append(f"- {sample.utterance}")
        lines.append("")
        print()

    lines += _phrasing_section(result.samples, {a.id for a in wanted})
    lines += _collapse_section(result.samples)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (args.out / "samples.json").write_text(
        json.dumps([s.model_dump() for s in result.samples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"报告已写入 {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
