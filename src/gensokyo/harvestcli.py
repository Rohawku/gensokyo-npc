"""偏好数据采集入口。

    python -m gensokyo.harvestcli --trajectories reports/trajectories --out reports/pairs.jsonl

**这个文件存在的第一个理由是让 `training/` 真的被执行一次。** 在它之前，
`label.py` / `preference.py` / `harvest.py` 三个模块没有任何调用者——没有测试、
没有入口，而工程日志把它们记成「已完成」。那正是这个项目自己数出来五次的
类 1 失效模式（写了但从来没在跑），见坑 #32。

采集要重放原局：`harvest_episode` 用轨迹里记下的原输出重建每个回合的世界、
记忆库与禁语清单，然后对**同一个 prompt** 重采样 k 条候选。同 prompt 是 DPO
的硬要求——跨回合配对（这一回合答得好、那一回合答得差）的梯度信号是错的。
而重放做得到，是因为世界与记忆两层都能从动作日志精确重建（取舍 #2、#7）。
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from gensokyo.cli import REPO_ROOT, load_dotenv
from gensokyo.llm.client import LlmClient, OpenAiCompatibleClient
from gensokyo.testkit.trajectory import Trajectory
from gensokyo.training.harvest import SAMPLES_PER_TURN, harvest_episode
from gensokyo.training.preference import Dataset, PreferencePair, assemble, write_jsonl
from gensokyo.world.defs import WorldDefs
from gensokyo.world.loader import load_defs

DEFAULT_SIZE = 200
"""目标数据集大小。取不够会在 `shortfall` 里如实报出来，不悄悄补齐。"""


def build_dataset(
    trajectories: Sequence[Trajectory],
    defs: WorldDefs,
    llm: LlmClient,
    samples: int = SAMPLES_PER_TURN,
    size: int = DEFAULT_SIZE,
) -> Dataset:
    """重放每一局、逐回合采样配对，再按维度配额组装。"""
    pool: list[PreferencePair] = []
    for traj in trajectories:
        pool += harvest_episode(traj, defs, llm, samples=samples)
    return assemble(pool, size)


def _load_trajectories(directory: Path) -> list[Trajectory]:
    return [
        Trajectory.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m gensokyo.harvestcli", description=__doc__)
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=REPO_ROOT / "reports" / "trajectories",
        help="`make eval` 落盘的轨迹目录",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "pairs.jsonl")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_TURN, help="每个回合采几条候选")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="目标数据集大小")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, llm: LlmClient | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv(REPO_ROOT / ".env")

    trajectories = _load_trajectories(args.trajectories) if args.trajectories.is_dir() else []
    if not trajectories:
        print(
            f"{args.trajectories} 里没有轨迹。先跑 `make eval` 产出对局。",
            file=sys.stderr,
        )
        return 2

    turns = sum(1 for t in trajectories for r in t.turns if r.npc_id and r.utterance)
    calls = turns * args.samples
    print(
        f"{len(trajectories)} 局、{turns} 个说话回合 × {args.samples} 条候选 = 约 {calls} 次调用\n"
    )

    dataset = build_dataset(
        trajectories,
        load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters"),
        llm or OpenAiCompatibleClient(),
        samples=args.samples,
        size=args.size,
    )
    written = write_jsonl(dataset, args.out)

    print(f"配对 {written} 条 → {args.out}")
    for dimension, count in sorted(dataset.counts().items()):
        print(f"   {dimension:<14} {count}")
    # 缺口必须印出来：空字典才是达标，而「悄悄补齐」会让配额那句话变成半真的。
    print(f"\n距配额的缺口：{dataset.shortfall or '无'}")
    print(
        f"叙事维度整体缺失：{dataset.narrative_share_missing:.0%}（要 judge，而 judge 未过 κ 门槛）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
