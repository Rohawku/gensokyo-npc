"""评测入口：跑指定人格各 N 局，落盘每条轨迹与一份 markdown 报告。

    python -m gensokyo.evalcli --episodes 2 --persona honest,jailbreak,fickle --out reports/

轨迹和报告分开落盘是刻意的：指标定义改了之后，拿 `reports/trajectories/`
里的 JSON 重算一遍就行，不用重跑模型。这也是 `Trajectory` 里存 `action_log`
的原因——一条轨迹能被 `WorldEngine.replay` 完整重建。
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from gensokyo.cli import REPO_ROOT, load_dotenv
from gensokyo.llm.client import LlmClient, OpenAiCompatibleClient
from gensokyo.testkit.personas import (
    FicklePlayer,
    GameMap,
    HonestPlayer,
    JailbreakPlayer,
    MemoryProbePlayer,
    Persona,
    SmoothTalkerPlayer,
)
from gensokyo.testkit.report import EvalReport, evaluate
from gensokyo.testkit.runner import RunConfig, run_episode
from gensokyo.testkit.trajectory import Trajectory
from gensokyo.world.loader import load_defs

PersonaFactory = Callable[[GameMap, LlmClient, int], Persona]

PERSONA_FACTORIES: dict[str, PersonaFactory] = {
    "honest": lambda game_map, _llm, seed: HonestPlayer(game_map, seed),
    "jailbreak": lambda _map, _llm, seed: JailbreakPlayer(seed),
    "fickle": lambda _map, _llm, seed: FicklePlayer(seed),
    "memory_probe": lambda game_map, _llm, seed: MemoryProbePlayer(game_map, seed),
    "smooth_talker": lambda _map, llm, seed: SmoothTalkerPlayer(llm, seed),
}

DEFAULT_PERSONAS = "honest,jailbreak,fickle"
"""默认不含 smooth_talker：它是唯一需要模型的人格，一局的成本会再涨 50%。"""


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m gensokyo.evalcli", description=__doc__)
    parser.add_argument("--episodes", type=int, default=2, help="每个人格跑几局")
    parser.add_argument("--persona", default=DEFAULT_PERSONAS, help="逗号分隔的人格名")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports", help="落盘目录")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0, help="第一局的种子，之后逐局 +1")
    return parser.parse_args(argv)


def run_batch(
    persona_names: list[str],
    episodes: int,
    llm: LlmClient,
    cfg: RunConfig,
    first_seed: int = 0,
    on_episode: Callable[[Trajectory], None] | None = None,
) -> list[Trajectory]:
    """跑一批对局。

    每局都新建人格实例：`HonestPlayer` 是有状态的（记着攻略过谁、哪个货架
    是空的），复用一个实例会让第二局带着第一局的记忆开跑，那条基线就不是
    「一局从头玩到尾」了。
    """
    game_map = GameMap.load(cfg.scenario_dir, cfg.characters_dir)
    trajectories: list[Trajectory] = []
    for name in persona_names:
        factory = PERSONA_FACTORIES[name]
        for index in range(episodes):
            seed = first_seed + index
            traj = run_episode(factory(game_map, llm, seed), llm, cfg, seed=seed)
            trajectories.append(traj)
            if on_episode is not None:
                on_episode(traj)
    return trajectories


def write_outputs(report: EvalReport, trajectories: list[Trajectory], out: Path) -> Path:
    for traj in trajectories:
        traj.save(out / "trajectories" / f"{traj.persona}-{traj.seed}.json")
    # 报告同时落 markdown 和 json：前者给人读，后者让下一次能做批次间对比。
    (out / "report.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    path = out / "report.md"
    path.write_text(report.to_markdown(), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv(REPO_ROOT / ".env")

    names = [n.strip() for n in args.persona.split(",") if n.strip()]
    unknown = [n for n in names if n not in PERSONA_FACTORIES]
    if unknown:
        known = "、".join(sorted(PERSONA_FACTORIES))
        print(f"没有这些人格：{'、'.join(unknown)}。可用：{known}", file=sys.stderr)
        return 2

    cfg = RunConfig(
        max_turns=args.max_turns,
        scenario_dir=REPO_ROOT / "scenario",
        characters_dir=REPO_ROOT / "characters",
    )
    defs = load_defs(cfg.scenario_dir, cfg.characters_dir)
    llm = OpenAiCompatibleClient()

    total = len(names) * args.episodes
    done = 0

    def progress(traj: Trajectory) -> None:
        nonlocal done
        done += 1
        ending = traj.ending or "未结束"
        print(
            f"[{done}/{total}] {traj.persona} seed={traj.seed} {len(traj.turns)} 条记录 → {ending}",
            flush=True,
        )

    trajectories = run_batch(names, args.episodes, llm, cfg, args.seed, progress)
    report = evaluate(trajectories, defs)
    path = write_outputs(report, trajectories, args.out)
    print(f"\n报告已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
