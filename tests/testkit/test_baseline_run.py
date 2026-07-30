import json
from pathlib import Path
from typing import Any

from gensokyo.llm.client import ScriptedLlmClient
from gensokyo.testkit.personas import ESCORT, QUESTION, STRIKE, HonestPlayer
from gensokyo.testkit.runner import RunConfig, run_episode
from gensokyo.testkit.trajectory import Trajectory

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cfg(max_turns: int = 40) -> RunConfig:
    return RunConfig(
        max_turns=max_turns,
        scenario_dir=REPO_ROOT / "scenario",
        characters_dir=REPO_ROOT / "characters",
    )


def _honest() -> HonestPlayer:
    return HonestPlayer.from_dirs(REPO_ROOT / "scenario", REPO_ROOT / "characters")


def _decide(tool: str, **args: Any) -> str:
    return json.dumps(
        {"thought": "…", "tool_calls": [{"tool": tool, "args": args}]}, ensure_ascii=False
    )


def _clearing_replies() -> list[str]:
    """一次通关所需的全部 NPC 回复：五次对话，每次「决策 JSON + 台词」。

    顺序就是 HonestPlayer 的行程：灵梦给线索 → 魔理沙给线索 → 芙兰给线索
    → 魔理沙被请去无缘塚 → 魔理沙动手。这份脚本只替掉模型，路线是人格
    自己走出来的——所以它验证的是策略，不是脚本。
    """
    return [
        _decide("reveal_info", fact="barrier_anomaly_time"),
        "结界那事……算你问对人了。",
        _decide("reveal_info", fact="flower_magic_composition"),
        "那花里有魔力结晶，就是这样。",
        _decide("reveal_info", fact="ancient_oblivion_memory"),
        "以前也开过一样的花！",
        _decide("travel_to", destination="muenzuka"),
        "走吧，我先到。",
        _decide("use_spellcard", name="恋符「マスタースパーク」"),
        "看好了。",
    ]


def _run() -> Trajectory:
    return run_episode(_honest(), ScriptedLlmClient(_clearing_replies()), _cfg())


def _inputs(traj: Trajectory) -> list[str]:
    return [t.player_input for t in traj.turns]


def test_honest_player_clears_the_game() -> None:
    """A7 的基线：这条脚本必须真的能通关，否则拿它当基线没有意义。

    NPC 侧全部脚本化，玩家侧零 LLM 调用，所以这个测试进 CI 也只花毫秒——
    「游戏还能不能通关」从此有人看守了（见工程日志坑 #6：它曾经不能）。
    """
    traj = _run()

    assert traj.finished is True
    assert traj.ending == "kirisame_burn"
    assert traj.final_stage == "S4_END"
    assert len(traj.turns) == 21


def test_honest_player_opens_with_four_offerings_then_asks() -> None:
    """好感不够就先投赛钱，门槛一开立刻问。少投一次问不出来，多投一次
    则浪费了后面给芙兰的份额——8 枚赛钱没有余量到可以乱花。"""
    traj = _run()

    assert _inputs(traj)[:5] == ["/give 赛钱"] * 4 + [QUESTION]
    assert [t.command_ok for t in traj.turns[:4]] == [True] * 4


def test_honest_player_trades_with_marisa_instead_of_paying_her() -> None:
    """魔理沙的门槛是交易品而不是好感，给她赛钱是白给。"""
    traj = _run()

    said = _inputs(traj)
    assert said.index("/pick 珍稀魔法书") < said.index("/give 珍稀魔法书")
    assert said[said.index("/give 珍稀魔法书") + 1] == QUESTION


def test_honest_player_collects_all_three_clues_and_then_finishes() -> None:
    traj = _run()

    spoke_to = [t.npc_id for t in traj.turns if t.kind == "say"]
    assert spoke_to == ["reimu", "marisa", "flandre", "marisa", "marisa"]

    said = _inputs(traj)
    assert said.index(ESCORT) < said.index("/go 无缘塚") < said.index(STRIKE)
    assert len(traj.turns[-1].view_after["known_facts"]) == 3


def test_honest_player_never_issues_a_command_the_engine_rejects() -> None:
    """一条会通关的路径不该沿途撞墙。撞墙说明导航或前置条件判断错了。"""
    traj = _run()

    assert [t.command_error_code for t in traj.turns if t.kind == "command"] == [None] * 16


def test_the_baseline_run_costs_exactly_two_llm_calls_per_conversation() -> None:
    """人格零调用，模型只花在 NPC 身上：5 次对话 ×（决策 + 说话）。
    玩家侧也用模型的话，同样一局的成本会再涨 50%。"""
    traj = _run()

    assert sum(t.llm_calls for t in traj.turns) == 10


def test_two_runs_with_the_same_seed_are_identical() -> None:
    a = run_episode(_honest(), ScriptedLlmClient(_clearing_replies()), _cfg(), seed=7)
    b = run_episode(_honest(), ScriptedLlmClient(_clearing_replies()), _cfg(), seed=7)

    def normalized(traj: Trajectory) -> dict[str, Any]:
        data = traj.model_dump()
        for turn in data["turns"]:
            turn["latency_ms"] = 0  # 墙上时钟，不参与比对
        return data

    assert normalized(a) == normalized(b)


def test_the_baseline_trajectory_survives_a_save_load_round_trip(tmp_path: Path) -> None:
    traj = _run()
    path = tmp_path / "baseline.json"

    traj.save(path)

    assert Trajectory.load(path) == traj
