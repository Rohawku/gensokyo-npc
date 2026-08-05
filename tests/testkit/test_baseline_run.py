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


def _chat() -> str:
    """门槛还没开时她只能敷衍：一条决策 JSON（什么也不做）加一句台词。

    **这一格是送礼递减改动逼出来的**：以前投四次赛钱就凑够 24，灵梦第一次被问
    就揭示，脚本里根本没有「她还不肯说」这一步。"""
    return json.dumps({"thought": "…", "tool_calls": []}, ensure_ascii=False)


def _clearing_replies() -> list[str]:
    """一次通关所需的全部 NPC 回复：十六次发言。

    **被问的 10 次是「决策 JSON + 台词」两条，主动开口的 6 次只有台词一条**——
    主动开口跳过决策阶段（`speech_only`），她只说话不动世界。

    行程：投第一枚赛钱灵梦搭话 → 闲聊两句后给线索（前两句还没凑够好感）→
    走进魔法店魔理沙搭话、拿她的书她抗议、收下交易品她认账、然后给线索 →
    走进地下室芙兰扑上来、收到赛钱她好奇、闲聊三句后给线索（第四句「外面」
    才凑够）→ 魔理沙被请去无缘塚 → 魔理沙动手。

    这份脚本只替掉模型，**路线是人格自己走出来的**——所以它验证的是策略，
    不是脚本。
    """
    return [
        "赛钱箱总算响了一声。",  # 主动：灵梦收到赛钱
        _chat(),
        "哼，赛钱是赛钱，情报是情报。",
        _chat(),
        "异变？我还没去看呢。",
        _decide("reveal_info", fact="barrier_anomaly_time"),
        "结界那事……算你问对人了。",
        "哟，稀客。",  # 主动：玩家走进魔法店
        "喂，那本书是我的。",  # 主动：玩家拿走她的书
        "算你懂规矩。",  # 主动：玩家把书交给她
        _decide("reveal_info", fact="flower_magic_composition"),
        "那花里有魔力结晶，就是这样。",
        "有人来了！新玩具！",  # 主动：玩家走进地下室
        "这个圆圆的东西是什么？",  # 主动：芙兰收到赛钱
        _chat(),
        "新玩具！陪我玩！",
        _chat(),
        "妖怪？我就是妖怪啊。",
        _chat(),
        "外面……外面是什么样子的？",
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
    assert len(traj.turns) == 28


def test_offerings_stop_paying_off_and_the_player_switches_to_talking() -> None:
    """**这是可玩性改动的核心断言。** 送礼有边际递减（6/3/1 然后 0），所以
    「重复投币到数值够」这条路被掐死了：投到底只有 10，灵梦门槛 16，最后
    一截必须靠聊天。

    玩家投的是四次而不是三次：他不知道递减表，只能看 `attitude` 有没有动，
    所以要花一枚币才发现第四次白给——这一枚的浪费是真实玩家也会犯的，
    留在指令直方图里可见。
    """
    traj = _run()
    said = _inputs(traj)

    assert said[:5] == ["/give 赛钱"] * 4 + [QUESTION]
    attitudes = [panel["attitude"] for t in traj.turns[:4] for panel in t.view_after["npcs_here"]]
    assert attitudes == [6, 9, 10, 10]


def test_talking_about_what_she_cares_about_is_what_opens_the_gate() -> None:
    """灵梦的门槛靠「异变」「妖怪」两个话题（+4 each）从 10 推到 18；
    芙兰的靠「外面」从 10 推到 14。**这三次就是全局 `topic_touched` 的全部**
    ——改动之前它在真实对局里是 0 次。"""
    traj = _run()

    topics = [e["payload"]["topic"] for e in traj.event_log if e["kind"] == "topic_touched"]

    assert topics == ["异变", "妖怪", "外面"]


def test_honest_player_trades_with_marisa_instead_of_paying_her() -> None:
    """魔理沙的门槛是交易品而不是好感，给她赛钱是白给。"""
    traj = _run()

    said = _inputs(traj)
    assert said.index("/pick 珍稀魔法书") < said.index("/give 珍稀魔法书")
    assert said[said.index("/give 珍稀魔法书") + 1] == QUESTION


def test_honest_player_collects_all_three_clues_and_then_finishes() -> None:
    traj = _run()

    spoke_to = [t.npc_id for t in traj.turns if t.kind == "say"]
    assert spoke_to == ["reimu"] * 3 + ["marisa"] + ["flandre"] * 4 + ["marisa"] * 2

    said = _inputs(traj)
    assert said.index(ESCORT) < said.index("/go 无缘塚") < said.index(STRIKE)
    assert len(traj.turns[-1].view_after["known_facts"]) == 3


def test_honest_player_never_issues_a_command_the_engine_rejects() -> None:
    """一条会通关的路径不该沿途撞墙。撞墙说明导航或前置条件判断错了。"""
    traj = _run()

    assert [t.command_error_code for t in traj.turns if t.kind == "command"] == [None] * 18


def test_the_baseline_run_costs_exactly_two_llm_calls_per_conversation() -> None:
    """人格零调用，模型只花在 NPC 身上：10 次问答 ×（决策 + 说话）+ 6 次主动
    开口 × 只说话 = 26。玩家侧也用模型的话，同样一局的成本会再涨 50%。

    **这个数从 10 涨到 26 是可玩性改动的成本**，分两笔：对话回合从 5 涨到 10
    （送礼递减，门槛的最后一截要靠聊天）= +10，主动开口 6 次 × 1 = +6。
    主动开口刻意只花一次调用——它不做决策，所以那次调用本来也没有用处。
    """
    traj = _run()

    assert sum(t.llm_calls for t in traj.turns) == 26


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


def test_she_speaks_up_on_command_turns_too() -> None:
    """**指令回合曾经全程沉默。** 走进神社、投币、从她店里拿走一本书，一句
    反应都没有——一局 18 个指令回合与 LLM 完全无关，而这是个 LLM 驱动的
    对话游戏。

    每个 NPC 对每种动作只开口一次，所以是 6 次而不是 18 次：灵梦（收礼）、
    魔理沙（进店 / 被拿书 / 收礼）、芙兰（进门 / 收礼）。上限不是为了省钱，
    是为了不复读——第二次投币她已经表过态了。
    """
    traj = _run()

    volunteered = [(t.npc_id, t.player_input.split()[0]) for t in traj.turns if t.volunteered]

    assert volunteered == [
        ("reimu", "/give"),
        ("marisa", "/go"),
        ("marisa", "/pick"),
        ("marisa", "/give"),
        ("flandre", "/go"),
        ("flandre", "/give"),
    ]
    assert all(t.kind == "command" for t in traj.turns if t.volunteered)


def test_repeating_the_same_command_does_not_get_her_talking_again() -> None:
    """投第二枚赛钱她不再开口。放开这个上限等于让复读率上升——而复读率是
    硬指标里守得最紧的一项，不该被一个体验改动悄悄推高。"""
    traj = _run()
    gives_to_reimu = [t for t in traj.turns[:4] if t.player_input == "/give 赛钱"]

    assert len(gives_to_reimu) == 4
    assert [t.volunteered for t in gives_to_reimu] == [True, False, False, False]
