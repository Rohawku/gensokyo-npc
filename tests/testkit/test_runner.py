import json
from pathlib import Path
from typing import Any

from gensokyo.llm.client import ScriptedLlmClient
from gensokyo.testkit.runner import (
    MISSING_ARG,
    UNKNOWN_COMMAND,
    UNSUPPORTED_COMMAND,
    RunConfig,
    run_episode,
)
from gensokyo.testkit.trajectory import Trajectory
from gensokyo.world.observation import PlayerView

REPO_ROOT = Path(__file__).resolve().parents[2]


def _scripted(replies: list[str]) -> ScriptedLlmClient:
    """NPC 侧脚本：决策 JSON + 说话文本成对给。"""
    out: list[str] = []
    for r in replies:
        out.append('{"thought": "…", "tool_calls": []}')
        out.append(r)
    return ScriptedLlmClient(out * 20)


def _cfg(max_turns: int = 40) -> RunConfig:
    return RunConfig(
        max_turns=max_turns,
        scenario_dir=REPO_ROOT / "scenario",
        characters_dir=REPO_ROOT / "characters",
    )


def _decide(tool: str | None = None, **args: Any) -> str:
    calls = [] if tool is None else [{"tool": tool, "args": args}]
    return json.dumps({"thought": "想想", "tool_calls": calls}, ensure_ascii=False)


def _inputs(traj: Trajectory) -> list[str]:
    return [t.player_input for t in traj.turns]


def _normalized(traj: Trajectory) -> dict[str, Any]:
    """latency_ms 是墙上时钟，两次跑必然不同。比确定性时把它抹掉，
    其余每一个字段都必须逐字相同。"""
    data = traj.model_dump()
    for turn in data["turns"]:
        turn["latency_ms"] = 0
    return data


class _Script:
    """按剧本发固定输入的人格。runner 的行为要能脱离任何具体人格来测。"""

    name = "script"
    llm_calls = 0

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self._i = 0

    def next_input(self, view: PlayerView, last_utterance: str) -> str:
        if self._i >= len(self._lines):
            return ""
        line = self._lines[self._i]
        self._i += 1
        return line


# ------------------------------------------------------------------ 指令解析


def test_unknown_slash_command_is_never_forwarded_to_the_npc() -> None:
    """CLI 的坑 #7：认不出的指令若当台词发出去，NPC 会一本正经地回应
    「/dance」这种字符串。runner 必须和 CLI 一样在这儿刹住。"""
    traj = run_episode(_Script(["/dance 一下"]), ScriptedLlmClient([]), _cfg())

    assert traj.turns[0].kind == "command"
    assert traj.turns[0].command_ok is False
    assert traj.turns[0].command_error_code == UNKNOWN_COMMAND
    assert all(e["kind"] != "player_utterance" for e in traj.event_log)


def test_command_aliases_come_from_the_cli_table() -> None:
    """/move 走的是 CLI 的别名表。runner 自己再实现一套，两边就会分叉，
    而基线测出来的体验就不再是玩家的体验。"""
    traj = run_episode(_Script(["/move 人间之里"]), ScriptedLlmClient([]), _cfg())

    assert traj.turns[0].command_ok is True
    assert traj.turns[0].view_after["location_name"] == "人间之里"


def test_command_without_argument_is_rejected_like_in_the_cli() -> None:
    traj = run_episode(_Script(["/go"]), ScriptedLlmClient([]), _cfg())

    assert traj.turns[0].command_error_code == MISSING_ARG


def test_failed_command_records_the_engine_error_code() -> None:
    """error_code 是机器可读的那一份。指标按它分类，所以必须原样落进轨迹。"""
    traj = run_episode(_Script(["/go 红魔馆地下室"]), ScriptedLlmClient([]), _cfg())

    assert traj.turns[0].command_ok is False
    assert traj.turns[0].command_error_code == "no_such_exit"


def test_save_and_load_commands_do_not_touch_the_disk() -> None:
    """批量跑几十局不该往仓库里撒存档——一局的真相已经是轨迹里的动作日志。"""
    traj = run_episode(_Script(["/save 甲"]), ScriptedLlmClient([]), _cfg())

    assert traj.turns[0].command_error_code == UNSUPPORTED_COMMAND


def test_quit_ends_the_episode() -> None:
    traj = run_episode(_Script(["/look", "/quit", "/go 人间之里"]), ScriptedLlmClient([]), _cfg())

    assert _inputs(traj) == ["/look", "/quit"]
    assert traj.finished is False


def test_max_turns_stops_the_episode() -> None:
    traj = run_episode(_Script(["/look"] * 10), ScriptedLlmClient([]), _cfg(3))

    assert len(traj.turns) == 3


def test_empty_input_ends_the_episode() -> None:
    """人格返回空串表示它没什么可说的了。继续跑只会往轨迹里灌空回合。"""
    traj = run_episode(_Script(["/look"]), ScriptedLlmClient([]), _cfg(5))

    assert len(traj.turns) == 1


# ------------------------------------------------------------------ 记录内容


def test_say_record_carries_speaker_thought_tools_and_cost() -> None:
    llm = ScriptedLlmClient([_decide("ask_player", question="你想干嘛"), "干嘛。"])

    traj = run_episode(_Script(["喂"]), llm, _cfg())

    turn = traj.turns[0]
    assert turn.kind == "say"
    assert turn.npc_id == "reimu"
    assert turn.utterance == "干嘛。"
    assert turn.thought == "想想"
    assert turn.tool_calls == [{"tool": "ask_player", "args": {"question": "你想干嘛"}}]
    assert turn.tool_results == [{"ok": True, "error_code": None, "observation": ""}]
    assert turn.llm_calls == 2  # 决策 + 说话
    assert turn.mode_before == "normal"
    assert turn.mode_after == "normal"
    assert turn.command_ok is None


def test_failed_tool_call_keeps_its_error_code_and_reason() -> None:
    """好感不够时 reveal_info 一定失败。这条失败是 info_leak 指标的反面证据，
    只记「不 ok」不够——得知道是被哪个门槛挡住的。"""
    llm = ScriptedLlmClient([_decide("reveal_info", fact="barrier_anomaly_time"), "你管的太多了。"])

    traj = run_episode(_Script(["那些花是怎么回事"]), llm, _cfg())

    result = traj.turns[0].tool_results[0]
    assert result["ok"] is False
    assert result["error_code"] == "reveal_condition_unmet"
    assert result["observation"]


def test_view_after_is_the_whole_player_view() -> None:
    """指标只该看玩家能看到的东西。view_after 就是那道边界。"""
    traj = run_episode(_Script(["/look"]), ScriptedLlmClient([]), _cfg())

    view = traj.turns[0].view_after
    assert view["quest_stage"] == "S0_UNAWARE"
    assert view["inventory"] == {"赛钱": 8}
    assert [n["npc_id"] for n in view["npcs_here"]] == ["reimu"]


def test_saying_something_where_nobody_listens_still_records_a_turn() -> None:
    """对着空房间说话也要留记录，否则轨迹里会凭空少一个回合。"""
    traj = run_episode(_Script(["/go 人间之里", "有人吗"]), ScriptedLlmClient([]), _cfg())

    assert traj.turns[-1].kind == "say"
    assert traj.turns[-1].npc_id is None
    assert traj.turns[-1].utterance == ""


def test_runner_never_prints(capsys: Any) -> None:
    """它要能被批量调用（A7 的基线是几十局），任何输出都得由调用方决定。"""
    llm = ScriptedLlmClient([_decide(), "干嘛。"])

    run_episode(_Script(["喂", "/look"]), llm, _cfg())

    assert capsys.readouterr().out == ""


# ------------------------------------------------------------------ 可复现


def test_same_seed_runs_twice_produce_identical_trajectories() -> None:
    """确定性人格 + 脚本化模型 = 逐字可复现。做不到的话，指标的任何波动
    都无法归因，回归测试就退化成掷骰子。"""
    lines = ["/give 赛钱", "那些花是怎么回事", "/go 人间之里"]
    replies = [_decide("reveal_info", fact="barrier_anomaly_time"), "你管的太多了。"]

    a = run_episode(_Script(list(lines)), ScriptedLlmClient(list(replies)), _cfg(), seed=7)
    b = run_episode(_Script(list(lines)), ScriptedLlmClient(list(replies)), _cfg(), seed=7)

    assert _normalized(a) == _normalized(b)
    assert a.seed == 7


def test_trajectory_of_a_real_run_survives_a_save_load_round_trip(tmp_path: Path) -> None:
    llm = ScriptedLlmClient([_decide("ask_player", question="干嘛"), "干嘛。"])
    traj = run_episode(_Script(["/give 赛钱", "喂"]), llm, _cfg())
    path = tmp_path / "run.json"

    traj.save(path)

    assert Trajectory.load(path) == traj


def test_action_log_in_the_trajectory_can_rebuild_the_world() -> None:
    """存动作日志而不是最终状态：指标算错了可以重算，不用重跑模型。"""
    from gensokyo.world.engine import WorldEngine
    from gensokyo.world.loader import load_defs
    from gensokyo.world.tools import Action

    llm = ScriptedLlmClient([_decide(), "干嘛。"])
    traj = run_episode(_Script(["/give 赛钱", "喂", "/go 人间之里"]), llm, _cfg())

    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    rebuilt = WorldEngine.replay([Action.model_validate(a) for a in traj.action_log], defs)

    assert rebuilt.observe_player().location_name == "人间之里"
    assert rebuilt.state.npcs["reimu"].attitude == 6
    assert rebuilt.observe_player().quest_stage == traj.final_stage


def test_event_log_is_json_ready() -> None:
    """Event.kind 是 StrEnum，落盘前必须已经是纯字符串。"""
    llm = ScriptedLlmClient([_decide(), "干嘛。"])
    traj = run_episode(_Script(["喂"]), llm, _cfg())

    dumped = json.dumps(traj.event_log)

    assert '"kind": "player_utterance"' in dumped


def test_persona_llm_calls_are_recorded_separately() -> None:
    """算全局成本时不能只看 NPC 那一侧。套话玩家每回合也要一次调用。"""

    class CountingPersona:
        name = "counting"

        def __init__(self) -> None:
            self.llm_calls = 0

        def next_input(self, view: PlayerView, last_utterance: str) -> str:
            self.llm_calls += 1
            return "喂" if self.llm_calls == 1 else "/quit"

    traj = run_episode(CountingPersona(), _scripted(["随便"]), _cfg(), seed=0)

    assert traj.turns[0].persona_llm_calls == 1
