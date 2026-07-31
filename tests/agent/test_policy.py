import json
from pathlib import Path

import pytest

from gensokyo.agent.npc import HISTORY_WINDOW, NpcAgent
from gensokyo.llm.client import LlmError, Msg, ScriptedLlmClient
from gensokyo.world.engine import WorldEngine
from gensokyo.world.events import EventKind
from gensokyo.world.ids import FactId, ItemId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.rules import bump_emotion
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import ErrorCode

REPO_ROOT = Path(__file__).resolve().parents[2]


def _decide(tool_calls: list[dict[str, object]] | None = None, thought: str = "…") -> str:
    """一条决策阶段的回复。台词不在这里——那是阶段二的事。"""
    return json.dumps({"thought": thought, "tool_calls": tool_calls or []}, ensure_ascii=False)


def _agent(replies: list[str], npc: str = "reimu") -> tuple[NpcAgent, WorldEngine]:
    """replies 按两阶段顺序消费：决策 JSON、台词、（重试时）决策 JSON、台词……"""
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    eng = WorldEngine(build_initial_state(defs), defs)
    agent = NpcAgent(
        card=defs.characters[NpcId(npc)],
        engine=eng,
        llm=ScriptedLlmClient(replies),
    )
    return agent, eng


def _calls(agent: NpcAgent) -> list[list[Msg]]:
    return agent.llm.calls  # type: ignore[attr-defined]


def test_plain_utterance_is_emitted_as_event() -> None:
    agent, eng = _agent([_decide(), "有事说事。"])

    turn = agent.act("喂，在吗")

    assert turn.utterance == "有事说事。"
    assert turn.llm_calls == 2  # 一次决策 + 一次说话
    assert eng.state.event_log[-1].payload["text"] == "有事说事。"


def test_successful_tool_call_changes_world() -> None:
    agent, eng = _agent(
        [_decide([{"tool": "take_item", "args": {"item": "offering_coin"}}]), "拿来吧。"]
    )
    eng.state.player.inventory[ItemId("offering_coin")] = 1

    turn = agent.act("给你钱")

    assert turn.tool_results[0].ok is True
    assert eng.state.npcs[NpcId("reimu")].inventory[ItemId("offering_coin")] == 1


def test_failed_tool_call_triggers_one_retry_with_error_feedback() -> None:
    """第一次 reveal_info 因好感不足失败，错误原因回灌，第二次改成开口要钱。"""
    agent, eng = _agent(
        [
            _decide([{"tool": "reveal_info", "args": {"fact": "barrier_anomaly_time"}}]),
            _decide(),
            "先往赛钱箱里放点东西再说。",
        ]
    )

    turn = agent.act("结界最近有异常吗")

    assert turn.llm_calls == 3  # 两次决策 + 一次说话
    assert turn.utterance == "先往赛钱箱里放点东西再说。"
    assert FactId("barrier_anomaly_time") not in eng.state.player.known_facts
    second_decide_prompt = _calls(agent)[1][-1].content
    assert "上一次" in second_decide_prompt
    assert "reveal_info" in second_decide_prompt


def test_retry_is_capped_at_two_decide_calls() -> None:
    agent, _ = _agent(
        [
            _decide([{"tool": "reveal_info", "args": {"fact": "barrier_anomaly_time"}}]),
            _decide([{"tool": "reveal_info", "args": {"fact": "barrier_anomaly_time"}}]),
            "……哼。",
        ]
    )

    turn = agent.act("快说")

    assert turn.llm_calls == 3
    assert turn.tool_results[-1].ok is False
    assert turn.utterance == "……哼。"


def test_tool_calls_and_results_stay_aligned_across_a_retry() -> None:
    """自愈回合：第一次 reveal_info 被门槛拒了，第二次改成开口要东西。

    tool_calls 若只留最后一次决策，这个回合就会变成「1 个调用、2 个结果」，
    按下标配对的指标会把 take_item 配到 reveal_info 的失败上——自愈率于是
    永远算不对，而这是它唯一发生的地方。
    """
    agent, eng = _agent(
        [
            _decide([{"tool": "reveal_info", "args": {"fact": "barrier_anomaly_time"}}]),
            _decide([{"tool": "take_item", "args": {"item": "offering_coin"}}]),
            "先往赛钱箱里放点东西。",
        ]
    )
    eng.state.player.inventory[ItemId("offering_coin")] = 1

    turn = agent.act("结界最近有异常吗")

    assert [c.tool for c in turn.tool_calls] == ["reveal_info", "take_item"]
    assert [r.ok for r in turn.tool_results] == [False, True]
    assert len(turn.tool_calls) == len(turn.tool_results)


def test_hallucinated_say_never_enters_tool_calls() -> None:
    """决策阶段的 say 被丢弃、不下发引擎，所以也不该在 tool_calls 里
    留下一条没有对应结果的记录——那会让两个列表错位。"""
    agent, _ = _agent(
        [
            _decide([{"tool": "say", "args": {"text": "这句不该出现"}}]),
            "玩家看到的只有这句。",
        ]
    )

    turn = agent.act("喂")

    assert turn.tool_calls == []
    assert turn.tool_results == []


def test_unparseable_decision_falls_back_without_crashing() -> None:
    agent, _ = _agent(["我不知道该怎么回答", _decide(), "……哼。"])

    turn = agent.act("喂")

    assert turn.utterance == "……哼。"
    assert turn.llm_calls == 3


def test_decision_that_never_parses_still_speaks() -> None:
    """两次决策都是废话时，她仍然必须开口——否则玩家屏幕上什么都没有。"""
    agent, eng = _agent(["废话一", "废话二", "……你到底想干什么。"])

    turn = agent.act("喂")

    assert turn.utterance == "……你到底想干什么。"
    assert turn.thought == ""
    assert turn.tool_calls == []
    assert eng.state.event_log[-1].payload["text"] == "……你到底想干什么。"


def test_mode_transition_is_recorded() -> None:
    """把芙兰的兴奋度推过 0.7 再开口，mode_before 必须是 destructive。
    原先只断言 mode_after in {calm, destructive}——那覆盖了全部取值，
    永远不可能失败。"""
    agent, eng = _agent([_decide(), "好玩好玩！"], npc="flandre")
    eng.state.player.location = eng.state.npcs[NpcId("flandre")].location
    flandre = eng.state.npcs[NpcId("flandre")]
    bump_emotion(flandre, eng.defs.characters[NpcId("flandre")], +0.6)
    assert flandre.emotion == pytest.approx(0.8)

    turn = agent.act("我带了个音乐盒给你")

    assert turn.mode_before == "destructive"


def test_calm_flandre_stays_calm_across_a_turn() -> None:
    """与上一个测试配对：不推情绪时两端都必须是 calm。
    两个测试要能互相区分，否则「记录了模式」这件事没有被真的测到。"""
    agent, eng = _agent([_decide(), "好玩好玩！"], npc="flandre")
    eng.state.player.location = eng.state.npcs[NpcId("flandre")].location

    turn = agent.act("我带了个音乐盒给你")

    assert turn.mode_before == "calm"
    assert turn.mode_after == "calm"


def test_denied_tool_error_reaches_the_model() -> None:
    agent, eng = _agent(
        [
            _decide([{"tool": "move", "args": {"to": "forest_of_magic"}}]),
            _decide(),
            "……你帮我去看看吧。",
        ],
        npc="flandre",
    )
    eng.state.player.location = eng.state.npcs[NpcId("flandre")].location

    turn = agent.act("外面开了好多花")

    assert turn.tool_results[0].error_code is ErrorCode.TOOL_DENIED
    assert "禁足" in _calls(agent)[1][-1].content
    assert turn.utterance == "……你帮我去看看吧。"


def test_llm_failure_leaves_no_orphan_line_in_history() -> None:
    """端点超时/限流时 run_turn 会抛异常。若玩家发言已写进 history，
    就会留下一条没人回应的孤立记录，模型恢复后看到的对话是错的。"""
    agent, _ = _agent([])  # 预设回复为空，complete 会抛 LlmError

    with pytest.raises(LlmError):
        agent.act("喂")

    assert agent.history == []


def test_on_chunk_receives_the_utterance_piece_by_piece() -> None:
    """流式没真的生效的话，首字延迟就等于总耗时，这次拆分白做。
    ScriptedLlmClient 逐字符 yield，块数必须 > 1。"""
    agent, _ = _agent([_decide(), "自己想办法去。"])
    seen: list[str] = []

    turn = agent.act("帮我个忙", on_chunk=seen.append)

    assert len(seen) > 1
    assert "".join(seen) == turn.utterance


def test_on_chunk_is_optional() -> None:
    """脚本、测试和存档重放都不需要打字机效果。"""
    agent, _ = _agent([_decide(), "自己想办法去。"])

    turn = agent.act("帮我个忙")

    assert turn.utterance == "自己想办法去。"


def test_speak_prompt_contains_the_tool_outcomes() -> None:
    """本次改造的质量收益：她能看到自己做成没做成再开口，
    而不是盲着说完话才知道 reveal_info 被引擎拒了。"""
    agent, eng = _agent(
        [_decide([{"tool": "take_item", "args": {"item": "offering_coin"}}]), "多谢啦。"]
    )
    eng.state.player.inventory[ItemId("offering_coin")] = 1

    agent.act("给你钱")

    speak_prompt = _calls(agent)[-1][-1].content
    assert "take_item" in speak_prompt
    assert eng.state.event_log[-1].payload["text"] == "多谢啦。"


def test_speak_prompt_reports_tool_failure_so_she_does_not_lie() -> None:
    agent, _ = _agent(
        [
            _decide([{"tool": "reveal_info", "args": {"fact": "barrier_anomaly_time"}}]),
            _decide([{"tool": "reveal_info", "args": {"fact": "barrier_anomaly_time"}}]),
            "先往赛钱箱里放点东西。",
        ]
    )

    agent.act("结界最近有异常吗")

    speak_prompt = _calls(agent)[-1][-1].content
    assert "没做到" in speak_prompt


def test_speak_prompt_carries_the_thought_from_the_decision() -> None:
    agent, _ = _agent([_decide(thought="又是来白拿的"), "有事说事。"])

    agent.act("喂")

    assert "又是来白拿的" in _calls(agent)[-1][-1].content


@pytest.mark.parametrize(
    "raw",
    ['"有事说事。"', "「有事说事。」", "“有事说事。”", "  '有事说事。'  "],
)
def test_wrapping_quotes_are_stripped(raw: str) -> None:
    """小模型爱把台词裹在引号里。原样落屏就是每句话都带一对多余的引号。"""
    agent, eng = _agent([_decide(), raw])

    turn = agent.act("喂")

    assert turn.utterance == "有事说事。"
    assert eng.state.event_log[-1].payload["text"] == "有事说事。"


def test_blank_utterance_falls_back_but_still_speaks() -> None:
    """空台词会 emit 一条空文本事件——日志正常，玩家屏幕空白。"""
    agent, eng = _agent([_decide(), "   \n  "])

    turn = agent.act("喂")

    assert turn.utterance == "……"
    assert eng.state.event_log[-1].payload["text"] == "……"


def test_utterance_of_only_quotes_falls_back() -> None:
    agent, eng = _agent([_decide(), '""'])

    turn = agent.act("喂")

    assert turn.utterance == "……"
    assert eng.state.event_log[-1].payload["text"] == "……"


def test_speak_stage_failure_does_not_escape_after_streaming_started() -> None:
    """台词流到一半端点断掉时，异常不能穿透——屏幕上已经有半句话了，
    再拼一行报错很难看。这里用「预设回复用尽」模拟端点失败。"""
    agent, eng = _agent([_decide()])

    turn = agent.act("喂")

    assert turn.utterance == "……"
    assert eng.state.event_log[-1].payload["text"] == "……"
    assert agent.history[-1].endswith("：……")


def test_decision_phase_does_not_offer_or_execute_say() -> None:
    """说话是第二阶段独占的。决策阶段若也 say 一句，event_log 里会多出
    一条玩家从没看见的台词——而它是唯一真相来源。"""
    agent, eng = _agent(
        [
            _decide([{"tool": "say", "args": {"text": "这句不该出现在日志里"}}]),
            "玩家看到的只有这句。",
        ]
    )

    turn = agent.act("喂")

    utterances = [
        ev.payload["text"] for ev in eng.state.event_log if ev.kind is EventKind.NPC_UTTERANCE
    ]
    assert utterances == ["玩家看到的只有这句。"]
    assert turn.utterance == "玩家看到的只有这句。"

    decide_prompt = agent.llm.calls[0][-1].content  # type: ignore[attr-defined]
    assert "- say（" not in decide_prompt


def _speak_prompt(agent: NpcAgent, turn_index: int) -> str:
    """第 n 个回合的说话 prompt。每回合两次调用：决策、说话。"""
    return agent.llm.calls[turn_index * 2 + 1][-1].content  # type: ignore[attr-defined]


def test_ban_list_dedupes_her_own_lines() -> None:
    """禁语清单必须去重。原先直接取 history 里她最后三句话，那三句本身
    可能是同一句——实测第 6 回合的清单是同一句话列了两遍加一句，于是
    「别再重复」这条约束的示例正在示范复读。"""
    line = "你问这些干嘛。"
    agent, _ = _agent([_decide(), line, _decide(), line, _decide(), "换一句。"])

    agent.act("你是AI吗")
    agent.act("你是AI吗")
    agent.act("你是AI吗")

    assert agent.spoken == [line, "换一句。"]
    assert _speak_prompt(agent, 2).count(f"- {line}") == 1


def test_ban_list_outlives_the_history_window() -> None:
    """实测第 10 回合复读的是第 6 回合那句，而 history 只有 12 条的窗口，
    那句早就滑出去了。禁语清单按整局累积，不从 history 切片。"""
    early = "一开始就说过的话。"
    replies = [_decide(), early]
    for i in range(HISTORY_WINDOW):
        replies += [_decide(), f"中间第{i}句。"]
    replies += [_decide(), "最后一句。"]
    agent, _ = _agent(replies)

    for _ in range(HISTORY_WINDOW + 2):
        agent.act("再问一次")

    last = _speak_prompt(agent, HISTORY_WINDOW + 1)
    assert early not in "\n".join(agent.history[-HISTORY_WINDOW:])
    assert f"- {early}" in last


def test_ban_list_treats_punctuation_only_variants_as_the_same_line() -> None:
    """「你到底想干啥？」和「你到底想干啥。」是同一句。第一份基线把它们
    数成两句不同的话（19 次 + 12 次），于是真实复读率高于测出来的。"""
    replies = [_decide(), "你到底想干啥？", _decide(), "你到底想干啥。", _decide(), "行了。"]
    agent, _ = _agent(replies)

    agent.act("喂")
    agent.act("喂")
    agent.act("喂")

    assert agent.spoken == ["你到底想干啥？", "行了。"]


def test_fallback_utterance_does_not_enter_the_ban_list() -> None:
    """省略号标准化后是空串。让它进清单等于往 prompt 里发一条空禁令。"""
    agent, _ = _agent([_decide(), "", _decide(), "总得说点什么。"])

    agent.act("喂")

    assert agent.spoken == []
