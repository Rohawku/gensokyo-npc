import json
from pathlib import Path

import pytest

from gensokyo.agent.npc import NpcAgent
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
