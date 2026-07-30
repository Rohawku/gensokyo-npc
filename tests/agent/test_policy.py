import json
from pathlib import Path

import pytest

from gensokyo.agent.npc import NpcAgent
from gensokyo.llm.client import LlmError, ScriptedLlmClient
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import FactId, ItemId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.rules import bump_emotion
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import ErrorCode

REPO_ROOT = Path(__file__).resolve().parents[2]


def _reply(utterance: str, tool_calls: list[dict[str, object]] | None = None) -> str:
    return json.dumps(
        {"thought": "…", "tool_calls": tool_calls or [], "utterance": utterance},
        ensure_ascii=False,
    )


def _agent(replies: list[str], npc: str = "reimu") -> tuple[NpcAgent, WorldEngine]:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    eng = WorldEngine(build_initial_state(defs), defs)
    agent = NpcAgent(
        card=defs.characters[NpcId(npc)],
        engine=eng,
        llm=ScriptedLlmClient(replies),
    )
    return agent, eng


def test_plain_utterance_is_emitted_as_event() -> None:
    agent, eng = _agent([_reply("有事说事。")])

    turn = agent.act("喂，在吗")

    assert turn.utterance == "有事说事。"
    assert turn.llm_calls == 1
    assert eng.state.event_log[-1].payload["text"] == "有事说事。"


def test_successful_tool_call_changes_world() -> None:
    agent, eng = _agent(
        [_reply("拿来吧。", [{"tool": "take_item", "args": {"item": "offering_coin"}}])]
    )
    eng.state.player.inventory[ItemId("offering_coin")] = 1

    turn = agent.act("给你钱")

    assert turn.tool_results[0].ok is True
    assert eng.state.npcs[NpcId("reimu")].inventory[ItemId("offering_coin")] == 1


def test_failed_tool_call_triggers_one_retry_with_error_feedback() -> None:
    """第一次 reveal_info 因好感不足失败，错误原因回灌，第二次改成开口要钱。"""
    agent, eng = _agent(
        [
            _reply(
                "告诉你吧。", [{"tool": "reveal_info", "args": {"fact": "barrier_anomaly_time"}}]
            ),
            _reply("先往赛钱箱里放点东西再说。"),
        ]
    )

    turn = agent.act("结界最近有异常吗")

    assert turn.llm_calls == 2
    assert turn.utterance == "先往赛钱箱里放点东西再说。"
    assert FactId("barrier_anomaly_time") not in eng.state.player.known_facts
    second_call_user_msg = agent.llm.calls[1][-1].content  # type: ignore[attr-defined]
    assert "上一次" in second_call_user_msg


def test_retry_is_capped_at_two_llm_calls() -> None:
    agent, _ = _agent(
        [
            _reply("说了。", [{"tool": "reveal_info", "args": {"fact": "barrier_anomaly_time"}}]),
            _reply(
                "再说一次。", [{"tool": "reveal_info", "args": {"fact": "barrier_anomaly_time"}}]
            ),
        ]
    )

    turn = agent.act("快说")

    assert turn.llm_calls == 2
    assert turn.tool_results[-1].ok is False


def test_unparseable_reply_falls_back_without_crashing() -> None:
    agent, _ = _agent(["我不知道该怎么回答", _reply("……哼。")])

    turn = agent.act("喂")

    assert turn.utterance == "……哼。"
    assert turn.llm_calls == 2


def test_mode_transition_is_recorded() -> None:
    """把芙兰的兴奋度推过 0.7 再开口，mode_before 必须是 destructive。
    原先只断言 mode_after in {calm, destructive}——那覆盖了全部取值，
    永远不可能失败。"""
    agent, eng = _agent([_reply("好玩好玩！")], npc="flandre")
    eng.state.player.location = eng.state.npcs[NpcId("flandre")].location
    flandre = eng.state.npcs[NpcId("flandre")]
    bump_emotion(flandre, eng.defs.characters[NpcId("flandre")], +0.6)
    assert flandre.emotion == pytest.approx(0.8)

    turn = agent.act("我带了个音乐盒给你")

    assert turn.mode_before == "destructive"


def test_calm_flandre_stays_calm_across_a_turn() -> None:
    """与上一个测试配对：不推情绪时两端都必须是 calm。
    两个测试要能互相区分，否则「记录了模式」这件事没有被真的测到。"""
    agent, eng = _agent([_reply("好玩好玩！")], npc="flandre")
    eng.state.player.location = eng.state.npcs[NpcId("flandre")].location

    turn = agent.act("我带了个音乐盒给你")

    assert turn.mode_before == "calm"
    assert turn.mode_after == "calm"


def test_denied_tool_error_reaches_the_model() -> None:
    agent, eng = _agent(
        [
            _reply("我出去看看。", [{"tool": "move", "args": {"to": "forest_of_magic"}}]),
            _reply("……你帮我去看看吧。"),
        ],
        npc="flandre",
    )
    eng.state.player.location = eng.state.npcs[NpcId("flandre")].location

    turn = agent.act("外面开了好多花")

    assert turn.tool_results[0].error_code is ErrorCode.TOOL_DENIED
    assert "禁足" in agent.llm.calls[1][-1].content  # type: ignore[attr-defined]
    assert turn.utterance == "……你帮我去看看吧。"


def test_llm_failure_leaves_no_orphan_line_in_history() -> None:
    """端点超时/限流时 run_turn 会抛异常。若玩家发言已写进 history，
    就会留下一条没人回应的孤立记录，模型恢复后看到的对话是错的。"""
    agent, _ = _agent([])  # 预设回复为空，complete 会抛 LlmError

    with pytest.raises(LlmError):
        agent.act("喂")

    assert agent.history == []


def test_empty_utterance_is_retried_then_falls_back() -> None:
    """空 utterance 会 emit 一条空文本事件——日志正常，玩家屏幕空白。"""
    agent, eng = _agent([_reply(""), _reply("……哼。")])

    turn = agent.act("喂")

    assert turn.llm_calls == 2
    assert turn.utterance == "……哼。"
    assert eng.state.event_log[-1].payload["text"] == "……哼。"


def test_persistently_empty_utterance_falls_back_but_still_speaks() -> None:
    agent, eng = _agent([_reply(""), _reply("   ")])

    turn = agent.act("喂")

    assert turn.utterance == "……"
    assert eng.state.event_log[-1].payload["text"] == "……"
