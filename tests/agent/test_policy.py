import json
from pathlib import Path

from gensokyo.agent.npc import NpcAgent
from gensokyo.llm.client import ScriptedLlmClient
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import FactId, ItemId, NpcId
from gensokyo.world.loader import load_defs
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
    agent, eng = _agent([_reply("好玩好玩！")], npc="flandre")
    eng.state.player.location = eng.state.npcs[NpcId("flandre")].location

    turn = agent.act("我带了个音乐盒给你")

    assert turn.mode_before == "calm"
    assert turn.mode_after in {"calm", "destructive"}


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
