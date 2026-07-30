import pytest

from gensokyo.agent.schema import NpcTurn, TurnParseError, parse_npc_turn


def test_parse_plain_json() -> None:
    raw = '{"thought": "又是来白拿的", "tool_calls": [], "utterance": "有事说事。"}'

    turn = parse_npc_turn(raw)

    assert turn.thought == "又是来白拿的"
    assert turn.utterance == "有事说事。"
    assert turn.tool_calls == []


def test_parse_json_wrapped_in_code_fence() -> None:
    raw = '```json\n{"thought": "t", "tool_calls": [], "utterance": "u"}\n```'

    turn = parse_npc_turn(raw)

    assert turn.utterance == "u"


def test_parse_json_with_surrounding_prose() -> None:
    raw = '好的，这是我的回应：\n{"thought": "t", "tool_calls": [], "utterance": "u"}\n希望有帮助'

    turn = parse_npc_turn(raw)

    assert turn.thought == "t"


def test_parse_tool_calls_into_actions() -> None:
    raw = (
        '{"thought": "收下", "tool_calls": ['
        '{"tool": "reveal_info", "args": {"fact": "barrier_anomaly_time"}}'
        '], "utterance": "行吧，告诉你。"}'
    )

    turn = parse_npc_turn(raw)

    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].tool == "reveal_info"
    assert turn.tool_calls[0].args == {"fact": "barrier_anomaly_time"}


def test_missing_utterance_raises() -> None:
    with pytest.raises(TurnParseError):
        parse_npc_turn('{"thought": "t", "tool_calls": []}')


def test_non_json_raises() -> None:
    with pytest.raises(TurnParseError):
        parse_npc_turn("我不知道该说什么")


def test_unknown_tool_name_raises() -> None:
    raw = '{"thought": "t", "tool_calls": [{"tool": "teleport", "args": {}}], "utterance": "u"}'

    with pytest.raises(TurnParseError):
        parse_npc_turn(raw)


def test_npc_turn_defaults_observability_fields() -> None:
    turn = NpcTurn(thought="t", utterance="u")

    assert turn.tool_calls == []
    assert turn.tool_results == []
    assert turn.llm_calls == 0
