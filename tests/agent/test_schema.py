import pytest

from gensokyo.agent.schema import NpcTurn, TurnParseError, parse_decision


def test_parse_plain_json() -> None:
    raw = '{"thought": "又是来白拿的", "tool_calls": []}'

    decision = parse_decision(raw)

    assert decision.thought == "又是来白拿的"
    assert decision.tool_calls == []


def test_parse_json_wrapped_in_code_fence() -> None:
    raw = '```json\n{"thought": "t", "tool_calls": []}\n```'

    decision = parse_decision(raw)

    assert decision.thought == "t"


def test_parse_json_with_surrounding_prose() -> None:
    raw = '好的，这是我的决定：\n{"thought": "t", "tool_calls": []}\n希望有帮助'

    decision = parse_decision(raw)

    assert decision.thought == "t"


def test_parse_tool_calls_into_actions() -> None:
    raw = (
        '{"thought": "收下", "tool_calls": ['
        '{"tool": "reveal_info", "args": {"fact": "barrier_anomaly_time"}}'
        "]}"
    )

    decision = parse_decision(raw)

    assert len(decision.tool_calls) == 1
    assert decision.tool_calls[0].tool == "reveal_info"
    assert decision.tool_calls[0].args == {"fact": "barrier_anomaly_time"}


def test_missing_utterance_is_fine_in_decision_phase() -> None:
    """决策阶段不该说话——台词由第二阶段流式生成。"""
    decision = parse_decision('{"thought": "t", "tool_calls": []}')

    assert decision.thought == "t"


def test_utterance_field_is_ignored_when_the_model_volunteers_one() -> None:
    """小模型会照着旧习惯多吐一个 utterance。多余字段不该让解析失败。"""
    decision = parse_decision('{"thought": "t", "tool_calls": [], "utterance": "多余的话"}')

    assert decision.thought == "t"
    assert decision.tool_calls == []


def test_missing_thought_defaults_to_empty() -> None:
    decision = parse_decision('{"tool_calls": []}')

    assert decision.thought == ""


def test_non_json_raises() -> None:
    with pytest.raises(TurnParseError):
        parse_decision("我不知道该说什么")


def test_unknown_tool_name_raises() -> None:
    raw = '{"thought": "t", "tool_calls": [{"tool": "teleport", "args": {}}]}'

    with pytest.raises(TurnParseError):
        parse_decision(raw)


def test_malformed_tool_call_element_raises() -> None:
    with pytest.raises(TurnParseError):
        parse_decision('{"thought": "t", "tool_calls": ["say"]}')


def test_npc_turn_defaults_observability_fields() -> None:
    turn = NpcTurn(thought="t", utterance="u")

    assert turn.tool_calls == []
    assert turn.tool_results == []
    assert turn.llm_calls == 0


def test_parse_survives_braces_in_surrounding_prose() -> None:
    """中文模型常带 {笑} 这类装饰标记。贪婪匹配会把它们一起吞进来。"""
    raw = '注意{重要}：{"thought": "t", "tool_calls": []} 顺便说下{笑}'

    decision = parse_decision(raw)

    assert decision.thought == "t"


def test_parse_takes_first_object_when_model_emits_two() -> None:
    """有些小模型会先输出草稿再输出终稿。解析失败会稳定浪费一次重试。"""
    raw = '{"thought": "草稿", "tool_calls": []} {"thought": "终稿", "tool_calls": []}'

    decision = parse_decision(raw)

    assert decision.thought == "草稿"


def test_parse_keeps_nested_braces_intact() -> None:
    raw = '{"thought":"t","tool_calls":[{"tool":"say","args":{"text":"a{b}c"}}]}'

    decision = parse_decision(raw)

    assert decision.tool_calls[0].args == {"text": "a{b}c"}


def test_parse_ignores_braces_inside_string_literals() -> None:
    raw = '{"thought": "他说「}」然后走了", "tool_calls": []}'

    decision = parse_decision(raw)

    assert decision.thought == "他说「}」然后走了"
