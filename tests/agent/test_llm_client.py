import pytest

from gensokyo.llm.client import LlmError, Msg, ScriptedLlmClient


def test_scripted_client_returns_queued_replies_in_order() -> None:
    client = ScriptedLlmClient(["第一句", "第二句"])

    assert client.complete([Msg(role="user", content="a")]) == "第一句"
    assert client.complete([Msg(role="user", content="b")]) == "第二句"


def test_scripted_client_records_calls() -> None:
    client = ScriptedLlmClient(["ok"])

    client.complete([Msg(role="system", content="人设"), Msg(role="user", content="喂")])

    assert len(client.calls) == 1
    assert client.calls[0][0].role == "system"


def test_scripted_client_raises_when_exhausted() -> None:
    client = ScriptedLlmClient([])

    with pytest.raises(LlmError):
        client.complete([Msg(role="user", content="a")])


def test_recorded_calls_are_snapshots_not_live_references() -> None:
    """策略层会复用同一个 messages 列表就地追加。若 calls 存引用，
    「第二次调用的 prompt 里含回灌的错误原因」这类断言会空洞通过。"""
    client = ScriptedLlmClient(["一", "二"])
    messages = [Msg(role="system", content="人设")]

    client.complete(messages)
    messages.append(Msg(role="user", content="回灌的错误原因"))
    client.complete(messages)

    assert len(client.calls[0]) == 1
    assert len(client.calls[1]) == 2
    assert client.calls[0] is not messages
    assert "回灌" not in client.calls[0][-1].content
    assert "回灌" in client.calls[1][-1].content
