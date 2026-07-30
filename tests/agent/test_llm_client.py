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
