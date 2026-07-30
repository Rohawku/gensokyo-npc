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


def test_scripted_client_streams_character_by_character() -> None:
    """逐字符 yield 让测试能区分「真的走了流式路径」和「攒齐整段再假装分块」。"""
    client = ScriptedLlmClient(["干嘛"])

    chunks = list(client.stream([Msg(role="user", content="喂")]))

    assert chunks == ["干", "嘛"]


def test_scripted_client_stream_shares_the_reply_queue_with_complete() -> None:
    """两阶段回合是 complete 之后紧跟一次 stream。若两者各走一条队列，
    测试里给的成对回复就会错位。"""
    client = ScriptedLlmClient(["决策", "台词"])

    assert client.complete([Msg(role="user", content="a")]) == "决策"
    assert "".join(client.stream([Msg(role="user", content="b")])) == "台词"


def test_scripted_client_stream_records_calls() -> None:
    client = ScriptedLlmClient(["台词"])

    list(client.stream([Msg(role="system", content="人设"), Msg(role="user", content="喂")]))

    assert len(client.calls) == 1
    assert client.calls[0][0].role == "system"


def test_scripted_client_stream_raises_when_exhausted_before_iteration() -> None:
    """生成器体内抛错的话，调用方不迭代就看不到错误，一次静默的空回复
    会被当成 NPC 真的沉默了。"""
    client = ScriptedLlmClient([])

    with pytest.raises(LlmError):
        client.stream([Msg(role="user", content="a")])
