import os
from collections.abc import Iterator
from typing import Any, Protocol, cast

from pydantic import BaseModel


class LlmError(RuntimeError):
    pass


class Msg(BaseModel):
    role: str
    content: str


class LlmClient(Protocol):
    def complete(self, messages: list[Msg], temperature: float = 0.8) -> str: ...
    def stream(self, messages: list[Msg], temperature: float = 0.8) -> Iterator[str]: ...


class ScriptedLlmClient:
    """测试替身：按顺序返回预设回复，并记录收到的 messages。
    让 agent 层的编排逻辑可以脱离真实 LLM 做确定性测试。"""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[Msg]] = []

    def _next_reply(self, messages: list[Msg]) -> str:
        # 存快照而非引用。策略层若复用同一个 messages 列表就地追加，
        # 存引用会让所有 calls[i] 指向同一对象——「第二次调用的 prompt
        # 里含回灌的错误原因」这类断言就会空洞通过。
        self.calls.append(list(messages))
        if not self._replies:
            raise LlmError("脚本化客户端的预设回复已用尽")
        return self._replies.pop(0)

    def complete(self, messages: list[Msg], temperature: float = 0.8) -> str:
        return self._next_reply(messages)

    def stream(self, messages: list[Msg], temperature: float = 0.8) -> Iterator[str]:
        # 逐字符 yield：测试要能区分「真的走了流式路径」和「一次性拿到
        # 整段再假装分块」。取回复要在生成器外面做，否则调用方不迭代
        # 就不会记录 calls，也不会在回复用尽时抛错。
        reply = self._next_reply(messages)

        def chunks() -> Iterator[str]:
            yield from reply

        return chunks()


class OpenAiCompatibleClient:
    """打本地 vLLM 或任意 OpenAI 兼容端点。"""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.model = model or os.environ.get("GENSOKYO_MODEL", "Qwen3-8B")
        self._client = OpenAI(
            base_url=base_url or os.environ.get("GENSOKYO_BASE_URL", "http://localhost:8000/v1"),
            api_key=api_key or os.environ.get("GENSOKYO_API_KEY", "not-needed"),
        )
        self.reasoning = (
            reasoning if reasoning is not None else os.environ.get("GENSOKYO_REASONING", "")
        )

    def _extra(self) -> dict[str, Any]:
        """思考模型会先写几百字推理链才吐出第一个可见字符。实测 qwen3:8b
        在这上面花掉 25 秒里的 23 秒，而首字延迟直接决定玩起来的手感。

        只在显式配置时才发这个参数——严格的端点（如 OpenAI 官方）会因为
        未知字段直接 400，不能无条件带上。
        """
        if not self.reasoning:
            return {}
        return {"extra_body": {"reasoning_effort": self.reasoning}}

    def complete(self, messages: list[Msg], temperature: float = 0.8) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=self._payload(messages),
            temperature=temperature,
            **self._extra(),
        )
        content = resp.choices[0].message.content
        if content is None:
            raise LlmError("模型返回了空内容")
        # **self._extra() 让 SDK 的返回类型退化成 Any，显式收回 str。
        return str(content)

    def stream(self, messages: list[Msg], temperature: float = 0.8) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=self._payload(messages),
            temperature=temperature,
            stream=True,
            **self._extra(),
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece

    @staticmethod
    def _payload(messages: list[Msg]) -> Any:
        # SDK 要求 role 是字面量类型的 TypedDict，而我们的 role 是运行时字符串。
        # 运行时等价，类型上无法收窄，故显式 cast。
        return cast(Any, [{"role": m.role, "content": m.content} for m in messages])
