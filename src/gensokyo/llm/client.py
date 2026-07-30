import os
from typing import Protocol

from pydantic import BaseModel


class LlmError(RuntimeError):
    pass


class Msg(BaseModel):
    role: str
    content: str


class LlmClient(Protocol):
    def complete(self, messages: list[Msg], temperature: float = 0.8) -> str: ...


class ScriptedLlmClient:
    """测试替身：按顺序返回预设回复，并记录收到的 messages。
    让 agent 层的编排逻辑可以脱离真实 LLM 做确定性测试。"""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[Msg]] = []

    def complete(self, messages: list[Msg], temperature: float = 0.8) -> str:
        # 存快照而非引用。策略层若复用同一个 messages 列表就地追加，
        # 存引用会让所有 calls[i] 指向同一对象——「第二次调用的 prompt
        # 里含回灌的错误原因」这类断言就会空洞通过。
        self.calls.append(list(messages))
        if not self._replies:
            raise LlmError("脚本化客户端的预设回复已用尽")
        return self._replies.pop(0)


class OpenAiCompatibleClient:
    """打本地 vLLM 或任意 OpenAI 兼容端点。"""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.model = model or os.environ.get("GENSOKYO_MODEL", "Qwen3-8B")
        self._client = OpenAI(
            base_url=base_url or os.environ.get("GENSOKYO_BASE_URL", "http://localhost:8000/v1"),
            api_key=api_key or os.environ.get("GENSOKYO_API_KEY", "not-needed"),
        )

    def complete(self, messages: list[Msg], temperature: float = 0.8) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        if content is None:
            raise LlmError("模型返回了空内容")
        return content
