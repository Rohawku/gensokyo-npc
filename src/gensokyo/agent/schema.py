import json
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from gensokyo.world.tools import TOOL_REGISTRY, Action, ActionResult


class TurnParseError(ValueError):
    pass


class NpcTurn(BaseModel):
    thought: str
    utterance: str
    tool_calls: list[Action] = Field(default_factory=list)

    # 以下为可观测性字段，不参与玩法，供调试面板使用。
    # 没有它们，一次糟糕的回复无法归因到检索还是生成。
    tool_results: list[ActionResult] = Field(default_factory=list)
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    mode_before: str = ""
    mode_after: str = ""
    llm_calls: int = 0
    latency_ms: int = 0


def _balanced_objects(raw: str) -> Iterator[str]:
    """依次产出每一段花括号配平的顶层片段。

    贪婪正则 `\\{.*\\}` 会从第一个左括号一直吞到最后一个右括号，于是
    「前后废话里的 {笑} 这类装饰标记」和「模型并排输出两个对象」都会
    让解析失败。配平扫描则能逐个给出候选，由调用方挑第一个真正是
    JSON 对象的。字符串字面量内的括号不计数。
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, ch in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                yield raw[start : i + 1]


def _extract_turn_payload(raw: str) -> dict[str, Any]:
    last_error: str | None = None
    for candidate in _balanced_objects(raw):
        try:
            parsed: Any = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if isinstance(parsed, dict):
            return parsed
    if last_error is not None:
        raise TurnParseError(f"JSON 解析失败：{last_error}")
    raise TurnParseError(f"回复里找不到 JSON 对象：{raw[:120]!r}")


def parse_npc_turn(raw: str, actor: str = "") -> NpcTurn:
    data = _extract_turn_payload(raw)

    calls: list[Action] = []
    for item in data.get("tool_calls") or []:
        if not isinstance(item, dict) or "tool" not in item:
            raise TurnParseError(f"tool_calls 元素格式不对：{item!r}")
        name = str(item["tool"])
        if name not in TOOL_REGISTRY:
            raise TurnParseError(f"未知工具：{name}")
        calls.append(
            Action(
                actor=actor or str(item.get("actor", "")), tool=name, args=item.get("args") or {}
            )
        )

    try:
        turn = NpcTurn(
            thought=str(data.get("thought", "")),
            utterance=data["utterance"],
            tool_calls=calls,
        )
    except KeyError as exc:
        raise TurnParseError("回复缺少 utterance 字段") from exc
    except ValidationError as exc:
        raise TurnParseError(f"字段校验失败：{exc}") from exc

    if not turn.utterance.strip():
        # 空 utterance 会 emit 一条空文本事件：玩家屏幕上什么都没有，
        # 但日志里看起来一切正常。当作解析失败，让策略层重试一次。
        raise TurnParseError("utterance 是空的，NPC 必须说出一句话")
    return turn
