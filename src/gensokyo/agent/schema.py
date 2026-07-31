import json
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from gensokyo.world.tools import TOOL_REGISTRY, Action, ActionResult

_TRAILING = " \t\r\n。，、！？…~～.,!?;:；：\"'「」『』（）()"


def normalize_utterance(text: str) -> str:
    """用于「这句是不是说过了」的比较。

    只差一个标点的两句话是同一句：实测报告里「你到底想干啥？」19 次、
    「你到底想干啥。」12 次被算成两句不同的话，于是真实复读率比测出来的高。
    """
    return text.strip().strip(_TRAILING).replace(" ", "")


class TurnParseError(ValueError):
    pass


class NpcTurn(BaseModel):
    thought: str
    utterance: str
    tool_calls: list[Action] = Field(default_factory=list)
    """本回合实际下发给引擎的调用，含重试时那次失败的，与 tool_results
    严格等长同序。刻意不是「最后一次决策说要做什么」——那个版本会在自愈
    回合里丢掉失败的那一次，让任何按下标配对的指标把成功配到失败上。"""

    # 以下为可观测性字段，不参与玩法，供调试面板使用。
    # 没有它们，一次糟糕的回复无法归因到检索还是生成。
    tool_results: list[ActionResult] = Field(default_factory=list)
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    mode_before: str = ""
    mode_after: str = ""
    llm_calls: int = 0
    latency_ms: int = 0


class Decision(BaseModel):
    """决策阶段的输出契约：只有想法和动作，没有台词。

    说话拆到第二阶段之后，第一次调用的输出变得很短，首字延迟从
    「整段 JSON 生成完」降到「几十个 token」；顺带让 NPC 能先看到
    工具执行的实际结果再开口，而不是盲着说话。"""

    thought: str = ""
    tool_calls: list[Action] = Field(default_factory=list)


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


def parse_decision(raw: str, actor: str = "") -> Decision:
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
        return Decision(thought=str(data.get("thought", "")), tool_calls=calls)
    except ValidationError as exc:
        raise TurnParseError(f"字段校验失败：{exc}") from exc
