import time
from collections.abc import Callable
from dataclasses import dataclass

from gensokyo.agent.prompt import build_decide_messages, build_speak_messages
from gensokyo.agent.schema import Decision, NpcTurn, TurnParseError, parse_decision
from gensokyo.llm.client import LlmClient
from gensokyo.world.defs import CharacterCard
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import NpcId
from gensokyo.world.tools import Action, ActionResult

MAX_LLM_CALLS = 2

SPEAK_TOOLS = frozenset({"say"})
"""决策阶段不提供的动作。说话由第二阶段独占。"""
FALLBACK_UTTERANCE = "……"
_QUOTE_CHARS = "\"'「」“”"

BAN_WINDOW = 24
"""进禁语清单的台词条数上限，取最近的若干条。

这是**prompt 体积的上限，不是记忆的时效**——「这句说过了」不会因为过了
多久而变假。取 24 是为了覆盖一整局：实测 honest 局 21 回合里她只说了 15
句话，所以整局的不重复台词一般装得下。上限只用来兜住一直玩下去的长会话，
否则这一段会无限增长并把场景描述挤到注意力之外。
"""


def _describe(action: Action, result: ActionResult) -> str:
    """把执行结果翻成给模型看的自然语言。她要看着这个开口，
    所以不能出现 ErrorCode 之类的内部标识符。"""
    if not result.ok:
        return f"{action.tool}：没做到——{result.error}"
    if result.observation_delta:
        return f"{action.tool}：{result.observation_delta}"
    return f"{action.tool}：做到了。"


@dataclass
class _Decided:
    """决策阶段的产出。

    `issued` 与 `results` 严格等长且同序——重试时两者一起追加。曾经
    `tool_calls` 只取最后一次决策、`results` 却跨重试累积，于是恰好在
    「第一次失败、第二次改招成功」这个回合里两者错位，而那正是失败自愈
    唯一发生的地方：任何按下标配对的指标都会把成功的调用配到失败的结果上。
    """

    decision: Decision | None
    issued: list[Action]
    results: list[ActionResult]
    outcomes: list[str]
    llm_calls: int


def _decide(
    card: CharacterCard,
    engine: WorldEngine,
    llm: LlmClient,
    history: list[str],
    npc_id: NpcId,
    recalled: list[str],
) -> _Decided:
    """阶段一：想什么、做什么。"""
    errors: list[str] = []
    issued: list[Action] = []
    results: list[ActionResult] = []
    outcomes: list[str] = []
    decision: Decision | None = None
    calls = 0

    while calls < MAX_LLM_CALLS:
        obs = engine.observe(npc_id)
        tools = [t for t in engine.available_tools(npc_id) if t.name not in SPEAK_TOOLS]
        messages = build_decide_messages(card, obs, history, tools, errors, recalled)

        raw = llm.complete(messages)
        calls += 1

        try:
            candidate = parse_decision(raw, actor=str(card.id))
        except TurnParseError as exc:
            errors = [f"上一次输出无法解析（{exc}）。请严格只输出 JSON 对象。"]
            continue

        decision = candidate
        # 每次重试都重算这一轮的结果描述：上一轮失败的动作已经把错误原因
        # 回灌进 prompt 了，再把它留在 outcomes 里会让她为同一次失败道歉两遍。
        outcomes = []
        step_errors: list[str] = []
        for call in candidate.tool_calls:
            if call.tool in SPEAK_TOOLS:
                # 说话是第二阶段的事。决策阶段若也 say 一句，event_log 里会
                # 多出一条玩家从没看见的台词——而它是唯一真相来源。
                # prompt 里已经不提供 say，这里再兜一层防模型幻觉。
                continue
            action = Action(actor=str(card.id), tool=call.tool, args=call.args)
            result = engine.apply(action)
            issued.append(action)
            results.append(result)
            outcomes.append(_describe(action, result))
            if not result.ok:
                step_errors.append(f"上一次 {call.tool} 失败：{result.error}")

        if not step_errors:
            break
        errors = step_errors

    return _Decided(
        decision=decision, issued=issued, results=results, outcomes=outcomes, llm_calls=calls
    )


def _speak(
    card: CharacterCard,
    engine: WorldEngine,
    llm: LlmClient,
    history: list[str],
    npc_id: NpcId,
    thought: str,
    outcomes: list[str],
    spoken: list[str],
    recalled: list[str],
    on_chunk: Callable[[str], None] | None,
) -> str:
    """阶段二：看着实际结果说一句话，逐块流给调用方。

    `spoken` 由 `NpcAgent` 维护：本局说过的台词，已按标准化去重。这里
    刻意不再从 `history` 切片——12 条的窗口只装得下 6 个回合，实测第 10
    回合复读的正是早已滑出窗口的第 6 回合那句；而且切片本身可能全是同
    一句话，于是「别再重复」这条约束的示例正在示范复读。
    """
    messages = build_speak_messages(
        card,
        engine.observe(npc_id),
        history,
        thought,
        outcomes,
        spoken[-BAN_WINDOW:],
        recalled,
    )

    pieces: list[str] = []
    try:
        for chunk in llm.stream(messages):
            pieces.append(chunk)
            if on_chunk is not None:
                on_chunk(chunk)
    except Exception:  # noqa: BLE001
        # 不让异常穿透：屏幕上已经打了半句话，穿透会把它和一行报错
        # 拼在一起。把已经流出去的部分当成她说完的话，空则落到省略号。
        pass

    text = "".join(pieces).strip().strip(_QUOTE_CHARS).strip()
    return text or FALLBACK_UTTERANCE


def run_turn(
    card: CharacterCard,
    engine: WorldEngine,
    llm: LlmClient,
    history: list[str],
    spoken: list[str] | None = None,
    recalled: list[str] | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> NpcTurn:
    """两阶段回合：先决策（短 JSON），再说话（流式散文）。

    合成一次调用时，半个 JSON 没法展示给玩家，本地 8B 模型那十几秒
    就是整段空白；而且她是盲着说话的——工具成没成功还不知道就开口了。"""
    npc_id = NpcId(card.id)
    started = time.monotonic()
    mode_before = engine.state.npcs[npc_id].mode

    decided = _decide(card, engine, llm, history, npc_id, recalled or [])

    utterance = _speak(
        card,
        engine,
        llm,
        history,
        npc_id,
        decided.decision.thought if decided.decision is not None else "",
        decided.outcomes,
        spoken or [],
        recalled or [],
        on_chunk,
    )
    calls = decided.llm_calls + 1

    engine.apply(Action(actor=str(card.id), tool="say", args={"text": utterance}))

    turn = NpcTurn(
        thought=decided.decision.thought if decided.decision is not None else "",
        utterance=utterance,
        tool_calls=decided.issued,
    )
    turn.tool_results = decided.results
    turn.llm_calls = calls
    turn.mode_before = mode_before
    turn.mode_after = engine.state.npcs[npc_id].mode
    turn.latency_ms = int((time.monotonic() - started) * 1000)
    return turn
