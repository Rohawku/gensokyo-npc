import time
from collections.abc import Callable

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


def _describe(action: Action, result: ActionResult) -> str:
    """把执行结果翻成给模型看的自然语言。她要看着这个开口，
    所以不能出现 ErrorCode 之类的内部标识符。"""
    if not result.ok:
        return f"{action.tool}：没做到——{result.error}"
    if result.observation_delta:
        return f"{action.tool}：{result.observation_delta}"
    return f"{action.tool}：做到了。"


def _decide(
    card: CharacterCard,
    engine: WorldEngine,
    llm: LlmClient,
    history: list[str],
    npc_id: NpcId,
) -> tuple[Decision | None, list[ActionResult], list[str], int]:
    """阶段一：想什么、做什么。返回决策、工具结果、给说话阶段看的
    结果描述，以及消耗的 LLM 调用次数。"""
    errors: list[str] = []
    results: list[ActionResult] = []
    outcomes: list[str] = []
    decision: Decision | None = None
    calls = 0

    while calls < MAX_LLM_CALLS:
        obs = engine.observe(npc_id)
        tools = [t for t in engine.available_tools(npc_id) if t.name not in SPEAK_TOOLS]
        messages = build_decide_messages(card, obs, history, tools, errors)

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
            results.append(result)
            outcomes.append(_describe(action, result))
            if not result.ok:
                step_errors.append(f"上一次 {call.tool} 失败：{result.error}")

        if not step_errors:
            break
        errors = step_errors

    return decision, results, outcomes, calls


def _speak(
    card: CharacterCard,
    engine: WorldEngine,
    llm: LlmClient,
    history: list[str],
    npc_id: NpcId,
    thought: str,
    outcomes: list[str],
    on_chunk: Callable[[str], None] | None,
) -> str:
    """阶段二：看着实际结果说一句话，逐块流给调用方。"""
    prefix = f"{card.name}："
    recent_own = [line[len(prefix) :] for line in history if line.startswith(prefix)][-3:]
    messages = build_speak_messages(
        card, engine.observe(npc_id), history, thought, outcomes, recent_own
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
    on_chunk: Callable[[str], None] | None = None,
) -> NpcTurn:
    """两阶段回合：先决策（短 JSON），再说话（流式散文）。

    合成一次调用时，半个 JSON 没法展示给玩家，本地 8B 模型那十几秒
    就是整段空白；而且她是盲着说话的——工具成没成功还不知道就开口了。"""
    npc_id = NpcId(card.id)
    started = time.monotonic()
    mode_before = engine.state.npcs[npc_id].mode

    decision, results, outcomes, calls = _decide(card, engine, llm, history, npc_id)

    utterance = _speak(
        card,
        engine,
        llm,
        history,
        npc_id,
        decision.thought if decision is not None else "",
        outcomes,
        on_chunk,
    )
    calls += 1

    engine.apply(Action(actor=str(card.id), tool="say", args={"text": utterance}))

    turn = NpcTurn(
        thought=decision.thought if decision is not None else "",
        utterance=utterance,
        tool_calls=decision.tool_calls if decision is not None else [],
    )
    turn.tool_results = results
    turn.llm_calls = calls
    turn.mode_before = mode_before
    turn.mode_after = engine.state.npcs[npc_id].mode
    turn.latency_ms = int((time.monotonic() - started) * 1000)
    return turn
