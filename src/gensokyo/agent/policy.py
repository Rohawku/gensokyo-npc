import time

from gensokyo.agent.prompt import build_messages
from gensokyo.agent.schema import NpcTurn, TurnParseError, parse_npc_turn
from gensokyo.llm.client import LlmClient
from gensokyo.world.defs import CharacterCard
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import NpcId
from gensokyo.world.tools import Action, ActionResult

MAX_LLM_CALLS = 2
FALLBACK_UTTERANCE = "……"


def run_turn(
    card: CharacterCard,
    engine: WorldEngine,
    llm: LlmClient,
    history: list[str],
) -> NpcTurn:
    npc_id = NpcId(card.id)
    started = time.monotonic()
    errors: list[str] = []
    results: list[ActionResult] = []
    turn: NpcTurn | None = None
    calls = 0

    mode_before = engine.state.npcs[npc_id].mode

    while calls < MAX_LLM_CALLS:
        obs = engine.observe(npc_id)
        tools = engine.available_tools(npc_id)
        messages = build_messages(card, obs, history, tools, errors)

        raw = llm.complete(messages)
        calls += 1

        try:
            candidate = parse_npc_turn(raw, actor=str(card.id))
        except TurnParseError as exc:
            errors = [f"上一次输出无法解析（{exc}）。请严格只输出 JSON 对象。"]
            continue

        turn = candidate
        step_errors: list[str] = []
        for call in candidate.tool_calls:
            result = engine.apply(Action(actor=str(card.id), tool=call.tool, args=call.args))
            results.append(result)
            if not result.ok:
                step_errors.append(f"上一次 {call.tool} 失败：{result.error}")

        if not step_errors:
            break
        errors = step_errors

    if turn is None:
        turn = NpcTurn(thought="", utterance=FALLBACK_UTTERANCE)

    engine.apply(Action(actor=str(card.id), tool="say", args={"text": turn.utterance}))

    turn.tool_results = results
    turn.llm_calls = calls
    turn.mode_before = mode_before
    turn.mode_after = engine.state.npcs[npc_id].mode
    turn.latency_ms = int((time.monotonic() - started) * 1000)
    return turn
