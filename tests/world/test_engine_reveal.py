from pathlib import Path

from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import FactId, ItemId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.rules import bump_attitude
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action, ErrorCode

REPO_ROOT = Path(__file__).resolve().parents[2]
BARRIER = FactId("barrier_anomaly_time")
FLOWER = FactId("flower_magic_composition")


def _engine() -> WorldEngine:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    return WorldEngine(build_initial_state(defs), defs)


def test_reveal_blocked_when_attitude_too_low() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="reimu", tool="reveal_info", args={"fact": BARRIER}))

    assert result.ok is False
    assert result.error_code is ErrorCode.REVEAL_CONDITION_UNMET
    assert BARRIER not in eng.state.player.known_facts


def test_reveal_succeeds_once_attitude_gate_met() -> None:
    eng = _engine()
    bump_attitude(eng.state.npcs[NpcId("reimu")], 40)

    result = eng.apply(Action(actor="reimu", tool="reveal_info", args={"fact": BARRIER}))

    assert result.ok is True
    assert BARRIER in eng.state.player.known_facts
    assert BARRIER in eng.state.npcs[NpcId("reimu")].revealed_facts


def test_reveal_blocked_until_trade_happens() -> None:
    eng = _engine()
    marisa = eng.state.npcs[NpcId("marisa")]

    blocked = eng.apply(Action(actor="marisa", tool="reveal_info", args={"fact": FLOWER}))
    assert blocked.ok is False
    assert blocked.error_code is ErrorCode.REVEAL_CONDITION_UNMET

    marisa.received_items.add(ItemId("rare_book"))
    allowed = eng.apply(Action(actor="marisa", tool="reveal_info", args={"fact": FLOWER}))

    assert allowed.ok is True
    assert FLOWER in eng.state.player.known_facts


def test_npc_cannot_reveal_a_fact_she_does_not_hold() -> None:
    eng = _engine()
    bump_attitude(eng.state.npcs[NpcId("reimu")], 100)

    result = eng.apply(Action(actor="reimu", tool="reveal_info", args={"fact": FLOWER}))

    assert result.ok is False
    assert result.error_code is ErrorCode.NOT_FACT_HOLDER


def test_reveal_carries_fact_content_in_observation() -> None:
    eng = _engine()
    bump_attitude(eng.state.npcs[NpcId("reimu")], 40)

    result = eng.apply(Action(actor="reimu", tool="reveal_info", args={"fact": BARRIER}))

    assert "结界" in result.observation_delta
