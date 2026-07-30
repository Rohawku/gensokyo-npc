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


def test_repeated_reveal_does_not_flood_event_log() -> None:
    """重复揭示不算失败（把它当失败会给策略层语义错乱的信号），
    但不再产生事件，否则同质条目会挤占 prompt 的近期事件窗口。"""
    eng = _engine()
    bump_attitude(eng.state.npcs[NpcId("reimu")], 24)

    first = eng.apply(Action(actor="reimu", tool="reveal_info", args={"fact": BARRIER}))
    log_after_first = len(eng.state.event_log)
    second = eng.apply(Action(actor="reimu", tool="reveal_info", args={"fact": BARRIER}))

    assert first.ok is True
    assert second.ok is True
    assert second.events == []
    assert len(eng.state.event_log) == log_after_first
    assert "已经告诉过" in second.observation_delta


def test_four_gifts_exactly_unlock_reimus_clue() -> None:
    """锁住游戏可通关性：灵梦的门槛必须能被初始赛钱打开。
    门槛与 ATTITUDE_DELTA 的比例一旦改动，这个测试会立刻报警。"""
    eng = _engine()

    for _ in range(4):
        assert (
            eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"})).ok
            is True
        )

    assert eng.apply(Action(actor="reimu", tool="reveal_info", args={"fact": BARRIER})).ok is True
    assert BARRIER in eng.state.player.known_facts
