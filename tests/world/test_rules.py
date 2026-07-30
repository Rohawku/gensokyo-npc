from pathlib import Path

from gensokyo.world.defs import RevealConditions
from gensokyo.world.ids import ItemId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.rules import (
    apply_emotion_decay,
    bump_attitude,
    bump_emotion,
    can_reveal,
    resolve_mode,
)
from gensokyo.world.state import build_initial_state

REPO_ROOT = Path(__file__).resolve().parents[2]


def _defs():
    return load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")


def test_resolve_mode_picks_bracket() -> None:
    card = _defs().characters[NpcId("flandre")]

    assert resolve_mode(card, 0.0) == "calm"
    assert resolve_mode(card, 0.69) == "calm"
    assert resolve_mode(card, 0.7) == "destructive"
    assert resolve_mode(card, 1.0) == "destructive"


def test_bump_emotion_clamps_and_updates_mode() -> None:
    defs = _defs()
    state = build_initial_state(defs)
    npc = state.npcs[NpcId("flandre")]

    bump_emotion(npc, defs.characters[NpcId("flandre")], 0.9)

    assert npc.emotion == 1.0
    assert npc.mode == "destructive"


def test_apply_emotion_decay_moves_back_to_calm() -> None:
    defs = _defs()
    state = build_initial_state(defs)
    card = defs.characters[NpcId("flandre")]
    npc = state.npcs[NpcId("flandre")]
    bump_emotion(npc, card, 0.55)
    assert npc.mode == "destructive"

    for _ in range(10):
        apply_emotion_decay(npc, card)

    assert npc.emotion < 0.7
    assert npc.mode == "calm"


def test_bump_attitude_clamps_to_range() -> None:
    state = build_initial_state(_defs())
    npc = state.npcs[NpcId("reimu")]

    bump_attitude(npc, 200)
    assert npc.attitude == 100

    bump_attitude(npc, -500)
    assert npc.attitude == -100


def test_can_reveal_attitude_gate() -> None:
    state = build_initial_state(_defs())
    npc = state.npcs[NpcId("reimu")]
    cond = RevealConditions(attitude_gte=40)

    assert can_reveal(npc, cond) is False
    bump_attitude(npc, 40)
    assert can_reveal(npc, cond) is True


def test_can_reveal_trade_gate() -> None:
    state = build_initial_state(_defs())
    npc = state.npcs[NpcId("marisa")]
    cond = RevealConditions(traded_item_in=[ItemId("rare_book"), ItemId("magic_mushroom")])

    assert can_reveal(npc, cond) is False
    npc.received_items.add(ItemId("magic_mushroom"))
    assert can_reveal(npc, cond) is True


def test_can_reveal_with_no_conditions_is_always_true() -> None:
    state = build_initial_state(_defs())
    npc = state.npcs[NpcId("reimu")]

    assert can_reveal(npc, RevealConditions()) is True
