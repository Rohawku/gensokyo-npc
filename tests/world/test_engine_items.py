from pathlib import Path

from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import ItemId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action, ErrorCode

REPO_ROOT = Path(__file__).resolve().parents[2]
COIN = ItemId("offering_coin")


def _engine() -> WorldEngine:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    eng = WorldEngine(build_initial_state(defs), defs)
    eng.state.player.inventory[COIN] = 2
    return eng


def test_player_gives_item_transfers_and_raises_attitude() -> None:
    eng = _engine()
    reimu = eng.state.npcs[NpcId("reimu")]
    before = reimu.attitude

    result = eng.apply(Action(actor="player", tool="give_item", args={"item": COIN, "count": 2}))

    assert result.ok is True
    assert eng.state.player.inventory.get(COIN, 0) == 0
    assert reimu.inventory[COIN] == 2
    assert reimu.attitude > before
    assert COIN in reimu.received_items


def test_giving_more_than_held_fails() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="player", tool="give_item", args={"item": COIN, "count": 5}))

    assert result.ok is False
    assert result.error_code is ErrorCode.INSUFFICIENT_ITEM
    assert eng.state.player.inventory[COIN] == 2


def test_giving_when_no_npc_present_fails() -> None:
    eng = _engine()
    eng.apply(Action(actor="player", tool="move", args={"to": "human_village"}))

    result = eng.apply(Action(actor="player", tool="give_item", args={"item": COIN}))

    assert result.ok is False
    assert result.error_code is ErrorCode.NOT_CO_LOCATED


def test_npc_takes_item_lowers_attitude_of_that_npc() -> None:
    eng = _engine()
    reimu = eng.state.npcs[NpcId("reimu")]
    before = reimu.attitude

    result = eng.apply(Action(actor="reimu", tool="take_item", args={"item": COIN, "count": 1}))

    assert result.ok is True
    assert reimu.inventory[COIN] == 1
    assert eng.state.player.inventory[COIN] == 1
    assert reimu.attitude < before


def test_npc_gives_item_to_player() -> None:
    eng = _engine()
    reimu = eng.state.npcs[NpcId("reimu")]
    reimu.inventory[ItemId("rare_book")] = 1

    result = eng.apply(Action(actor="reimu", tool="give_item", args={"item": "rare_book"}))

    assert result.ok is True
    assert eng.state.player.inventory[ItemId("rare_book")] == 1
    assert reimu.inventory.get(ItemId("rare_book"), 0) == 0
