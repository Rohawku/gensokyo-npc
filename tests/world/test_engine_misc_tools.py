from pathlib import Path

from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import ItemId, LocationId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action, ErrorCode

REPO_ROOT = Path(__file__).resolve().parents[2]


def _engine() -> WorldEngine:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    return WorldEngine(build_initial_state(defs), defs)


def test_ask_player_emits_event() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="reimu", tool="ask_player", args={"question": "你来干什么"}))

    assert result.ok is True
    assert eng.state.event_log[0].payload["question"] == "你来干什么"


def test_use_spellcard_emits_event() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="reimu", tool="use_spellcard", args={"name": "梦想封印"}))

    assert result.ok is True
    assert eng.state.event_log[0].payload["name"] == "梦想封印"


def test_break_item_denied_in_calm_mode() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="flandre", tool="break_item", args={"item": "old_music_box"}))

    assert result.ok is False
    assert result.error_code is ErrorCode.TOOL_DENIED


def test_break_item_destroys_held_item_in_destructive_mode() -> None:
    eng = _engine()
    flandre = eng.state.npcs[NpcId("flandre")]
    flandre.emotion = 0.9
    flandre.mode = "destructive"
    flandre.inventory[ItemId("old_music_box")] = 1

    result = eng.apply(Action(actor="flandre", tool="break_item", args={"item": "old_music_box"}))

    assert result.ok is True
    assert ItemId("old_music_box") not in flandre.inventory


def test_break_item_fails_when_item_absent() -> None:
    eng = _engine()
    flandre = eng.state.npcs[NpcId("flandre")]
    flandre.emotion = 0.9
    flandre.mode = "destructive"

    result = eng.apply(Action(actor="flandre", tool="break_item", args={"item": "old_music_box"}))

    assert result.ok is False
    assert result.error_code is ErrorCode.INSUFFICIENT_ITEM


def test_player_can_pick_up_location_item_via_take() -> None:
    """玩家在无缘塚捡花——W1 用 take_item 的玩家分支实现地面拾取。"""
    eng = _engine()
    eng.apply(Action(actor="player", tool="move", args={"to": "human_village"}))
    eng.apply(Action(actor="player", tool="move", args={"to": "muenzuka"}))

    result = eng.apply(Action(actor="player", tool="take_item", args={"item": "withered_flower"}))

    assert result.ok is True
    assert eng.state.player.inventory[ItemId("withered_flower")] == 1
    assert eng.state.locations[LocationId("muenzuka")].items == {}
