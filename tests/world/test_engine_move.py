from pathlib import Path

from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import LocationId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action, ErrorCode

REPO_ROOT = Path(__file__).resolve().parents[2]


def _engine() -> WorldEngine:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    return WorldEngine(build_initial_state(defs), defs)


def test_player_moves_along_exit() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="player", tool="move", args={"to": "human_village"}))

    assert result.ok is True
    assert eng.state.player.location == LocationId("human_village")


def test_player_cannot_move_to_non_adjacent_location() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="player", tool="move", args={"to": "scarlet_devil_basement"}))

    assert result.ok is False
    assert result.error_code is ErrorCode.NO_SUCH_EXIT
    assert eng.state.player.location == LocationId("hakurei_shrine")


def test_flandre_move_is_always_denied_with_in_character_reason() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="flandre", tool="move", args={"to": "forest_of_magic"}))

    assert result.ok is False
    assert result.error_code is ErrorCode.TOOL_DENIED
    assert result.error is not None
    assert "禁足" in result.error
    assert eng.state.npcs[NpcId("flandre")].location == LocationId("scarlet_devil_basement")


def test_available_tools_excludes_denied_always() -> None:
    eng = _engine()

    names = {spec.name for spec in eng.available_tools(NpcId("flandre"))}

    assert "move" not in names
    assert "say" in names


def test_available_tools_reflects_emotion_mode() -> None:
    eng = _engine()
    flandre = eng.state.npcs[NpcId("flandre")]

    calm_names = {spec.name for spec in eng.available_tools(NpcId("flandre"))}
    assert "break_item" not in calm_names
    assert "ask_player" in calm_names

    flandre.emotion = 0.9
    flandre.mode = "destructive"
    hot_names = {spec.name for spec in eng.available_tools(NpcId("flandre"))}

    assert "break_item" in hot_names
    assert "ask_player" not in hot_names
