from pathlib import Path

from gensokyo.world.ids import FactId, LocationId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.state import QuestStage, build_initial_state

REPO_ROOT = Path(__file__).resolve().parents[2]


def _defs():
    return load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")


def test_initial_state_places_npcs_at_home() -> None:
    state = build_initial_state(_defs())

    assert state.npcs[NpcId("reimu")].location == LocationId("hakurei_shrine")
    assert state.npcs[NpcId("flandre")].location == LocationId("scarlet_devil_basement")


def test_initial_state_seeds_npc_facts_and_emotion() -> None:
    state = build_initial_state(_defs())
    flandre = state.npcs[NpcId("flandre")]

    assert flandre.holds_facts == {FactId("ancient_oblivion_memory")}
    assert flandre.emotion_var == "兴奋度"
    assert flandre.emotion == 0.2
    assert flandre.mode == "calm"


def test_initial_state_player_starts_at_shrine_with_coins() -> None:
    state = build_initial_state(_defs())

    assert state.player.location == LocationId("hakurei_shrine")
    assert state.player.inventory == {"offering_coin": 8}
    assert state.player.known_facts == set()
    assert state.quest.stage is QuestStage.S0_UNAWARE
    assert state.event_log == []
    assert state.tick == 0


def test_initial_state_copies_location_items() -> None:
    state = build_initial_state(_defs())

    assert state.locations[LocationId("muenzuka")].items == {"withered_flower": 1}
