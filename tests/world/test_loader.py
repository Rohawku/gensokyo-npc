from pathlib import Path

from gensokyo.world.ids import FactId, LocationId, NpcId
from gensokyo.world.loader import load_defs

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_defs_from_repo() -> None:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")

    assert set(defs.characters) == {NpcId("reimu"), NpcId("marisa"), NpcId("flandre")}
    assert len(defs.locations) == 7
    assert defs.locations[LocationId("muenzuka")].items == {"withered_flower": 1}
    assert defs.characters[NpcId("flandre")].tools.deny_always == ["move"]


def test_clue_facts_are_exactly_three() -> None:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")

    assert defs.clue_facts() == {
        FactId("barrier_anomaly_time"),
        FactId("flower_magic_composition"),
        FactId("ancient_oblivion_memory"),
    }


def test_every_exit_points_to_existing_location() -> None:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")

    for loc in defs.locations.values():
        for exit_id in loc.exits:
            assert exit_id in defs.locations, f"{loc.id} 的出口 {exit_id} 不存在"


def test_every_fact_holder_is_a_known_npc() -> None:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")

    for fact in defs.facts.values():
        assert fact.holder in defs.characters
