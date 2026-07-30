from pathlib import Path

import pytest

from gensokyo.world.defs import WorldDefs
from gensokyo.world.ids import FactId, LocationId, NpcId
from gensokyo.world.loader import _index, load_defs

REPO_ROOT = Path(__file__).resolve().parents[2]


def _defs() -> WorldDefs:
    return load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")


def test_load_defs_from_repo() -> None:
    defs = _defs()

    assert set(defs.characters) == {NpcId("reimu"), NpcId("marisa"), NpcId("flandre")}
    assert len(defs.locations) == 7
    assert defs.locations[LocationId("muenzuka")].items == {"withered_flower": 1}
    assert defs.characters[NpcId("flandre")].tools.deny_always == ["move"]


def test_clue_facts_are_exactly_three() -> None:
    assert _defs().clue_facts() == {
        FactId("barrier_anomaly_time"),
        FactId("flower_magic_composition"),
        FactId("ancient_oblivion_memory"),
    }


def test_every_exit_points_to_existing_location() -> None:
    defs = _defs()
    known = set(defs.locations)

    dangling = [
        f"{loc.id} → {exit_id}"
        for loc in defs.locations.values()
        for exit_id in loc.exits
        if exit_id not in known
    ]

    assert dangling == []


def test_every_fact_holder_is_a_known_npc() -> None:
    defs = _defs()
    known = set(defs.characters)

    orphans = [f.id for f in defs.facts.values() if f.holder not in known]

    assert orphans == []


def test_every_referenced_item_exists() -> None:
    """地点物品与交易门槛引用的物品必须都有定义，
    否则工具调用会指向不存在的 item id。"""
    defs = _defs()
    known = set(defs.items)

    referenced: set[str] = set()
    for loc in defs.locations.values():
        referenced |= set(loc.items)
    for fact in defs.facts.values():
        referenced |= set(fact.reveal_conditions.traded_item_in)
    for card in defs.characters.values():
        for dormant in card.knowledge.dormant_memories:
            referenced |= set(dormant.trigger_keys)

    assert sorted(referenced - known) == []


def test_every_held_fact_exists() -> None:
    defs = _defs()
    known = set(defs.facts)

    missing = sorted(
        f"{card.id} 声称持有 {fid}"
        for card in defs.characters.values()
        for fid in card.knowledge.holds_facts
        if fid not in known
    )

    assert missing == []


def test_every_home_is_a_real_location() -> None:
    defs = _defs()
    known = set(defs.locations)

    missing = sorted(
        f"{card.id} 的 home {card.home}"
        for card in defs.characters.values()
        if card.home not in known
    )

    assert missing == []


def test_fact_holder_actually_holds_the_fact() -> None:
    """facts.yaml 的 holder 与角色卡的 holds_facts 必须双向一致，
    否则事实存在但没人能说出来。"""
    defs = _defs()

    mismatched = sorted(
        f"{fact.id} 归属 {fact.holder}，但该角色的 holds_facts 里没有它"
        for fact in defs.facts.values()
        if fact.id not in defs.characters[fact.holder].knowledge.holds_facts
    )

    assert mismatched == []


def test_duplicate_ids_are_rejected() -> None:
    class Entry:
        def __init__(self, id: str) -> None:
            self.id = id

    with pytest.raises(ValueError, match="重复 id：dup"):
        _index([Entry("a"), Entry("dup"), Entry("dup")], "测试项")
