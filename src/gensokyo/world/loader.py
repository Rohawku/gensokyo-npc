from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import yaml

from gensokyo.world.defs import CharacterCard, FactDef, ItemDef, LocationDef, WorldDefs


class _HasStrId(Protocol):
    @property
    def id(self) -> str: ...


def _read_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层必须是列表")
    return data


def _read_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层必须是映射")
    return data


def _index[T: _HasStrId](entries: Sequence[T], what: str) -> dict[Any, T]:
    """按 id 建索引，重复 id 报错。

    直接用 dict 推导式是 last-write-wins，重复 id 会静默吞掉一条记录——
    表现成「某个地点凭空消失」这类难查的 bug。
    """
    seen: dict[Any, T] = {}
    for entry in entries:
        if entry.id in seen:
            raise ValueError(f"{what} 出现重复 id：{entry.id}")
        seen[entry.id] = entry
    return seen


def load_defs(scenario_dir: Path, characters_dir: Path) -> WorldDefs:
    locations = [LocationDef.model_validate(d) for d in _read_list(scenario_dir / "locations.yaml")]
    items = [ItemDef.model_validate(d) for d in _read_list(scenario_dir / "items.yaml")]
    facts = [FactDef.model_validate(d) for d in _read_list(scenario_dir / "facts.yaml")]

    cards: list[CharacterCard] = []
    for card_path in sorted(characters_dir.glob("*.yaml")):
        cards.append(CharacterCard.model_validate(_read_mapping(card_path)))

    return WorldDefs(
        locations=_index(locations, "地点"),
        items=_index(items, "物品"),
        facts=_index(facts, "事实"),
        characters=_index(cards, "角色"),
    )
