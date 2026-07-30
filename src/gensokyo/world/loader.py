from pathlib import Path
from typing import Any

import yaml

from gensokyo.world.defs import CharacterCard, FactDef, ItemDef, LocationDef, WorldDefs


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


def load_defs(scenario_dir: Path, characters_dir: Path) -> WorldDefs:
    locations = [LocationDef.model_validate(d) for d in _read_list(scenario_dir / "locations.yaml")]
    items = [ItemDef.model_validate(d) for d in _read_list(scenario_dir / "items.yaml")]
    facts = [FactDef.model_validate(d) for d in _read_list(scenario_dir / "facts.yaml")]

    cards: list[CharacterCard] = []
    for card_path in sorted(characters_dir.glob("*.yaml")):
        cards.append(CharacterCard.model_validate(_read_mapping(card_path)))

    return WorldDefs(
        locations={loc.id: loc for loc in locations},
        items={item.id: item for item in items},
        facts={fact.id: fact for fact in facts},
        characters={card.id: card for card in cards},
    )
