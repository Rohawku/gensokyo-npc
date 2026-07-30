import json
from pathlib import Path

from gensokyo.world.tools import Action


def save_actions(path: Path, actions: list[Action]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [action.model_dump() for action in actions]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_actions(path: Path) -> list[Action]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Action.model_validate(item) for item in raw]
