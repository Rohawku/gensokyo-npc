import pytest
from pydantic import ValidationError

from gensokyo.world.tools import (
    TOOL_REGISTRY,
    Action,
    ActionResult,
    ErrorCode,
    GiveItemArgs,
    MoveArgs,
    parse_args,
)


def test_registry_contains_expected_tools() -> None:
    assert set(TOOL_REGISTRY) == {
        "say",
        "move",
        "travel_to",
        "give_item",
        "take_item",
        "reveal_info",
        "ask_player",
        "use_spellcard",
        "break_item",
    }


def test_parse_args_returns_typed_model() -> None:
    args = parse_args("move", {"to": "human_village"})

    assert isinstance(args, MoveArgs)
    assert args.to == "human_village"


def test_parse_args_rejects_unknown_tool() -> None:
    with pytest.raises(KeyError):
        parse_args("teleport", {})


def test_parse_args_rejects_bad_payload() -> None:
    with pytest.raises(ValidationError):
        parse_args("give_item", {"count": 1})


def test_give_item_count_defaults_to_one() -> None:
    args = parse_args("give_item", {"item": "offering_coin"})

    assert isinstance(args, GiveItemArgs)
    assert args.count == 1


def test_action_is_frozen() -> None:
    action = Action(actor="player", tool="say", args={"text": "喂"})

    with pytest.raises(ValidationError):
        action.tool = "move"  # type: ignore[misc]


def test_failed_result_helper() -> None:
    result = ActionResult.failed(ErrorCode.TOOL_DENIED, "你被禁足在地下室，无法离开。")

    assert result.ok is False
    assert result.error_code == ErrorCode.TOOL_DENIED
    assert result.events == []
