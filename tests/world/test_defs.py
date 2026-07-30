import pytest
from pydantic import ValidationError

from gensokyo.world.defs import CharacterCard, EmotionMode, FactDef, RevealConditions
from gensokyo.world.ids import FactId, LocationId, NpcId

MINIMAL_CARD: dict[str, object] = {
    "id": "flandre",
    "name": "芙兰朵露·斯卡雷特",
    "home": "scarlet_devil_basement",
    "persona": {"core": "地下室的吸血鬼", "speech": {"style": "幼稚"}},
    "memory": {"lambda_decay": 0.25, "reflection_threshold": 8.0},
    "emotion": {
        "variable": "excitement",
        "initial": 0.2,
        "modes": [
            {"name": "calm", "range": [0.0, 0.7], "tools_deny": ["break_item"]},
            {"name": "destructive", "range": [0.7, 1.0], "tools_allow": ["break_item"]},
        ],
    },
    "tools": {"deny_always": ["move"]},
}


def test_character_card_minimal() -> None:
    card = CharacterCard.model_validate(
        {
            "id": "flandre",
            "name": "芙兰朵露·斯卡雷特",
            "home": "scarlet_devil_basement",
            "persona": {"core": "地下室的吸血鬼", "speech": {"style": "幼稚"}},
            "memory": {"lambda_decay": 0.25, "reflection_threshold": 8.0},
            "emotion": {
                "variable": "excitement",
                "initial": 0.2,
                "modes": [
                    {"name": "calm", "range": [0.0, 0.7], "tools_deny": ["break_item"]},
                    {"name": "destructive", "range": [0.7, 1.0], "tools_allow": ["break_item"]},
                ],
            },
            "tools": {"deny_always": ["move"]},
        }
    )
    assert card.id == NpcId("flandre")
    assert card.home == LocationId("scarlet_devil_basement")
    assert card.tools.deny_always == ["move"]
    assert card.emotion.modes[1].tools_allow == ["break_item"]
    assert card.persona.speech.forbidden_phrases == []


def test_fact_reveal_conditions_default_empty() -> None:
    fact = FactDef.model_validate(
        {
            "id": "barrier_anomaly_time",
            "holder": "reimu",
            "content": "结界三天前异常",
            "is_clue": True,
        }
    )
    assert fact.id == FactId("barrier_anomaly_time")
    assert fact.reveal_conditions == RevealConditions()
    assert fact.is_clue is True


def test_emotion_mode_contains() -> None:
    mode = EmotionMode(name="calm", range=(0.0, 0.7))
    assert mode.contains(0.0) is True
    assert mode.contains(0.5) is True
    assert mode.contains(0.7) is False


def test_typo_in_top_level_field_is_rejected() -> None:
    card = dict(MINIMAL_CARD) | {"behaviour_baseline": {}}

    with pytest.raises(ValidationError, match="behaviour_baseline"):
        CharacterCard.model_validate(card)


def test_typo_in_nested_field_is_rejected() -> None:
    """拼错的 key 若被静默忽略，情绪 gate 会带着默认值上线——
    表现成难查的行为 bug 而不是加载失败。所以必须报错。"""
    card = dict(MINIMAL_CARD) | {
        "emotion": {
            "variable": "excitement",
            "initial": 0.2,
            "modes": [{"name": "destructive", "range": [0.7, 1.0], "tools_alow": ["break_item"]}],
        }
    }

    with pytest.raises(ValidationError, match="tools_alow"):
        CharacterCard.model_validate(card)


def test_typo_in_optional_fact_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="is_cule"):
        FactDef.model_validate({"id": "f", "holder": "reimu", "content": "c", "is_cule": True})
