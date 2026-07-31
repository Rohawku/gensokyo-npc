from pathlib import Path

import pytest
from pydantic import ValidationError

from gensokyo.memory.salience import salience_for
from gensokyo.world.defs import SALIENCE_BASELINE, CharacterCard
from gensokyo.world.ids import NpcId
from gensokyo.world.loader import load_defs

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cards() -> dict[NpcId, CharacterCard]:
    return load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters").characters


def _raw_card() -> dict[str, object]:
    """一张最小可用的角色卡，用来单独测 salience 校验。"""
    return {
        "id": "test_npc",
        "name": "测试角色",
        "home": "hakurei_shrine",
        "persona": {"core": "测试用", "speech": {"style": "普通"}},
        "memory": {
            "lambda_decay": 0.1,
            "salience_multipliers": {"player_gave_item": 1.5},
            "reflection_threshold": 4.0,
        },
        "emotion": {
            "variable": "测试值",
            "modes": [{"name": "normal", "range": [0.0, 1.0]}],
            "event_deltas": {"player_talked": 0.01},
        },
    }


def test_a_misspelled_salience_key_fails_to_load() -> None:
    """dict 字段的**键**拼错不会被 extra="forbid" 拦住。三张角色卡原先写的
    `receive_gift` / `someone_plays_with_me` / `magic_theory` 一个都对不上真实
    事件名，系数于是静默退回 1.0——「芙兰对陪玩敏感」这个人设差异从来没
    生效过。表现是「三个 NPC 的记忆行为一模一样」而不是加载失败。"""
    raw = _raw_card()
    raw["memory"] = {
        "lambda_decay": 0.1,
        "salience_multipliers": {"receive_gift": 2.0},
        "reflection_threshold": 4.0,
    }

    with pytest.raises(ValidationError, match="receive_gift"):
        CharacterCard.model_validate(raw)


def test_the_error_message_lists_the_available_keys() -> None:
    """报错只说「键不认识」的话，写角色卡的人得去翻源码。"""
    raw = _raw_card()
    raw["memory"] = {
        "lambda_decay": 0.1,
        "salience_multipliers": {"magic_theory": 2.0},
        "reflection_threshold": 4.0,
    }

    with pytest.raises(ValidationError, match="player_gave_item"):
        CharacterCard.model_validate(raw)


def test_shipped_cards_only_use_canonical_keys() -> None:
    for npc_id, card in _cards().items():
        unknown = set(card.memory.salience_multipliers) - set(SALIENCE_BASELINE)
        assert unknown == set(), f"{npc_id} 用了未登记的 salience 键：{unknown}"


def test_salience_is_baseline_times_character_factor() -> None:
    cards = _cards()

    # 芙兰的 player_arrived 系数 2.5，基线 0.3 → 0.75。
    assert salience_for(cards[NpcId("flandre")], "player_arrived") == pytest.approx(0.75)
    # 魔理沙没配 player_arrived 之外的默认，闲聊系数 0.5，基线 0.1 → 0.05。
    assert salience_for(cards[NpcId("marisa")], "player_talked") == pytest.approx(0.05)


def test_unconfigured_event_falls_back_to_the_baseline_not_to_zero() -> None:
    """角色卡没提到的事件类型仍应按基线记住，只是没有人格加成。"""
    marisa = _cards()[NpcId("marisa")]

    assert "spellcard_duel" not in marisa.memory.salience_multipliers
    assert salience_for(marisa, "spellcard_duel") == pytest.approx(
        SALIENCE_BASELINE["spellcard_duel"]
    )


def test_an_unregistered_event_type_scores_zero() -> None:
    """没人给过基线说明它还没被设计成「值得记住的事」。凭空给分会让它在
    检索里和真正重要的条目竞争，所以返回 0.0，由调用方决定不写入。"""
    assert salience_for(_cards()[NpcId("reimu")], "someone_sneezed") == 0.0


def test_salience_is_clamped_to_one() -> None:
    """基线 0.9 的符卡决斗配上 2.0 的系数会算出 1.8。检索打分把 salience
    当成 [0,1] 的一路信号，越界会让这一路凭空压过其他三路。"""
    raw = _raw_card()
    raw["memory"] = {
        "lambda_decay": 0.1,
        "salience_multipliers": {"spellcard_duel": 2.0},
        "reflection_threshold": 4.0,
    }
    card = CharacterCard.model_validate(raw)

    assert salience_for(card, "spellcard_duel") == 1.0


def test_forgetting_rates_are_ordered_by_persona() -> None:
    """芙兰忘得最快、魔理沙记得最久，这是差异化遗忘的实现点。数值一旦被
    调平，三个 NPC 的记忆表现就没差别了，而那正是这一层要买的东西。"""
    cards = _cards()
    flandre = cards[NpcId("flandre")].memory.lambda_decay
    reimu = cards[NpcId("reimu")].memory.lambda_decay
    marisa = cards[NpcId("marisa")].memory.lambda_decay

    assert marisa < reimu < flandre
