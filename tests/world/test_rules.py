from pathlib import Path

from gensokyo.world.defs import RevealConditions
from gensokyo.world.ids import ItemId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.rules import (
    apply_emotion_decay,
    bump_attitude,
    bump_emotion,
    can_reveal,
    resolve_mode,
)
from gensokyo.world.state import build_initial_state

REPO_ROOT = Path(__file__).resolve().parents[2]


def _defs():
    return load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")


def test_resolve_mode_picks_bracket() -> None:
    card = _defs().characters[NpcId("flandre")]

    assert resolve_mode(card, 0.0) == "calm"
    assert resolve_mode(card, 0.69) == "calm"
    assert resolve_mode(card, 0.7) == "destructive"
    assert resolve_mode(card, 1.0) == "destructive"


def test_bump_emotion_clamps_and_updates_mode() -> None:
    defs = _defs()
    state = build_initial_state(defs)
    npc = state.npcs[NpcId("flandre")]

    bump_emotion(npc, defs.characters[NpcId("flandre")], 0.9)

    assert npc.emotion == 1.0
    assert npc.mode == "destructive"


def test_apply_emotion_decay_moves_back_to_calm() -> None:
    defs = _defs()
    state = build_initial_state(defs)
    card = defs.characters[NpcId("flandre")]
    npc = state.npcs[NpcId("flandre")]
    bump_emotion(npc, card, 0.55)
    assert npc.mode == "destructive"

    for _ in range(10):
        apply_emotion_decay(npc, card)

    assert npc.emotion < 0.7
    assert npc.mode == "calm"


def test_bump_attitude_clamps_to_range() -> None:
    state = build_initial_state(_defs())
    npc = state.npcs[NpcId("reimu")]

    bump_attitude(npc, 200)
    assert npc.attitude == 100

    bump_attitude(npc, -500)
    assert npc.attitude == -100


def test_can_reveal_attitude_gate() -> None:
    state = build_initial_state(_defs())
    npc = state.npcs[NpcId("reimu")]
    cond = RevealConditions(attitude_gte=40)

    assert can_reveal(npc, cond) is False
    bump_attitude(npc, 40)
    assert can_reveal(npc, cond) is True


def test_can_reveal_trade_gate() -> None:
    state = build_initial_state(_defs())
    npc = state.npcs[NpcId("marisa")]
    cond = RevealConditions(traded_item_in=[ItemId("rare_book"), ItemId("magic_mushroom")])

    assert can_reveal(npc, cond) is False
    npc.received_items.add(ItemId("magic_mushroom"))
    assert can_reveal(npc, cond) is True


def test_can_reveal_with_no_conditions_is_always_true() -> None:
    state = build_initial_state(_defs())
    npc = state.npcs[NpcId("reimu")]

    assert can_reveal(npc, RevealConditions()) is True


def test_entering_a_mode_needs_more_than_the_bare_threshold() -> None:
    """施密特触发器：进入的门槛比裸阈值高一个迟滞带宽。"""
    card = _defs().characters[NpcId("reimu")]

    assert resolve_mode(card, 0.62, "normal") == "normal"
    assert resolve_mode(card, 0.66, "normal") == "irritated"


def test_leaving_a_mode_needs_dropping_below_the_bare_threshold() -> None:
    card = _defs().characters[NpcId("reimu")]

    assert resolve_mode(card, 0.58, "irritated") == "irritated"
    assert resolve_mode(card, 0.54, "irritated") == "normal"


def test_without_a_current_mode_the_bare_threshold_applies() -> None:
    """初始化时还没有「当前模式」，只能按裸阈值算。"""
    card = _defs().characters[NpcId("reimu")]

    assert resolve_mode(card, 0.62) == "irritated"


def test_the_mode_does_not_flicker_when_the_value_oscillates() -> None:
    """复刻实测的抖动：玩家连续搭话时烦躁度每回合 +0.05、回合末衰减 -0.03，
    恰好跨在灵梦 0.6 的门槛上来回。

    没有迟滞时玩家屏幕上会出现「平常的懒散语气」和「不打算再理你了」同框
    ——她在回合内越过门槛触发拒绝，回合末又掉回门槛以下，而面板是在衰减
    之后画的。这条测试锁住的就是那一帧不再出现。
    """
    card = _defs().characters[NpcId("reimu")]
    mode = "normal"
    seen = [mode]
    value = 0.56
    for _ in range(6):
        value += 0.05
        mode = resolve_mode(card, value, mode)
        value -= 0.03
        mode = resolve_mode(card, value, mode)
        seen.append(mode)

    switches = sum(1 for a, b in zip(seen, seen[1:], strict=False) if a != b)
    assert switches == 1, f"模式在阈值上抖动了 {switches} 次：{seen}"
    assert seen[-1] == "irritated"
