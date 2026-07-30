from pathlib import Path

from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import ItemId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.rules import resolve_mode
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action, ErrorCode

REPO_ROOT = Path(__file__).resolve().parents[2]
COIN = ItemId("offering_coin")


def _engine() -> WorldEngine:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    eng = WorldEngine(build_initial_state(defs), defs)
    eng.state.player.inventory[COIN] = 2
    return eng


def test_player_gives_item_transfers_and_raises_attitude() -> None:
    eng = _engine()
    reimu = eng.state.npcs[NpcId("reimu")]
    before = reimu.attitude

    result = eng.apply(Action(actor="player", tool="give_item", args={"item": COIN, "count": 2}))

    assert result.ok is True
    assert eng.state.player.inventory.get(COIN, 0) == 0
    assert reimu.inventory[COIN] == 2
    assert reimu.attitude > before
    assert COIN in reimu.received_items


def test_giving_more_than_held_fails() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="player", tool="give_item", args={"item": COIN, "count": 5}))

    assert result.ok is False
    assert result.error_code is ErrorCode.INSUFFICIENT_ITEM
    assert eng.state.player.inventory[COIN] == 2


def test_giving_when_no_npc_present_fails() -> None:
    eng = _engine()
    eng.apply(Action(actor="player", tool="move", args={"to": "human_village"}))

    result = eng.apply(Action(actor="player", tool="give_item", args={"item": COIN}))

    assert result.ok is False
    assert result.error_code is ErrorCode.NOT_CO_LOCATED


def test_npc_taking_players_item_worsens_the_relationship() -> None:
    """态度是单一的「关系亲疏」轴。NPC 不问一声就从玩家手里拿走东西，
    关系该变差——ATTITUDE_DELTA["npc_took_item"] 作用在这条 NPC 分支上，
    而不是「玩家拿走 NPC 的东西」（那条路径不存在）。"""
    eng = _engine()
    reimu = eng.state.npcs[NpcId("reimu")]
    before = reimu.attitude

    result = eng.apply(Action(actor="reimu", tool="take_item", args={"item": COIN, "count": 1}))

    assert result.ok is True
    assert reimu.inventory[COIN] == 1
    assert eng.state.player.inventory[COIN] == 1
    assert reimu.attitude < before


def test_npc_gives_item_to_player() -> None:
    eng = _engine()
    reimu = eng.state.npcs[NpcId("reimu")]
    reimu.inventory[ItemId("rare_book")] = 1

    result = eng.apply(Action(actor="reimu", tool="give_item", args={"item": "rare_book"}))

    assert result.ok is True
    assert eng.state.player.inventory[ItemId("rare_book")] == 1
    assert reimu.inventory.get(ItemId("rare_book"), 0) == 0


def test_emotion_and_mode_stay_consistent_through_the_engine() -> None:
    """经由 apply(give_item) 跨越情绪模式边界后，mode 必须跟着 emotion 走。

    test_rules.py 直接调 bump_emotion 验证过这层耦合，但没有测试走过
    完整的引擎路径——一旦某个 _do_* 绕开 bump_emotion 自己改 emotion，
    就会出现「情绪已过阈值、模式还是旧的」。
    """
    eng = _engine()
    eng.apply(Action(actor="player", tool="move", args={"to": "human_village"}))
    eng.apply(Action(actor="player", tool="move", args={"to": "kirisame_magic_shop"}))
    eng.state.player.inventory[COIN] = 4  # fixture 只给 2 枚，跨阈值需要 4 次
    marisa = eng.state.npcs[NpcId("marisa")]
    card = eng.defs.characters[NpcId("marisa")]
    assert marisa.mode == "normal"

    crossed = False
    for _ in range(4):
        eng.apply(Action(actor="player", tool="give_item", args={"item": COIN}))
        assert marisa.mode == resolve_mode(card, marisa.emotion)
        if marisa.mode == "excited":
            crossed = True

    assert crossed, "送四次礼应当把魔理沙的热切度推过 0.75 的阈值"


def test_npc_taking_an_item_still_counts_as_having_received_it() -> None:
    """抢来的也算收到过。不记的话交易门槛会永久打不开而东西已经没了——
    魔理沙的 take_item 行为基线是全场最高的 0.30，她自己抢走珍稀魔法书
    就会把第二条线索锁死。"""
    eng = _engine()
    eng.apply(Action(actor="player", tool="move", args={"to": "human_village"}))
    eng.apply(Action(actor="player", tool="move", args={"to": "kirisame_magic_shop"}))
    eng.apply(Action(actor="player", tool="take_item", args={"item": "rare_book"}))

    eng.apply(Action(actor="marisa", tool="take_item", args={"item": "rare_book"}))

    marisa = eng.state.npcs[NpcId("marisa")]
    assert ItemId("rare_book") in marisa.received_items
    result = eng.apply(
        Action(actor="marisa", tool="reveal_info", args={"fact": "flower_magic_composition"})
    )
    assert result.ok is True
