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


def test_npc_taking_players_item_costs_him_the_item_but_not_her_goodwill() -> None:
    """**她自己的行为不该改「她愿不愿意开口」那个数。**

    原先这里是 `attitude < before`（`ATTITUDE_DELTA["npc_took_item"] = -8`），
    理由是「态度是单一的关系亲疏轴」。实测那是个死循环：芙兰在六个对话回合里
    偷三次，好感从 10 掉到 −2，门槛 12 再也够不到，而她的线索是通关必需的——
    三局里两局卡死。玩家既阻止不了也补不回来。

    惩罚仍然存在，只是全在玩家那一侧：东西真的没了，而它正是别的门槛要用的。
    """
    eng = _engine()
    reimu = eng.state.npcs[NpcId("reimu")]
    before = reimu.attitude

    result = eng.apply(Action(actor="reimu", tool="take_item", args={"item": COIN, "count": 1}))

    assert result.ok is True
    assert reimu.inventory[COIN] == 1
    assert eng.state.player.inventory[COIN] == 1
    assert reimu.attitude == before


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
        assert marisa.mode == resolve_mode(card, marisa.emotion, marisa.mode)
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


# ------------------------------------------------------------------ 同场（多 NPC）


def _two_here(eng: WorldEngine) -> None:
    """把魔理沙也叫到神社。**这条路人类玩家走得到**：把两个能收尾的人都请去
    无缘塚，她们都会 `travel_to` 过来，于是玩家面前站着两个人。"""
    eng.state.npcs[NpcId("marisa")].location = eng.state.player.location


def test_giving_with_two_npcs_present_requires_naming_one() -> None:
    """**同场时不能猜。** 原先目标是 `_npcs_here()[0]`——按 `state.npcs` 的
    插入序取第一个，实际上是字母序：芙兰 < 魔理沙 < 灵梦。于是玩家在灵梦和
    魔理沙面前投赛钱，钱**静悄悄进了魔理沙的口袋**，而魔理沙不吃赛钱。

    错误必须可执行：告诉玩家在场都有谁，否则他只知道「失败了」。
    """
    eng = _engine()
    _two_here(eng)

    result = eng.apply(Action(actor="player", tool="give_item", args={"item": COIN}))

    assert result.ok is False
    assert result.error_code is ErrorCode.BAD_ARGS
    assert "博丽灵梦" in (result.error or "") and "雾雨魔理沙" in (result.error or "")
    assert eng.state.player.inventory[COIN] == 2  # 东西还在，没有被误送


def test_naming_the_target_delivers_it_to_her() -> None:
    eng = _engine()
    _two_here(eng)

    result = eng.apply(Action(actor="player", tool="give_item", args={"item": COIN, "to": "reimu"}))

    assert result.ok is True
    assert eng.state.npcs[NpcId("reimu")].inventory[COIN] == 1
    assert COIN not in eng.state.npcs[NpcId("marisa")].inventory


def test_naming_someone_who_is_not_here_fails_instead_of_falling_back() -> None:
    """写错名字要报错，不能悄悄送给在场的另一个人——那比失败更糟。"""
    eng = _engine()

    result = eng.apply(
        Action(actor="player", tool="give_item", args={"item": COIN, "to": "flandre"})
    )

    assert result.ok is False
    assert result.error_code is ErrorCode.NOT_CO_LOCATED
    assert eng.state.player.inventory[COIN] == 2


def test_one_npc_present_still_needs_no_target() -> None:
    """单人场景不该被这条改动波及——绝大多数回合都是单人场景。"""
    eng = _engine()

    result = eng.apply(Action(actor="player", tool="give_item", args={"item": COIN}))

    assert result.ok is True
    assert eng.state.npcs[NpcId("reimu")].inventory[COIN] == 1
