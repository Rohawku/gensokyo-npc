from pathlib import Path

from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import ItemId, LocationId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action, ErrorCode

REPO_ROOT = Path(__file__).resolve().parents[2]


def _engine() -> WorldEngine:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    return WorldEngine(build_initial_state(defs), defs)


def test_ask_player_emits_event() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="reimu", tool="ask_player", args={"question": "你来干什么"}))

    assert result.ok is True
    assert eng.state.event_log[0].payload["question"] == "你来干什么"


def test_use_spellcard_emits_event() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="reimu", tool="use_spellcard", args={"name": "梦想封印"}))

    assert result.ok is True
    assert eng.state.event_log[0].payload["name"] == "梦想封印"


def test_break_item_denied_in_calm_mode() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="flandre", tool="break_item", args={"item": "old_music_box"}))

    assert result.ok is False
    assert result.error_code is ErrorCode.TOOL_DENIED


def test_break_item_destroys_held_item_in_destructive_mode() -> None:
    eng = _engine()
    flandre = eng.state.npcs[NpcId("flandre")]
    flandre.emotion = 0.9
    flandre.mode = "destructive"
    flandre.inventory[ItemId("old_music_box")] = 1

    result = eng.apply(Action(actor="flandre", tool="break_item", args={"item": "old_music_box"}))

    assert result.ok is True
    assert ItemId("old_music_box") not in flandre.inventory


def test_break_item_fails_when_item_absent() -> None:
    eng = _engine()
    flandre = eng.state.npcs[NpcId("flandre")]
    flandre.emotion = 0.9
    flandre.mode = "destructive"

    result = eng.apply(Action(actor="flandre", tool="break_item", args={"item": "old_music_box"}))

    assert result.ok is False
    assert result.error_code is ErrorCode.INSUFFICIENT_ITEM


def test_player_can_pick_up_location_item_via_take() -> None:
    """玩家在无缘塚捡花——W1 用 take_item 的玩家分支实现地面拾取。"""
    eng = _engine()
    eng.apply(Action(actor="player", tool="move", args={"to": "human_village"}))
    eng.apply(Action(actor="player", tool="move", args={"to": "muenzuka"}))

    result = eng.apply(Action(actor="player", tool="take_item", args={"item": "withered_flower"}))

    assert result.ok is True
    assert eng.state.player.inventory[ItemId("withered_flower")] == 1
    assert eng.state.locations[LocationId("muenzuka")].items == {}


def test_break_item_is_restricted_to_flandre() -> None:
    """受限工具不默认发放。灵梦和魔理沙的任何模式都不该出现 break_item——
    否则每新增一个工具就会自动授予所有角色。"""
    eng = _engine()

    for npc_id in (NpcId("reimu"), NpcId("marisa")):
        card = eng.defs.characters[npc_id]
        for mode in card.emotion.modes:
            eng.state.npcs[npc_id].mode = mode.name
            names = {spec.name for spec in eng.available_tools(npc_id)}
            assert "break_item" not in names, f"{npc_id} 在 {mode.name} 模式下拿到了 break_item"


def test_gift_lowers_reimus_annoyance_but_raises_flandres_excitement() -> None:
    """同一事件的情绪方向按角色卡走：灵梦爱钱，收到赛钱该消气；
    芙兰收到东西该更兴奋。用全局表会让灵梦在玩家讨好她时越来越烦。"""
    eng = _engine()
    eng.state.player.inventory[ItemId("offering_coin")] = 4
    reimu_before = eng.state.npcs[NpcId("reimu")].emotion

    eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))

    assert eng.state.npcs[NpcId("reimu")].emotion < reimu_before

    eng.state.player.location = eng.state.npcs[NpcId("flandre")].location
    flandre_before = eng.state.npcs[NpcId("flandre")].emotion

    eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))

    assert eng.state.npcs[NpcId("flandre")].emotion > flandre_before


def test_reimu_stays_out_of_irritated_while_being_paid() -> None:
    """通关必经之路：送 4 次赛钱打开线索门槛。
    这个过程不该把她推进 irritated（那会禁掉 ask_player，恰在给玩家奖励时让对话变差）。"""
    eng = _engine()

    for _ in range(4):
        eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))

    reimu = eng.state.npcs[NpcId("reimu")]
    assert reimu.attitude >= 24
    assert reimu.mode == "normal"


NPC_ONLY_ACTIONS = [
    Action(actor="player", tool="break_item", args={"item": "offering_coin"}),
    Action(actor="player", tool="reveal_info", args={"fact": "barrier_anomaly_time"}),
    Action(actor="player", tool="ask_player", args={"question": "你在吗"}),
    Action(actor="player", tool="use_spellcard", args={"name": "梦想封印"}),
]


def test_player_calling_npc_only_actions_fails_without_raising() -> None:
    """apply() 对任何坏输入都必须返回 ActionResult，不能抛异常——
    否则玩家侧一条走错的指令会掀掉整个进程。break_item 原先直接
    KeyError('player')，因为它按 actor 查 npcs 表而玩家不在表里。"""
    for action in NPC_ONLY_ACTIONS:
        eng = _engine()

        result = eng.apply(action)

        assert result.ok is False, f"玩家调用 {action.tool} 竟然成功了"
        assert result.error_code is ErrorCode.TOOL_DENIED
        assert eng.state.event_log == []
        assert eng.state.seq == 0
