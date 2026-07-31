from pathlib import Path

from gensokyo.world.engine import WorldEngine
from gensokyo.world.events import EventKind
from gensokyo.world.ids import NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action, ErrorCode

REPO_ROOT = Path(__file__).resolve().parents[2]


def _engine() -> WorldEngine:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    return WorldEngine(build_initial_state(defs), defs)


def test_player_say_appends_event() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="player", tool="say", args={"text": "有人吗"}))

    assert result.ok is True
    assert len(eng.state.event_log) == 1
    ev = eng.state.event_log[0]
    assert ev.kind is EventKind.PLAYER_UTTERANCE
    assert ev.payload == {"text": "有人吗"}
    assert ev.location == eng.state.player.location


def test_npc_say_uses_npc_kind_and_location() -> None:
    eng = _engine()

    eng.apply(Action(actor="flandre", tool="say", args={"text": "新玩具"}))

    ev = eng.state.event_log[0]
    assert ev.kind is EventKind.NPC_UTTERANCE
    assert ev.actor == "flandre"
    assert ev.location == eng.state.npcs[NpcId("flandre")].location


def test_event_ids_are_deterministic_and_sequential() -> None:
    eng = _engine()

    eng.apply(Action(actor="player", tool="say", args={"text": "一"}))
    eng.apply(Action(actor="player", tool="say", args={"text": "二"}))

    assert [ev.id for ev in eng.state.event_log] == ["e00001", "e00002"]


def test_unknown_tool_is_rejected() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="player", tool="teleport", args={}))

    assert result.ok is False
    assert result.error_code is ErrorCode.UNKNOWN_TOOL
    assert eng.state.event_log == []


def test_bad_args_is_rejected_without_mutating_log() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="player", tool="say", args={}))

    assert result.ok is False
    assert result.error_code is ErrorCode.BAD_ARGS
    assert eng.state.event_log == []


def test_tick_advances_and_decays_emotion() -> None:
    eng = _engine()
    before = eng.state.npcs[NpcId("flandre")].emotion

    eng.tick()

    assert eng.state.tick == 1
    assert eng.state.npcs[NpcId("flandre")].emotion < before


def _badger(eng: WorldEngine, turns: int) -> None:
    """玩家一直搭话：每轮一次 say 加一次 tick，和会话层的顺序一致。"""
    for _ in range(turns):
        eng.apply(Action(actor="player", tool="say", args={"text": "你是不是AI？"}))
        eng.tick()


def test_badgering_reimu_reaches_irritated_and_takes_away_ask_player() -> None:
    """情绪状态机曾经对「说什么」完全失聪：在此之前唯一能动情绪的事件是
    give_item，而灵梦收到赛钱是**消气**——于是她的烦躁度只有向下的路，
    `irritated` 在真实玩法里根本到不了（实测连问 40 轮后 0.10 → 0.00）。
    单测都直接调 bump_emotion 构造情绪，所以这件事一直没让谁红过。

    而 irritated 会拿掉 ask_player，是「情绪真的 gate 住工具集」这条主张
    唯一的实盘证据，它必须能在玩法里被触发。"""
    eng = _engine()
    reimu = eng.state.npcs[NpcId("reimu")]
    assert reimu.mode == "normal"
    assert any(t.name == "ask_player" for t in eng.available_tools(NpcId("reimu")))

    _badger(eng, 40)

    assert reimu.mode == "irritated"
    assert not any(t.name == "ask_player" for t in eng.available_tools(NpcId("reimu")))


def test_the_golden_path_amount_of_talking_does_not_irritate_reimu() -> None:
    """实测 honest 局 21 回合里只搭话 15 次（其余回合在 /give /go）。
    正常玩法就把她惹毛的话，奖励玩家的那一刻反而让对话变差——
    取舍 #5 已经因为同一个方向的错误改过一次全局情绪表。"""
    eng = _engine()

    for _ in range(15):
        eng.apply(Action(actor="player", tool="say", args={"text": "结界怎么了"}))
        eng.tick()
    for _ in range(6):
        eng.tick()

    assert eng.state.npcs[NpcId("reimu")].mode == "normal"


def test_talking_only_moves_the_emotion_of_npcs_in_the_room() -> None:
    eng = _engine()
    flandre_before = eng.state.npcs[NpcId("flandre")].emotion

    eng.apply(Action(actor="player", tool="say", args={"text": "喂"}))

    assert eng.state.npcs[NpcId("reimu")].emotion > 0.1
    assert eng.state.npcs[NpcId("flandre")].emotion == flandre_before


def test_npc_speech_does_not_move_her_own_emotion() -> None:
    """只有玩家搭话算施压。她自己说话也累积的话，情绪会随回合无条件上涨，
    那就退化成一个换了名字的计时器。"""
    eng = _engine()
    before = eng.state.npcs[NpcId("reimu")].emotion

    eng.apply(Action(actor="reimu", tool="say", args={"text": "干嘛"}))

    assert eng.state.npcs[NpcId("reimu")].emotion == before


def test_every_character_card_declares_a_talk_delta() -> None:
    """搭话的情绪方向按角色而定，所以刻意没有全局兜底值。代价是新增角色
    忘了写这一项会静默变成 0.0——她于是永远停在第一个情绪模式里，
    表现是「情绪 gate 不生效」而不是加载失败，正是坑 #5 那类最难查的 bug。"""
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")

    missing = [
        str(npc_id)
        for npc_id, card in defs.characters.items()
        if "player_talked" not in card.emotion.event_deltas
    ]
    assert missing == []
