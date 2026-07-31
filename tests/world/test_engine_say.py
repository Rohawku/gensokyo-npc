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


def test_an_irritated_reimu_refuses_to_talk_at_all() -> None:
    """prompt 层的禁令在 8B 模型上到顶了：实测对抗人格下她 87.5% 的复读是
    「对语义不同的问题塌缩成同一句敷衍」，而那句话当时就列在禁语清单里。

    被缠得太久就不搭话是她 irritated 的人设本身（角色卡写的是「可能直接
    赶人」），把它变成机制比继续加一句提示可靠。"""
    eng = _engine()
    reimu = NpcId("reimu")
    assert eng.observe_player().npcs_here[0].refusal == ""

    _badger(eng, 40)

    panel = eng.observe_player().npcs_here[0]
    assert eng.state.npcs[reimu].mode == "irritated"
    assert "不打算再理你" in panel.refusal


def test_backing_off_lets_her_cool_down_and_talk_again() -> None:
    """必须能恢复，否则一次纠缠就把这局锁死了——坑 #6 是「游戏做出来不可
    通关」，这类单向门是同一个风险。停下来不说话就够了（衰减是无条件的）。"""
    eng = _engine()
    _badger(eng, 40)
    assert eng.observe_player().npcs_here[0].refusal

    for _ in range(30):
        eng.tick()

    assert eng.observe_player().npcs_here[0].refusal == ""


def test_giving_her_something_also_calms_her_down() -> None:
    """投赛钱让灵梦消气（取舍 #5），所以它同样是一条恢复路径。"""
    eng = _engine()
    _badger(eng, 40)
    assert eng.observe_player().npcs_here[0].refusal

    for _ in range(6):
        eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))
        eng.tick()

    assert eng.observe_player().npcs_here[0].refusal == ""


def test_the_golden_path_never_makes_her_refuse() -> None:
    """honest 局 21 回合里只搭话 15 次。正常玩法把她逼到不搭话的话，
    通关率会直接掉——这条测试和 test_the_golden_path_amount_of_talking...
    守的是同一条线，但断言的是玩家可见的后果。"""
    eng = _engine()

    for _ in range(15):
        eng.apply(Action(actor="player", tool="say", args={"text": "结界怎么了"}))
        eng.tick()
        assert eng.observe_player().npcs_here[0].refusal == ""


def test_only_modes_that_declare_a_refusal_stop_the_conversation() -> None:
    """芙兰的 destructive 是兴奋到失控，她话更多而不是更少；魔理沙的
    excited 同理。一个字段同时表达「拒绝」和「拒绝时说什么」，所以不会出现
    「开关开着而文案是空的」这种自相矛盾的配置（坑 #4 的教训）。"""
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    refusing = {
        (str(npc_id), mode.name)
        for npc_id, card in defs.characters.items()
        for mode in card.emotion.modes
        if mode.refusal
    }

    assert refusing == {("reimu", "irritated")}
