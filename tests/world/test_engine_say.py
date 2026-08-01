from pathlib import Path

import pytest

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


def _warning(eng: WorldEngine) -> str:
    return eng.observe_player().npcs_here[0].mood_warning


def test_she_telegraphs_before_she_stops_talking() -> None:
    """从「懒散」直接跳到「转身走开」是断崖式的，玩家不知道自己踩到了什么。
    引擎里已经有同一套做法的先例——无缘塚的遗忘机制会提前说「再在这里待
    2 步」。**可预告的惩罚才是机制，不可预告的是陷阱。**"""
    eng = _engine()
    assert _warning(eng) == ""

    warned_at = None
    for i in range(40):
        _badger(eng, 1)
        if eng.observe_player().npcs_here[0].refusal:
            break
        if _warning(eng) and warned_at is None:
            warned_at = i

    assert warned_at is not None, "她一路走到不搭话都没给过预警"


def test_the_countdown_is_accurate_to_the_turn() -> None:
    """倒计时必须按**回合内的时序**算，不能按「每回合净增多少」除一除。

    一个回合里的顺序是「玩家发言推高情绪 → 判定模式 → 回合末衰减」，所以
    翻脸发生在发言那一刻，那一次的衰减还没扣。第一版按净增算，屏幕上写着
    「再缠 3 次」而她下一回合就翻脸了——预警比没有预警更糟，因为它给了一个
    错误的安全感。

    这条测试断言的是 `[3, 2, 1]` 而不是「单调下降」：只要单调的话，
    `[3]` 后面直接翻脸也算通过，而那正是第一版的样子。
    """
    eng = _engine()
    seen: list[int] = []
    for _ in range(40):
        _badger(eng, 1)
        warning = _warning(eng)
        if warning:
            seen.append(int("".join(c for c in warning if c.isdigit())))
        if eng.observe_player().npcs_here[0].refusal:
            break

    assert seen == [3, 2, 1], f"倒计时不准：{seen}"


def test_no_warning_once_she_has_already_stopped_talking() -> None:
    """已经翻脸了还预警「再缠 1 次她就不理你」是自相矛盾的一帧——
    和迟滞修掉的那个问题同一类。"""
    eng = _engine()
    _badger(eng, 40)

    panel = eng.observe_player().npcs_here[0]
    assert panel.refusal
    assert panel.mood_warning == ""


def test_nobody_warns_when_talking_cannot_anger_her() -> None:
    """魔理沙和芙兰的模式都没声明 approaching，而且搭话让她们更兴奋不是更烦。
    给一个走不到的状态预警等于常驻一条永远不会兑现的警告。"""
    eng = _engine()
    for npc in ("marisa", "flandre"):
        eng.state.player.location = eng.state.npcs[NpcId(npc)].location
        _badger(eng, 40)
        assert eng.observe_player().npcs_here[0].mood_warning == ""


def test_a_warning_without_a_countdown_fails_to_load() -> None:
    """预警文案漏掉占位符会渲染成一句没有数字的话，而它的全部价值就是那个
    数字——静默退化成一句废话，正是坑 #5 那类。"""
    from pydantic import ValidationError

    from gensokyo.world.defs import EmotionMode

    with pytest.raises(ValidationError, match="turns"):
        EmotionMode(name="x", range=(0.6, 1.0), approaching="她快生气了。")


def _set_mood(eng: WorldEngine, npc: str, emotion: float, mode: str) -> None:
    """直接摆放情绪与模式。

    这里可以绕过动作日志，因为被测的是**显示函数**——`_mood_warning` 只读
    当前状态、不写任何东西，回放一致性与它无关。涉及回放的测试仍然必须
    全程用动作构造（坑 #9）。
    """
    state = eng.state.npcs[NpcId(npc)]
    state.emotion = emotion
    state.mode = mode


def test_no_warning_when_she_is_already_past_the_threshold() -> None:
    """她还在 normal 但情绪已经超过门槛（迟滞让模式滞后一拍）时，倒计时会
    算成 0——屏幕上出现「再缠 0 次她大概就不会理你了」。"""
    eng = _engine()
    _set_mood(eng, "reimu", 0.63, "normal")

    assert eng.observe_player().npcs_here[0].mood_warning == ""


def test_no_warning_while_she_is_already_refusing() -> None:
    """翻脸之后衰减会把她带回 0.55~0.60，而那个区间按倒计时公式算出来是
    「再缠 2 次」——于是「不打算再理你了」和「再缠 2 次她就不理你了」同框。
    和迟滞修掉的那一帧是同一类矛盾。"""
    eng = _engine()
    _set_mood(eng, "reimu", 0.58, "irritated")

    panel = eng.observe_player().npcs_here[0]
    assert panel.refusal
    assert panel.mood_warning == ""


def test_a_character_whose_mood_cannot_rise_never_warns() -> None:
    """净增为零时倒计时公式会除以零。而「情绪涨不上去」是个合法配置——
    某个角色的 player_talked 增量正好等于衰减率就会这样。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]
    card.emotion.event_deltas["player_talked"] = card.emotion.decay_per_tick
    _set_mood(eng, "reimu", 0.56, "normal")

    assert eng.observe_player().npcs_here[0].mood_warning == ""
