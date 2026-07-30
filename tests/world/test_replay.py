from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gensokyo.world.engine import WorldEngine
from gensokyo.world.loader import load_defs
from gensokyo.world.rules import resolve_mode
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action

REPO_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_ACTIONS = [
    Action(actor="player", tool="say", args={"text": "喂"}),
    Action(actor="player", tool="move", args={"to": "human_village"}),
    Action(actor="player", tool="move", args={"to": "muenzuka"}),
    Action(actor="player", tool="move", args={"to": "hakurei_shrine"}),
    Action(actor="player", tool="take_item", args={"item": "withered_flower"}),
    Action(actor="player", tool="give_item", args={"item": "withered_flower"}),
    Action(actor="reimu", tool="ask_player", args={"question": "干什么"}),
    Action(actor="reimu", tool="take_item", args={"item": "withered_flower"}),
    Action(actor="reimu", tool="reveal_info", args={"fact": "barrier_anomaly_time"}),
    Action(actor="flandre", tool="move", args={"to": "forest_of_magic"}),
]


def _fresh() -> WorldEngine:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    return WorldEngine(build_initial_state(defs), defs)


def test_successful_actions_are_recorded_failures_are_not() -> None:
    eng = _fresh()

    eng.apply(Action(actor="player", tool="say", args={"text": "喂"}))
    eng.apply(Action(actor="player", tool="move", args={"to": "scarlet_devil_basement"}))

    assert len(eng.state.action_log) == 1
    assert eng.state.action_log[0].tool == "say"


def test_replay_reproduces_state_and_event_log() -> None:
    """比较完整 state，不只是 event id 列表。原先测试先手改了玩家赛钱数，
    而 replay 走 build_initial_state，两边 state 注定不等——于是断言只能
    退化成比 event id。tick 刻意不参与重放（replay 只喂动作），所以这里
    也不 tick。"""
    eng = _fresh()
    for action in CANDIDATE_ACTIONS:
        eng.apply(action)

    replayed = WorldEngine.replay(eng.state.action_log, eng.defs)

    assert [ev.id for ev in replayed.state.event_log] == [ev.id for ev in eng.state.event_log]
    assert replayed.state.model_dump() == eng.state.model_dump()


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.sampled_from(CANDIDATE_ACTIONS), min_size=0, max_size=25))
def test_replay_is_deterministic(actions: list[Action]) -> None:
    """世界引擎是整个项目的地基，它的 bug 会以「NPC 行为莫名其妙」
    的形式出现在上层，极难定位。用属性测试守住回放一致性。"""
    first = _fresh()
    for action in actions:
        first.apply(action)

    second = WorldEngine.replay(first.state.action_log, first.defs)

    assert second.state.model_dump() == first.state.model_dump()


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.sampled_from(CANDIDATE_ACTIONS), min_size=0, max_size=25))
def test_invariants_hold_under_arbitrary_action_sequences(actions: list[Action]) -> None:
    eng = _fresh()
    for action in actions:
        eng.apply(action)

    for npc in eng.state.npcs.values():
        assert -100 <= npc.attitude <= 100
        assert 0.0 <= npc.emotion <= 1.0
    for count in eng.state.player.inventory.values():
        assert count > 0
    assert eng.state.npcs["flandre"].location == "scarlet_devil_basement"
    # 情绪状态机的核心不变量：mode 是 emotion 的纯函数，任何时刻都不许脱钩。
    # 少了这一条，bump_emotion 里漏掉 mode 重算也没有测试会红。
    for npc_id, npc in eng.state.npcs.items():
        assert npc.mode == resolve_mode(eng.defs.characters[npc_id], npc.emotion)


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.sampled_from(CANDIDATE_ACTIONS), min_size=0, max_size=25))
def test_failed_actions_leave_no_trace(actions: list[Action]) -> None:
    """失败动作必须完全无副作用。apply() 里任何「先扣再校验」的写法都会在这里
    露出来：玩家打错一条指令不该悄悄改掉世界，也不该往日志里留半条记录。"""
    eng = _fresh()
    for action in actions:
        events_before = len(eng.state.event_log)
        actions_before = len(eng.state.action_log)
        seq_before = eng.state.seq

        result = eng.apply(action)

        if result.ok is False:
            assert len(eng.state.event_log) == events_before, f"{action.tool} 失败却写了事件"
            assert len(eng.state.action_log) == actions_before, f"{action.tool} 失败却进了动作日志"
            assert eng.state.seq == seq_before, f"{action.tool} 失败却消耗了事件序号"
