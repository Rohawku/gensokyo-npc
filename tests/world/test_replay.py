from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import ItemId
from gensokyo.world.loader import load_defs
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
    eng = _fresh()
    eng.state.player.inventory[ItemId("offering_coin")] = 1
    for action in CANDIDATE_ACTIONS:
        eng.apply(action)
        eng.tick()

    replayed = WorldEngine.replay(eng.state.action_log, eng.defs)
    replayed.state.player.inventory.setdefault(ItemId("offering_coin"), 0)

    assert [ev.id for ev in replayed.state.event_log] == [ev.id for ev in eng.state.event_log]
    assert replayed.state.quest.stage == eng.state.quest.stage


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
