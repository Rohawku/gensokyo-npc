from pathlib import Path

from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import ItemId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.rules import bump_attitude
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action

REPO_ROOT = Path(__file__).resolve().parents[2]


def _engine() -> WorldEngine:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    return WorldEngine(build_initial_state(defs), defs)


def test_observation_reports_own_location_and_mode() -> None:
    eng = _engine()

    obs = eng.observe(NpcId("flandre"))

    assert obs.npc_name == "芙兰朵露·斯卡雷特"
    assert obs.location_name == "红魔馆地下室"
    assert obs.mode == "calm"
    assert obs.emotion_var == "兴奋度"
    assert obs.player_is_here is False


def test_observation_knows_when_player_is_present() -> None:
    eng = _engine()

    obs = eng.observe(NpcId("reimu"))

    assert obs.player_is_here is True


def test_flandre_cannot_see_quest_progress() -> None:
    """芙兰的 forbidden_knowledge 含 outside_basement_events，
    她不该知道外面的异变调查进展。"""
    eng = _engine()

    assert eng.observe(NpcId("flandre")).quest_hint is None
    assert eng.observe(NpcId("reimu")).quest_hint is not None


def test_observation_lists_facts_with_gate_status() -> None:
    eng = _engine()

    before = eng.observe(NpcId("reimu")).facts
    assert len(before) == 1
    assert before[0].fact_id == "barrier_anomaly_time"
    assert before[0].can_reveal_now is False
    assert before[0].already_revealed is False

    bump_attitude(eng.state.npcs[NpcId("reimu")], 40)
    after = eng.observe(NpcId("reimu")).facts

    assert after[0].can_reveal_now is True


def test_observation_lists_items_here_and_own_inventory() -> None:
    eng = _engine()
    eng.state.npcs[NpcId("reimu")].inventory[ItemId("rare_book")] = 1

    obs = eng.observe(NpcId("reimu"))

    assert obs.own_inventory == {"珍稀魔法书": 1}
    assert obs.items_here == {}


def test_player_view_reports_panel_data() -> None:
    eng = _engine()
    eng.apply(Action(actor="player", tool="move", args={"to": "human_village"}))

    view = eng.observe_player()

    assert view.location_name == "人间之里"
    assert set(view.exits) == {"博丽神社", "雾雨魔法店", "无缘塚"}
    assert view.npcs_here == []
    assert view.quest_stage == "S0_UNAWARE"


def test_player_view_includes_npc_attitude_panel() -> None:
    eng = _engine()

    view = eng.observe_player()

    assert len(view.npcs_here) == 1
    panel = view.npcs_here[0]
    assert panel.name == "博丽灵梦"
    assert panel.attitude == 0
    assert panel.emotion_var == "烦躁度"


def test_gate_hint_uses_chinese_item_names() -> None:
    """gate_hint 会原样进 prompt。渲染成 rare_book 这类 id 会让
    魔理沙在对话里说出英文 id，破坏角色扮演。"""
    eng = _engine()

    hint = eng.observe(NpcId("marisa")).facts[0].gate_hint

    assert "珍稀魔法书" in hint
    assert "魔法蘑菇" in hint
    assert "rare_book" not in hint
    assert "magic_mushroom" not in hint
