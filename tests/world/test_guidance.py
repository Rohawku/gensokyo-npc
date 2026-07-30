from pathlib import Path

from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import FactId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.state import QuestStage, build_initial_state
from gensokyo.world.tools import Action

REPO_ROOT = Path(__file__).resolve().parents[2]
CLUES = [
    FactId("barrier_anomaly_time"),
    FactId("flower_magic_composition"),
    FactId("ancient_oblivion_memory"),
]


def _engine() -> WorldEngine:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    return WorldEngine(build_initial_state(defs), defs)


def test_no_suggestion_before_the_gate_opens() -> None:
    eng = _engine()

    assert eng.observe(NpcId("reimu")).suggestion == ""


def test_suggestion_names_reveal_info_once_the_gate_opens() -> None:
    """引擎知道门槛开了，就该直说。实测让小模型自己从情报清单里推，
    命中率只有 20%；把结论直接写进 prompt 后是 100%。"""
    eng = _engine()
    for _ in range(4):
        eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))

    suggestion = eng.observe(NpcId("reimu")).suggestion

    assert "reveal_info" in suggestion
    assert "barrier_anomaly_time" in suggestion


def test_suggestion_stops_once_the_fact_is_out() -> None:
    eng = _engine()
    for _ in range(4):
        eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))
    eng.apply(Action(actor="reimu", tool="reveal_info", args={"fact": CLUES[0]}))

    assert "reveal_info" not in eng.observe(NpcId("reimu")).suggestion


def test_suggestion_points_at_the_source_once_clues_are_complete() -> None:
    """结局要靠 NPC 自己走到无缘塚再发符卡，这一步也不能指望小模型自己想到。"""
    eng = _engine()
    eng.state.player.known_facts.update(CLUES)
    eng.refresh_quest()

    away = eng.observe(NpcId("reimu")).suggestion
    assert "无缘塚" in away and "move" in away

    eng.state.npcs[NpcId("reimu")].location = "muenzuka"
    at_source = eng.observe(NpcId("reimu")).suggestion
    assert "use_spellcard" in at_source


def test_flandre_gets_no_ending_suggestion() -> None:
    """她被禁足、不能收尾，就不该被建议去无缘塚。"""
    eng = _engine()
    eng.state.player.known_facts.update(CLUES)
    eng.refresh_quest()

    assert "无缘塚" not in eng.observe(NpcId("flandre")).suggestion


def test_player_objective_announces_that_she_will_talk() -> None:
    """玩家看不到好感门槛是多少。门槛一开必须明说，否则他会一直投赛钱。"""
    eng = _engine()
    before = eng.observe_player().objective

    for _ in range(4):
        eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))

    after = eng.observe_player().objective
    assert after != before
    assert "博丽灵梦" in after and "愿意开口" in after


def test_player_objective_falls_back_to_the_stage_text() -> None:
    eng = _engine()
    eng.state.player.known_facts.update(CLUES)
    eng.refresh_quest()
    eng.state.player.location = "muenzuka"

    assert eng.observe_player().objective == eng.defs.stages[QuestStage.S3_SOURCE.name].objective
