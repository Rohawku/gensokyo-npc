from pathlib import Path

from gensokyo.world.engine import WorldEngine
from gensokyo.world.events import EventKind
from gensokyo.world.ids import FactId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.rules import bump_attitude
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


def test_entering_muenzuka_advances_to_s1() -> None:
    eng = _engine()
    assert eng.state.quest.stage is QuestStage.S0_UNAWARE

    eng.apply(Action(actor="player", tool="move", args={"to": "human_village"}))
    eng.apply(Action(actor="player", tool="move", args={"to": "muenzuka"}))

    assert eng.state.quest.stage is QuestStage.S1_ANOMALY
    assert any(ev.kind is EventKind.QUEST_ADVANCE for ev in eng.state.event_log)


def test_first_clue_advances_to_s2() -> None:
    eng = _engine()
    eng.state.quest.stage = QuestStage.S1_ANOMALY
    bump_attitude(eng.state.npcs[NpcId("reimu")], 40)

    eng.apply(Action(actor="reimu", tool="reveal_info", args={"fact": CLUES[0]}))

    assert eng.state.quest.stage is QuestStage.S2_CLUES
    assert eng.state.quest.clues_obtained == {CLUES[0]}


def test_s3_requires_all_three_clues() -> None:
    eng = _engine()
    eng.state.quest.stage = QuestStage.S2_CLUES

    for clue in CLUES[:2]:
        eng.state.player.known_facts.add(clue)
        eng.refresh_quest()
    assert eng.state.quest.stage is QuestStage.S2_CLUES

    eng.state.player.known_facts.add(CLUES[2])
    eng.refresh_quest()

    assert eng.state.quest.stage is QuestStage.S3_SOURCE
    assert eng.state.quest.clues_obtained == set(CLUES)


def test_quest_stage_never_goes_backwards() -> None:
    eng = _engine()
    eng.state.quest.stage = QuestStage.S3_SOURCE
    eng.state.player.known_facts.clear()

    eng.refresh_quest()

    assert eng.state.quest.stage is QuestStage.S3_SOURCE


def test_quest_advance_emits_exactly_one_event_per_transition() -> None:
    eng = _engine()
    eng.state.quest.stage = QuestStage.S1_ANOMALY
    eng.state.player.known_facts.add(CLUES[0])

    eng.refresh_quest()
    eng.refresh_quest()

    advances = [ev for ev in eng.state.event_log if ev.kind is EventKind.QUEST_ADVANCE]
    assert len(advances) == 1
