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
