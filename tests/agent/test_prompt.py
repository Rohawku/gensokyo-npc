from pathlib import Path

from gensokyo.agent.prompt import build_messages
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.rules import bump_attitude
from gensokyo.world.state import build_initial_state

REPO_ROOT = Path(__file__).resolve().parents[2]


def _engine() -> WorldEngine:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    return WorldEngine(build_initial_state(defs), defs)


def test_system_prompt_contains_persona_and_forbidden_phrases() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    messages = build_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )

    system = messages[0].content
    assert "博丽神社的巫女" in system
    assert "我很乐意" in system
    assert "禁止" in system


def test_user_prompt_lists_available_tools_only() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("flandre")]

    messages = build_messages(
        card, eng.observe(NpcId("flandre")), [], eng.available_tools(NpcId("flandre")), []
    )

    user = messages[-1].content
    assert "move" not in user
    assert "break_item" not in user
    assert "ask_player" in user


def test_user_prompt_includes_gate_hint_for_unrevealed_facts() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "好感需达到 24" in user


def test_user_prompt_marks_fact_as_revealable_when_gate_met() -> None:
    eng = _engine()
    bump_attitude(eng.state.npcs[NpcId("reimu")], 40)
    card = eng.defs.characters[NpcId("reimu")]

    user = build_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "现在可以说" in user


def test_dialogue_history_and_errors_are_rendered() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_messages(
        card,
        eng.observe(NpcId("reimu")),
        ["玩家：喂", "灵梦：干嘛"],
        eng.available_tools(NpcId("reimu")),
        ["上一次 move 失败：从博丽神社没法直接过去。"],
    )[-1].content

    assert "玩家：喂" in user
    assert "上一次 move 失败" in user
