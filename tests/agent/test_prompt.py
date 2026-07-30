from pathlib import Path

from gensokyo.agent.prompt import build_messages
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import ItemId, NpcId
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


def test_prompt_prose_contains_no_internal_identifiers() -> None:
    """内部标识符（物品 id、情绪变量名、模式名、阶段枚举名、信息隔离开关名）
    不能进 prompt 散文，紧贴中文出现时模型有概率把它们说出口。
    system 和 user 两条消息都要查——角色卡里的 forbidden_knowledge 渲染进
    system，只查 user 等于漏掉半个 prompt。工具名与参数名是例外——
    那是模型必须原样填进 JSON 的东西。"""
    eng = _engine()
    eng.state.npcs[NpcId("marisa")].inventory[ItemId("rare_book")] = 1

    leaks = ["rare_book", "magic_mushroom", "withered_flower", "old_music_box"]
    leaks += ["annoyance", "eagerness", "excitement"]
    leaks += ["S0_UNAWARE", "S1_ANOMALY", "S2_CLUES", "S3_SOURCE"]
    leaks += ["modern_technology", "outside_basement_events", "blind_to_outside"]

    for npc_id in (NpcId("reimu"), NpcId("marisa"), NpcId("flandre")):
        card = eng.defs.characters[npc_id]
        messages = build_messages(card, eng.observe(npc_id), [], eng.available_tools(npc_id), [])
        for msg in messages:
            for token in leaks:
                assert token not in msg.content, (
                    f"{npc_id} 的 {msg.role} prompt 泄漏了内部标识符 {token}"
                )


def test_unrevealable_fact_hides_its_id_from_the_model() -> None:
    """门槛未满足时连 fact_id 都不给，模型凑不出 reveal_info 的参数。
    这是引擎门槛之外的第二层防线。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "barrier_anomaly_time" not in user

    bump_attitude(eng.state.npcs[NpcId("reimu")], 24)
    unlocked = build_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "barrier_anomaly_time" in unlocked
