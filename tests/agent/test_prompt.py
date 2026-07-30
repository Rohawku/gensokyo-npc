from pathlib import Path

from gensokyo.agent.prompt import build_decide_messages, build_speak_messages
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

    messages = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )

    system = messages[0].content
    assert "博丽神社的巫女" in system
    assert "我很乐意" in system
    assert "禁止" in system


def test_user_prompt_lists_available_tools_only() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("flandre")]

    messages = build_decide_messages(
        card, eng.observe(NpcId("flandre")), [], eng.available_tools(NpcId("flandre")), []
    )

    user = messages[-1].content
    assert "move" not in user
    assert "break_item" not in user
    assert "ask_player" in user


def test_user_prompt_includes_gate_hint_for_unrevealed_facts() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "好感需达到 24" in user


def test_user_prompt_marks_fact_as_revealable_when_gate_met() -> None:
    eng = _engine()
    bump_attitude(eng.state.npcs[NpcId("reimu")], 40)
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "现在可以说" in user


def test_dialogue_history_and_errors_are_rendered() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
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
    system，只查 user 等于漏掉半个 prompt。决策与说话两个阶段都要查——
    说话阶段的输出会原样落到玩家屏幕上，那里泄漏比决策阶段更致命。
    工具名与参数名是例外——那是模型必须原样填进 JSON 的东西。"""
    eng = _engine()
    eng.state.npcs[NpcId("marisa")].inventory[ItemId("rare_book")] = 1

    leaks = ["rare_book", "magic_mushroom", "withered_flower", "old_music_box"]
    leaks += ["annoyance", "eagerness", "excitement"]
    leaks += ["S0_UNAWARE", "S1_ANOMALY", "S2_CLUES", "S3_SOURCE"]
    leaks += ["modern_technology", "outside_basement_events", "blind_to_outside"]

    for npc_id in (NpcId("reimu"), NpcId("marisa"), NpcId("flandre")):
        card = eng.defs.characters[npc_id]
        obs = eng.observe(npc_id)
        both = build_decide_messages(card, obs, [], eng.available_tools(npc_id), [])
        both += build_speak_messages(card, obs, [], "在想事情", ["ask_player：做到了。"])
        for msg in both:
            for token in leaks:
                assert token not in msg.content, (
                    f"{npc_id} 的 {msg.role} prompt 泄漏了内部标识符 {token}"
                )


def test_unrevealable_fact_hides_its_id_from_the_model() -> None:
    """门槛未满足时连 fact_id 都不给，模型凑不出 reveal_info 的参数。
    这是引擎门槛之外的第二层防线。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "barrier_anomaly_time" not in user

    bump_attitude(eng.state.npcs[NpcId("reimu")], 24)
    unlocked = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "barrier_anomaly_time" in unlocked


def test_decide_prompt_asks_for_no_utterance() -> None:
    """决策阶段的输出契约里不能再有 utterance——留着它模型就会照旧
    在第一次调用里把话说完，两阶段拆分等于白做。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "utterance" not in user
    assert "现在不要说话" in user


def test_speak_prompt_carries_thought_and_outcomes() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_speak_messages(
        card,
        eng.observe(NpcId("reimu")),
        ["玩家：结界怎么了", "灵梦：不知道"],
        "这家伙又来白拿",
        ["reveal_info：没做到——好感不够。"],
    )[-1].content

    assert "这家伙又来白拿" in user
    assert "没做到——好感不够。" in user
    assert "玩家：结界怎么了" in user


def test_speak_prompt_stays_short() -> None:
    """说话阶段重复整个场景 / 物品 / 情报清单会把 prompt 处理时间加回来，
    而首字延迟正是这次拆分要买的东西。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]
    obs = eng.observe(NpcId("reimu"))

    decide = build_decide_messages(card, obs, [], eng.available_tools(NpcId("reimu")), [])[-1]
    speak = build_speak_messages(card, obs, [], "t", [])[-1]

    assert len(speak.content) < len(decide.content) / 2
    assert "【你知道的情报】" not in speak.content


def test_speak_prompt_tolerates_empty_thought_and_outcomes() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_speak_messages(card, eng.observe(NpcId("reimu")), [], "", [])[-1].content

    assert "（没想什么）" in user
    assert "你什么也没做。" in user


def test_speak_prompt_forbids_quotes_and_narration() -> None:
    """小模型爱把台词裹在引号里、加上名字或旁白。策略层有清理兜底，
    但先在 prompt 里说清楚能少踩一层。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_speak_messages(card, eng.observe(NpcId("reimu")), [], "t", [])[-1].content

    assert "不要加引号" in user
    assert "不要输出 JSON" in user
