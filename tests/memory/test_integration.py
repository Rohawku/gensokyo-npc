from pathlib import Path

from gensokyo.agent.prompt import build_decide_messages
from gensokyo.llm.client import ScriptedLlmClient
from gensokyo.memory.item import MemoryItem, Tier
from gensokyo.memory.query import build_focus, build_query
from gensokyo.memory.render import _VAGUE, render_recall
from gensokyo.memory.retrieve import Scored
from gensokyo.session.loop import Session
from gensokyo.world.defs import SALIENCE_BASELINE
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.state import build_initial_state

REPO_ROOT = Path(__file__).resolve().parents[2]

_DECIDE = '{"thought": "…", "tool_calls": []}'


def _session(replies: list[str] | None = None) -> Session:
    return Session.create(
        REPO_ROOT / "scenario",
        REPO_ROOT / "characters",
        ScriptedLlmClient(replies if replies is not None else [_DECIDE, "干嘛"] * 40),
    )


def _engine() -> WorldEngine:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    return WorldEngine(build_initial_state(defs), defs)


def _scored(
    content: str,
    tier: Tier = Tier.ACTIVE,
    kind: str = "player_talked",
    duplicates: int = 0,
) -> Scored:
    return Scored(
        item=_item(content, tier, kind),
        similarity=0.0,
        recency=1.0,
        salience=0.5,
        relevance=0.0,
        duplicates=duplicates,
    )


def _item(content: str, tier: Tier = Tier.ACTIVE, kind: str = "player_talked") -> MemoryItem:
    return MemoryItem(
        id=f"m-{content}",
        npc_id=NpcId("reimu"),
        seq=1,
        content=content,
        source_event_id=None,
        kind=kind,
        salience=0.5,
        tier=tier,
    )


# ---------------------------------------------------------------- 渲染


def test_active_memories_are_rendered_verbatim() -> None:
    assert render_recall([_scored("来访者给了我 3 个赛钱。")]) == ["来访者给了我 3 个赛钱。"]


def test_compressed_memories_become_one_vague_line_per_kind() -> None:
    """压缩是「记得大概但记不清具体」。实现成渲染而不是改写记忆库——改写会
    毁掉 source_event_id，而那个指针是把「忘了」和「记错了」拆开的唯一依据。"""
    items = [
        _scored("来访者说：「一」", Tier.COMPRESSED),
        _scored("来访者说：「二」", Tier.COMPRESSED),
        _scored("来访者给了我 1 个赛钱。", Tier.COMPRESSED, kind="player_gave_item"),
    ]

    lines = render_recall(items)

    assert len(lines) == 2
    assert any("2 次" in ln and "记不清" in ln for ln in lines)
    assert not any("一" in ln for ln in lines)


def test_vivid_memories_come_before_vague_ones() -> None:
    """模糊印象信息量低，放前面会占掉注意力，而这一段的目的是让她说出
    具体的事。"""
    lines = render_recall([_scored("模糊的", Tier.COMPRESSED), _scored("清楚的")])

    assert lines[0] == "清楚的"


def test_dormant_memories_are_not_rendered_at_all() -> None:
    assert render_recall([_scored("495 年前的事", Tier.DORMANT)]) == []


def test_every_salience_kind_has_a_vague_phrase() -> None:
    """缺一个键那一类记忆压缩后会静默消失——表现是「她忘得比设定快」而不是
    报错，又是坑 #5 那类。"""
    assert set(_VAGUE) == set(SALIENCE_BASELINE)


# ---------------------------------------------------------------- 查询构造


def test_the_query_carries_the_engine_quest_hint_not_only_the_players_words() -> None:
    """只用玩家原话的话，「你还记得我给过你什么吗」里没有任何和条目共享的
    字面（条目写的是「来访者给了我 3 个赛钱」），bigram 相似度几乎为零。
    引擎已经知道当前该关心什么，直接拼进去。"""
    eng = _engine()
    obs = eng.observe(NpcId("reimu"))
    assert obs.quest_hint

    query = build_query(obs, "你还记得吗")

    assert "你还记得吗" in query
    assert obs.quest_hint in query


def test_focus_is_chinese_names_so_it_can_match_the_content() -> None:
    """focus 走的是精确子串匹配那一路，而条目正文里是中文（坑 #10 清了四轮
    内部标识符）。两边必须用同一种表示。"""
    eng = _engine()
    eng.state.npcs[NpcId("reimu")].inventory[next(iter(eng.defs.items))] = 1
    obs = eng.observe(NpcId("reimu"))

    focus = build_focus(obs)

    assert obs.location_name in focus
    assert all(not k.isascii() for k in focus if k)


# ---------------------------------------------------------------- 接进 prompt


def test_recalled_memories_reach_the_decide_prompt() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card,
        eng.observe(NpcId("reimu")),
        [],
        eng.available_tools(NpcId("reimu")),
        [],
        ["来访者上次空手来的。"],
    )[-1].content

    assert "你还记得的事" in user
    assert "来访者上次空手来的。" in user
    assert "别编" in user


def test_the_section_is_omitted_when_nothing_was_recalled() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), [], []
    )[-1].content

    assert "你还记得的事" not in user


# ---------------------------------------------------------------- 端到端


def test_talking_to_her_builds_up_memory_across_turns() -> None:
    session = _session()

    session.say("结界怎么了")
    session.say("你还记得我刚问了什么吗")

    store = session.stores[NpcId("reimu")]
    assert len(store.items) >= 2
    assert any("结界怎么了" in i.content for i in store.items)


def test_a_gift_given_by_command_also_becomes_a_memory() -> None:
    """`/give` 不触发 NPC 发言，所以摄入不能只挂在 say 上——实测 honest 局
    21 回合里大部分回合都是这类指令。"""
    session = _session()

    result = session.give("赛钱")

    assert result.ok
    store = session.stores[NpcId("reimu")]
    assert any(i.kind == "player_gave_item" for i in store.items)


def test_the_turn_reports_which_memories_it_used() -> None:
    """retrieved_memory_ids 是 W1 就留好的可观测性字段，一直是空的。没有它，
    一次糟糕的回复无法归因到检索还是生成。"""
    session = _session()

    session.say("结界怎么了")
    turns = session.say("你还记得吗")

    assert turns[0].retrieved_memory_ids
    known = {i.id for i in session.stores[NpcId("reimu")].items}
    assert set(turns[0].retrieved_memory_ids) <= known


def test_loading_a_save_restores_her_long_term_but_not_her_short_term_memory(
    tmp_path: Path,
) -> None:
    """读档后她记得**发生过什么**，但不记得原话——短期窗口是 agent 层的东西，
    不进存档。这正好是人的样子。"""
    session = _session()
    session.say("结界怎么了")
    session.give("赛钱")
    before = session.stores[NpcId("reimu")].model_dump()
    save = tmp_path / "s.json"
    session.save(save)

    fresh = _session()
    fresh.load(save)

    assert fresh.stores[NpcId("reimu")].model_dump() == before
    assert fresh.agents[NpcId("reimu")].history == []
    assert fresh.agents[NpcId("reimu")].store is fresh.stores[NpcId("reimu")]
