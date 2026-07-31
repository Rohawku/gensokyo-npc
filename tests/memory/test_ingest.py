from pathlib import Path

from gensokyo.memory.ingest import ingest
from gensokyo.memory.item import MemoryStore
from gensokyo.world.defs import SALIENCE_BASELINE, WorldDefs
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action

REPO_ROOT = Path(__file__).resolve().parents[2]


def _engine() -> WorldEngine:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    return WorldEngine(build_initial_state(defs), defs)


def _store(npc: str) -> MemoryStore:
    return MemoryStore(npc_id=NpcId(npc))


def _absorb(eng: WorldEngine, store: MemoryStore, defs: WorldDefs) -> None:
    """把 event_log 里她感知到的部分全部摄入。测试里位置不变，可以整段喂。"""
    npc = eng.state.npcs[store.npc_id]
    ingest(store, eng.state.event_log, defs.characters[store.npc_id], npc.location, defs)


def test_a_gift_becomes_a_memory_with_the_chinese_item_name() -> None:
    """条目正文会被检索拼进 prompt。坑 #10 清了四轮同一类问题：英文 id
    紧贴中文出现时模型有概率把它说出口。"""
    eng = _engine()
    store = _store("reimu")

    eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))
    _absorb(eng, store, eng.defs)

    assert len(store.items) == 1
    item = store.items[0]
    assert "赛钱" in item.content
    assert "offering_coin" not in item.content
    assert item.kind == "player_gave_item"
    assert item.source_event_id == eng.state.event_log[0].id


def test_seq_comes_from_the_event_id_not_the_tick() -> None:
    """replay 不重放 tick（取舍 #2），回放后 Event.tick 全是 0——时间衰减
    若读 tick，读档会把所有记忆的新鲜度无声归零。事件 id 递增且回放逐字
    复现，已有测试锁住。"""
    eng = _engine()
    store = _store("reimu")

    for _ in range(3):
        eng.apply(Action(actor="player", tool="say", args={"text": "喂"}))
    eng.tick()
    eng.apply(Action(actor="player", tool="say", args={"text": "还在吗"}))
    _absorb(eng, store, eng.defs)

    assert [i.seq for i in store.items] == [1, 2, 3, 4]
    assert {e.tick for e in eng.state.event_log} == {0, 1}


def test_she_does_not_remember_what_happened_elsewhere() -> None:
    """感知过滤只有一条规则：事件发生在她所在的地点。芙兰在地下室，
    所以 blind_to_outside 是这条规则的自然结果，不需要第二套机制。"""
    eng = _engine()
    store = _store("flandre")

    eng.apply(Action(actor="player", tool="say", args={"text": "神社里有人吗"}))
    eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))
    _absorb(eng, store, eng.defs)

    assert store.items == []


def test_the_same_event_is_never_ingested_twice() -> None:
    """检索按分数排序。两条一模一样的条目会挤掉本该被召回的第二名。"""
    eng = _engine()
    store = _store("reimu")

    eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))
    _absorb(eng, store, eng.defs)
    _absorb(eng, store, eng.defs)
    _absorb(eng, store, eng.defs)

    assert len(store.items) == 1


def test_she_remembers_her_own_words_but_not_another_npcs() -> None:
    eng = _engine()
    store = _store("reimu")
    eng.state.npcs[NpcId("marisa")].location = eng.state.npcs[NpcId("reimu")].location

    eng.apply(Action(actor="reimu", tool="say", args={"text": "有事说事"}))
    eng.apply(Action(actor="marisa", tool="say", args={"text": "就是这样"}))
    _absorb(eng, store, eng.defs)

    assert len(store.items) == 1
    assert store.items[0].kind == "npc_talked"
    assert "有事说事" in store.items[0].content


def test_salience_differs_by_character_for_the_same_event() -> None:
    """同一件事在不同角色身上分量不同，这是记忆人格的实现点。芙兰的
    player_arrived 系数 2.5、灵梦 0.8——有人来看她对芙兰是天大的事。"""
    eng = _engine()
    flandre = _store("flandre")
    reimu = _store("reimu")
    basement = eng.state.npcs[NpcId("flandre")].location

    eng.state.player.location = basement
    eng.apply(Action(actor="player", tool="say", args={"text": "在吗"}))
    _absorb(eng, flandre, eng.defs)

    eng2 = _engine()
    eng2.apply(Action(actor="player", tool="say", args={"text": "在吗"}))
    _absorb(eng2, reimu, eng2.defs)

    assert flandre.items[0].salience > reimu.items[0].salience


def test_long_utterances_are_truncated() -> None:
    """一条 200 字的台词能把整个【你还记得】段落撑爆，而说话阶段 prompt
    短小正是拆两阶段买到的东西（坑 #1）。"""
    eng = _engine()
    store = _store("reimu")

    eng.apply(Action(actor="player", tool="say", args={"text": "啊" * 200}))
    _absorb(eng, store, eng.defs)

    assert len(store.items[0].content) < 80
    assert store.items[0].content.endswith("」")


def test_unregistered_event_kinds_are_skipped_rather_than_scored_zero() -> None:
    """ask_player 的基线是 0.15，NPC 自己问话该记住；而没登记基线的事件
    类型一条都不许进库——凭空给分会让它和真正重要的条目竞争。"""
    eng = _engine()
    store = _store("reimu")

    eng.apply(Action(actor="reimu", tool="ask_player", args={"question": "你到底想干嘛"}))
    _absorb(eng, store, eng.defs)

    assert [i.kind for i in store.items] == ["asked_player"]
    assert store.items[0].salience == SALIENCE_BASELINE["asked_player"]
