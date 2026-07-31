from pathlib import Path

import pytest

from gensokyo.memory.decay import COMPRESS_BELOW, DORMANT_BELOW, demote, retention
from gensokyo.memory.item import MemoryItem, MemoryStore, Tier
from gensokyo.memory.pipeline import absorb, new_stores, now_seq, rebuild
from gensokyo.memory.retrieve import recall_dormant, retrieve
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action

REPO_ROOT = Path(__file__).resolve().parents[2]


def _defs():  # type: ignore[no-untyped-def]
    return load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")


def _engine() -> WorldEngine:
    defs = _defs()
    return WorldEngine(build_initial_state(defs), defs)


def _item(seq: int, salience: float, tier: Tier = Tier.ACTIVE) -> MemoryItem:
    return MemoryItem(
        id=f"m-{seq}",
        npc_id=NpcId("reimu"),
        seq=seq,
        content="随便一件事。",
        source_event_id=None,
        kind="player_talked",
        salience=salience,
        tier=tier,
    )


# ---------------------------------------------------------------- 保留强度


def test_salience_slows_decay_rather_than_setting_a_floor() -> None:
    """做成下限的话，高显著性条目永远停在活跃层，第三级就成了只有种子记忆
    才到得了的死档位——又一条空转配置。乘法形式让三档都可达。"""
    important = _item(seq=0, salience=0.9)

    assert retention(important, now_seq=0, lambda_decay=0.08) > 1.0
    assert retention(important, now_seq=200, lambda_decay=0.08) < DORMANT_BELOW


def test_a_more_salient_memory_outlives_a_trivial_one() -> None:
    trivial = _item(seq=0, salience=0.05)
    important = _item(seq=0, salience=0.9)

    assert retention(important, 30, 0.08) > retention(trivial, 30, 0.08)


# ---------------------------------------------------------------- 降级


def test_all_three_tiers_are_reachable() -> None:
    defs = _defs()
    card = defs.characters[NpcId("reimu")]
    store = MemoryStore(npc_id=NpcId("reimu"), items=[_item(seq=0, salience=0.14)])
    item = store.items[0]

    demote(store, card, now_seq=0)
    assert item.tier is Tier.ACTIVE

    demote(store, card, now_seq=20)
    assert item.tier is Tier.COMPRESSED

    demote(store, card, now_seq=60)
    assert item.tier is Tier.DORMANT


def test_demotion_is_one_way() -> None:
    """分层单向：一条已经压缩的记忆不会因为时间又变回原文。"""
    defs = _defs()
    card = defs.characters[NpcId("reimu")]
    store = MemoryStore(npc_id=NpcId("reimu"), items=[_item(seq=50, salience=0.5)])
    store.items[0].tier = Tier.COMPRESSED

    demote(store, card, now_seq=50)

    assert store.items[0].tier is Tier.COMPRESSED


def test_a_recalled_memory_never_slides_back_to_sleep() -> None:
    """种子记忆的 seq 是 0，保留强度早就跌到底。不跳过 recalled 的话，
    芙兰那条线索会在玩家刚问出口的下一个回合又消失。"""
    defs = _defs()
    card = defs.characters[NpcId("flandre")]
    store = MemoryStore(npc_id=NpcId("flandre"), items=[])
    seed = _item(seq=0, salience=0.2, tier=Tier.DORMANT)
    seed.npc_id = NpcId("flandre")
    seed.trigger_keys = ("枯萎的花",)
    store.items.append(seed)

    recall_dormant(store, frozenset({"枯萎的花"}), now_seq=80)
    demote(store, card, now_seq=200)

    assert seed.tier is Tier.ACTIVE


def test_forgetting_speed_is_ordered_by_persona() -> None:
    """同一批条目、同一个时刻，芙兰忘得最多、魔理沙最少。这是记忆人格的
    端到端体现——λ 一旦被调平，三个 NPC 的记忆表现就没差别了。"""
    defs = _defs()
    survivors: dict[str, int] = {}
    for npc in ("flandre", "reimu", "marisa"):
        store = MemoryStore(
            npc_id=NpcId(npc), items=[_item(seq=s, salience=0.3) for s in range(0, 40, 4)]
        )
        for it in store.items:
            it.npc_id = NpcId(npc)
        demote(store, defs.characters[NpcId(npc)], now_seq=40)
        survivors[npc] = sum(1 for i in store.items if i.tier is Tier.ACTIVE)

    assert survivors["flandre"] < survivors["reimu"] < survivors["marisa"]


def test_compressed_memories_are_still_retrievable() -> None:
    """压缩是「记得大概但记不清具体」，不是消失。只有沉睡才退出常规检索。"""
    defs = _defs()
    store = MemoryStore(npc_id=NpcId("reimu"), items=[_item(seq=1, salience=0.5)])
    store.items[0].tier = Tier.COMPRESSED

    assert retrieve(store, "随便", defs.characters[NpcId("reimu")], now_seq=2) != []


def test_thresholds_are_ordered() -> None:
    assert DORMANT_BELOW < COMPRESS_BELOW


# ---------------------------------------------------------------- 回放一致性


def _played() -> WorldEngine:
    """一局用动作构造的对局。**全程只用动作**——直接改 state 不进动作日志，
    重建自然重现不出来，而坑 #9 那条测试第一次就是这么写错的。"""
    eng = _engine()
    for _ in range(4):
        eng.apply(Action(actor="player", tool="say", args={"text": "结界怎么了"}))
        eng.tick()
        eng.apply(Action(actor="reimu", tool="say", args={"text": "你管的太多了"}))
    for _ in range(4):
        eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))
        eng.tick()
    eng.apply(Action(actor="reimu", tool="reveal_info", args={"fact": "barrier_anomaly_time"}))
    eng.apply(Action(actor="player", tool="travel_to", args={"destination": "human_village"}))
    eng.tick()
    eng.apply(Action(actor="player", tool="say", args={"text": "有人吗"}))
    return eng


def test_rebuilt_memory_matches_the_live_one_field_for_field() -> None:
    """存档只有动作日志（取舍 #2），记忆是推导产物。**实时与读档必须同一
    条码路**——若两者给出不同的记忆库，存档就改变了「她记得什么」，
    比丢一条线索更糟，而且无声。"""
    eng = _played()
    live = new_stores(eng.defs)
    absorb(eng, live)

    _, rebuilt = rebuild(eng.state.action_log, eng.defs)

    assert set(rebuilt) == set(live)
    for npc_id, store in live.items():
        assert rebuilt[npc_id].model_dump() == store.model_dump(), f"{npc_id} 的记忆库分叉了"


def test_absorbing_repeatedly_changes_nothing() -> None:
    """`absorb` 整段喂日志而不是喂增量，于是调用频率不影响结果——「每回合
    一次」和「重放时每个动作一次」因此必然一致。"""
    eng = _played()
    once = new_stores(eng.defs)
    absorb(eng, once)
    many = new_stores(eng.defs)
    for _ in range(5):
        absorb(eng, many)

    for npc_id in once:
        assert many[npc_id].model_dump() == once[npc_id].model_dump()


def test_a_memory_that_decayed_before_saving_stays_decayed_after_loading() -> None:
    """这是「回放一致」里最容易漏的一半：不只要求条目数一样，还要求分层
    一样。若衰减挂在 tick 上，回放后 Event.tick 全是 0，所有记忆会凭空
    变新鲜——玩家读档后她突然什么都记得。"""
    eng = _played()
    live = new_stores(eng.defs)
    absorb(eng, live)
    faded = {i.id for i in live[NpcId("reimu")].items if i.tier is not Tier.ACTIVE}
    assert faded, "这局太短，没有任何条目降级——测试前提不成立"

    _, rebuilt = rebuild(eng.state.action_log, eng.defs)

    assert {i.id for i in rebuilt[NpcId("reimu")].items if i.tier is not Tier.ACTIVE} == faded


def test_now_seq_of_an_empty_log_is_zero() -> None:
    assert now_seq(_engine()) == 0


def test_now_seq_tracks_the_last_event() -> None:
    eng = _engine()
    for _ in range(3):
        eng.apply(Action(actor="player", tool="say", args={"text": "喂"}))

    assert now_seq(eng) == 3


def test_each_npc_gets_her_own_store() -> None:
    """记忆库互不可见是设计：三个 NPC 各自只知道自己经历过的事，玩家因此
    可以对不同人说不同的话。"""
    eng = _played()
    stores = new_stores(eng.defs)
    absorb(eng, stores)

    reimu = {i.content for i in stores[NpcId("reimu")].items}
    flandre = {i.content for i in stores[NpcId("flandre")].items}

    assert reimu
    assert reimu & flandre == set()


def test_retention_is_never_negative() -> None:
    assert retention(
        _item(seq=0, salience=0.0), now_seq=10_000, lambda_decay=0.25
    ) == pytest.approx(0.0)


def test_she_remembers_what_happened_after_she_moved() -> None:
    """`ingest` 从事件流自己推导她当时在哪。这条路径只有在**NPC 自己移动过**
    的对局里才走到——第一版的回放测试里只有玩家在动，于是「去掉位置推导」
    这个突变活了下来（坑 #11：没见过红的分支等于没测）。

    魔理沙走到无缘塚之后发生的事，她必须记得；而按 `card.home` 一路过滤的
    实现会把它们全部丢掉。
    """
    eng = _engine()
    marisa = NpcId("marisa")

    eng.apply(Action(actor="player", tool="say", args={"text": "在店里吗"}))
    eng.apply(Action(actor=str(marisa), tool="travel_to", args={"destination": "muenzuka"}))
    # travel_to 是 NPC 专用的，玩家得一步步走。
    eng.apply(Action(actor="player", tool="move", args={"to": "human_village"}))
    eng.apply(Action(actor="player", tool="move", args={"to": "muenzuka"}))
    eng.apply(Action(actor="player", tool="say", args={"text": "花就在这儿"}))

    stores = new_stores(eng.defs)
    absorb(eng, stores)
    remembered = " ".join(i.content for i in stores[marisa].items)

    assert "花就在这儿" in remembered
    assert "在店里吗" not in remembered
