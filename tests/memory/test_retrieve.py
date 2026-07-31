from pathlib import Path

import pytest

from gensokyo.memory.item import MemoryItem, MemoryStore, Tier
from gensokyo.memory.retrieve import (
    recall_dormant,
    recency,
    relevance,
    retrieve,
    score_all,
)
from gensokyo.memory.similarity import bigram_cosine
from gensokyo.world.defs import CharacterCard
from gensokyo.world.ids import NpcId
from gensokyo.world.loader import load_defs

REPO_ROOT = Path(__file__).resolve().parents[2]


def _card(npc: str = "reimu") -> CharacterCard:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    return defs.characters[NpcId(npc)]


def _item(
    seq: int,
    content: str,
    salience: float = 0.5,
    tier: Tier = Tier.ACTIVE,
    triggers: tuple[str, ...] = (),
) -> MemoryItem:
    return MemoryItem(
        id=f"m-{seq}",
        npc_id=NpcId("reimu"),
        seq=seq,
        content=content,
        source_event_id=None,
        kind="player_talked",
        salience=salience,
        tier=tier,
        trigger_keys=triggers,
    )


def _store(*items: MemoryItem) -> MemoryStore:
    return MemoryStore(npc_id=NpcId("reimu"), items=list(items))


# ---------------------------------------------------------------- 相似度


def test_bigram_cosine_is_one_for_identical_text() -> None:
    assert bigram_cosine("赛钱箱空着", "赛钱箱空着") == pytest.approx(1.0)


def test_bigram_cosine_is_zero_with_no_shared_bigrams() -> None:
    assert bigram_cosine("赛钱", "魔法") == 0.0


def test_bigram_cosine_finds_a_substring_in_a_longer_sentence() -> None:
    """检索 query 短、条目长是常态，相似度不能因为长度差就归零。"""
    assert bigram_cosine("赛钱", "来访者给了我 3 个赛钱。") > 0.0


def test_a_single_character_query_scores_zero_by_contract() -> None:
    """不足两个字得 0 分是明确的契约。query 由引擎拼装、条目正文是模板
    散文，两边都不可能只有一个字；补一个 unigram 回退会引入永远走不到的
    分支，而单字匹配（「的」命中一切）反而拉低精度。单字线索走 relevance
    那一路，它做的是子串匹配。"""
    assert bigram_cosine("花", "无缘塚的花开得很怪") == 0.0
    assert relevance(_item(seq=1, content="无缘塚的花开得很怪"), frozenset({"花"})) == 1.0


def test_empty_text_scores_zero_instead_of_raising() -> None:
    assert bigram_cosine("", "任何内容") == 0.0


# ---------------------------------------------------------------- 单路信号


def test_recency_decays_with_the_event_gap() -> None:
    item = _item(seq=10, content="旧事")

    fresh = recency(item, now_seq=10, lambda_decay=0.1)
    stale = recency(item, now_seq=60, lambda_decay=0.1)

    assert fresh == pytest.approx(1.0)
    assert stale < 0.01


def test_a_bigger_lambda_forgets_faster() -> None:
    """差异化遗忘的实现点。数值一旦被调平，三个 NPC 的记忆表现就没差别了。"""
    item = _item(seq=0, content="旧事")

    flandre_like = recency(item, now_seq=20, lambda_decay=0.25)
    marisa_like = recency(item, now_seq=20, lambda_decay=0.04)

    assert flandre_like < marisa_like


def test_recency_of_a_future_item_is_clamped_instead_of_exploding() -> None:
    """重建存档时会短暂出现 now_seq 落后于条目的顺序。不钳的话指数会算出
    大于 1 的分，这一路会凭空压过其他三路。"""
    assert recency(_item(seq=99, content="x"), now_seq=0, lambda_decay=0.1) == 1.0


def test_relevance_matches_chinese_names_not_internal_ids() -> None:
    """focus 与条目正文两边都必须是中文才匹配得上——正文里不许有内部
    标识符（坑 #10），所以 focus 也不能用 id。"""
    item = _item(seq=1, content="来访者给了我 1 个枯萎的花。")

    assert relevance(item, frozenset({"枯萎的花"})) == 1.0
    assert relevance(item, frozenset({"withered_flower"})) == 0.0


def test_relevance_ignores_empty_focus_keys() -> None:
    """空串是任何字符串的子串。混进一个空 focus 键会让所有条目都「相关」。"""
    assert relevance(_item(seq=1, content="随便"), frozenset({""})) == 0.0


# ---------------------------------------------------------------- 四路融合


def test_every_signal_is_reported_separately() -> None:
    """只看总分排序的测试，在「四路里只有一路真的起作用」的实现上也会
    通过。坑 #11 那批空转测试就是这么来的。"""
    store = _store(_item(seq=5, content="来访者给了我赛钱。", salience=0.9))

    scored = score_all(store, "赛钱", _card(), now_seq=5, focus=frozenset({"赛钱"}))

    assert len(scored) == 1
    s = scored[0]
    assert s.similarity > 0.0
    assert s.recency == pytest.approx(1.0)
    assert s.salience == 0.9
    assert s.relevance == 1.0
    assert s.total == pytest.approx(s.similarity + s.recency + s.salience + s.relevance)


def test_a_salient_old_memory_can_beat_a_trivial_fresh_one() -> None:
    """这正是「情绪显著性」这一路要买的东西：只按时间排序的话，重要的
    往事永远被刚才那句闲聊压掉。"""
    store = _store(
        _item(seq=1, content="来访者砸了赛钱箱。", salience=1.0),
        _item(seq=90, content="今天天气不错。", salience=0.05),
    )

    top = retrieve(store, "赛钱箱", _card(), now_seq=100, k=1)

    assert top[0].item.seq == 1


def test_quest_focus_lifts_a_relevant_memory_over_a_similar_one() -> None:
    store = _store(
        _item(seq=50, content="来访者提过无缘塚的花。", salience=0.1),
        _item(seq=50, content="来访者提过无缘塚的天气。", salience=0.1),
    )

    top = retrieve(store, "无缘塚", _card(), now_seq=50, focus=frozenset({"花"}), k=1)

    assert "花" in top[0].item.content


def test_dormant_items_never_surface_in_normal_retrieval() -> None:
    """沉睡记忆只能被强线索召回。普通对话检索得到的话，芙兰那条线索就
    退化成「多聊几句就给你」，玩法机制没了。"""
    store = _store(_item(seq=1, content="495 年前也开过一样的花。", tier=Tier.DORMANT))

    assert retrieve(store, "花", _card(), now_seq=1) == []


def test_a_genuine_tie_is_broken_by_id_not_by_insertion_order() -> None:
    """排序键必须全序。依赖 sort 的稳定性（即插入顺序）不算确定：重建存档时
    插入顺序由动作日志决定，压缩也会重排记忆库。

    构造真同分要让四路信号完全一致。第一版这个测试用了 seq 7 和 8，而
    recency 依赖 seq，两者分数根本不相等，于是它测的不是同分（坑 #11 那类
    空转测试）。正文也不能一模一样——同内容会被检索合并成一条，于是
    tie-break 根本不会被调用。两条正文长度相同、与 query 的重叠也相同，
    所以相似度相等而内容不同。这里按 id 倒序插入，去掉排序键里的 id 就会红。
    """
    store = _store(
        _item(seq=7, content="一模一样的东西甲。"),
        _item(seq=7, content="一模一样的东西乙。"),
    )
    store.items[0].id = "m-b"
    store.items[1].id = "m-a"
    args = ("一模一样", _card(), 7)

    first = [s.item.id for s in retrieve(store, *args, k=2)]
    second = [s.item.id for s in retrieve(store, *args, k=2)]

    assert first == ["m-a", "m-b"]
    assert first == second


def test_retrieval_records_the_access() -> None:
    """访问计数影响降级——常被想起的事不容易忘。"""
    store = _store(_item(seq=1, content="来访者给了我赛钱。"))

    retrieve(store, "赛钱", _card(), now_seq=40, k=1)

    assert store.items[0].access_count == 1
    assert store.items[0].last_access_seq == 40


def test_score_all_has_no_side_effects() -> None:
    """「只是看看」的调用不该改访问计数，否则评测跑一遍指标就把记忆库
    改了，第二遍算出来的数不一样。"""
    store = _store(_item(seq=1, content="来访者给了我赛钱。"))

    score_all(store, "赛钱", _card(), now_seq=40)

    assert store.items[0].access_count == 0


# ---------------------------------------------------------------- 沉睡召回


def test_a_strong_cue_wakes_a_dormant_memory() -> None:
    store = _store(
        _item(seq=1, content="495 年前也开过一样的花。", tier=Tier.DORMANT, triggers=("枯花",))
    )

    woken = recall_dormant(store, frozenset({"枯花"}), now_seq=30)

    assert [i.content for i in woken] == ["495 年前也开过一样的花。"]
    assert store.items[0].tier is Tier.ACTIVE
    assert retrieve(store, "花", _card(), now_seq=30) != []


def test_a_wrong_cue_leaves_it_asleep() -> None:
    store = _store(_item(seq=1, content="往事。", tier=Tier.DORMANT, triggers=("枯花",)))

    assert recall_dormant(store, frozenset({"赛钱"}), now_seq=30) == []
    assert store.items[0].tier is Tier.DORMANT


def test_active_memories_are_not_touched_by_dormant_recall() -> None:
    """线索键撞上一条活跃条目时不该把它当成「召回了一段往事」——那会让
    「她想起来了」这个事件凭空多出来。"""
    store = _store(_item(seq=1, content="普通的事。", triggers=("枯花",)))

    assert recall_dormant(store, frozenset({"枯花"}), now_seq=30) == []


def test_identical_memories_do_not_each_take_a_recall_slot() -> None:
    """实测：玩家投了 4 次赛钱，四条记忆的正文一模一样，于是 4 个召回名额
    全塞成同一句「来访者给了我 1 个赛钱。」——那一段 prompt 等于只说了一件
    事，另外三件真正有用的记忆被挤掉了。

    这和坑 #19 是同一类错误：**把上下文塞进 prompt 的机制，要先看塞进去的
    内容本身有没有意义。** 防复读那段提示当时也是自己带着重复。
    """
    store = _store(
        *[_item(seq=i, content="来访者给了我 1 个赛钱。") for i in range(1, 5)],
        _item(seq=5, content="来访者问过结界的事。"),
        _item(seq=6, content="来访者提过无缘塚。"),
    )

    top = retrieve(store, "赛钱 结界 无缘塚", _card(), now_seq=6, k=4)

    contents = [s.item.content for s in top]
    assert len(contents) == len(set(contents))
    assert len(contents) == 3


def test_the_merged_count_is_reported_so_the_information_is_not_lost() -> None:
    """同一件事发生四次是四条独立记忆。合并成一条不等于把次数丢掉——
    她该记得是 4 次而不是 1 次，否则「投了几次赛钱」这个玩法数字就没了。"""
    store = _store(*[_item(seq=i, content="来访者给了我 1 个赛钱。") for i in range(1, 5)])

    top = retrieve(store, "赛钱", _card(), now_seq=4, k=4)

    assert len(top) == 1
    assert top[0].duplicates == 3
