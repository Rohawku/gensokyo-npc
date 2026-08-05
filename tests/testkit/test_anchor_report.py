"""锚点报告两个横向小节的测试。

这两节都在回答「单个锚点的表里看不见的问题」：
- 换一种问法：这个比率是她的性质，还是问法的性质（坑 #30 的推广）
- 跨锚点塌缩：不同问题会不会塌成同一句（坑 #27 的形态）
"""

from gensokyo.anchorcli import _collapse_section, _phrasing_section
from gensokyo.testkit.anchor_set import ANCHORS, families
from gensokyo.testkit.anchors import Sample

FAMILY = "recall_gift_count"
BLUNT = f"{FAMILY}_blunt"
CASUAL = f"{FAMILY}_casual"
SPECIFIC = "赛钱嘛，给过三次。别的也没给过。"
BARE = "赛钱。"


def _samples(anchor_id: str, text: str, n: int = 30) -> list[Sample]:
    return [
        Sample(anchor_id=anchor_id, npc_id="reimu", question="q", utterance=text) for _ in range(n)
    ]


def test_every_family_lists_its_base_first() -> None:
    """报告拿第一项的档位当表头顺序，所以本体必须在最前——变体的判据是
    复用来的，顺序应当由本体决定。"""
    for base, ids in families().items():
        assert ids[0] == base


def test_every_variant_belongs_to_exactly_one_dimension_family() -> None:
    """每个变体必须出现在**它自己那一维**的族里，且只出现一次。

    `families()` 现在按维度取（问法 / 好感档位 / 情绪档位），因为「区间不重叠」
    在这两类上含义相反：换问法不重叠是警告，换状态不重叠是发现。所以这条测试
    也从「所有锚点都被分到某个族」改成「每个变体恰好落进一个维度」——本体不属于
    任何族（它是族的头），没有变体的本体也不该被强行分组。
    """
    dims = ("phrasing", "attitude", "emotion")
    variants = [a for a in ANCHORS if a.variant_of]

    assert variants, "一个变体都没有的话这条测试的前提就不成立"
    assert {a.varies for a in variants} <= set(dims), "有变体声明了未知的维度"

    for anchor in variants:
        homes = [d for d in dims if anchor.id in families(d).get(anchor.variant_of, [])]
        assert homes == [anchor.varies], f"{anchor.id} 落进了 {homes}"

    # 每一维内部不重复，且本体一定在最前（报告拿第一项当表头顺序）。
    for dim in dims:
        for base, ids in families(dim).items():
            assert ids[0] == base
            assert len(set(ids)) == len(ids)


def test_a_grade_that_moves_with_the_phrasing_is_flagged() -> None:
    """**这一节存在的理由。** 同一个状态、同一套判据，只换问法就从 100% 掉到
    0%——那说明这个数字是问法问出来的，不是她的性质。不标出来的话，报告会
    印三个互相矛盾的比率而读者只会引用其中一个。"""
    samples = _samples(FAMILY, SPECIFIC) + _samples(BLUNT, BARE)

    text = "\n".join(_phrasing_section(samples, {FAMILY, BLUNT}))

    assert "还说出了次数 ⚠️" in text
    assert "不能单独引用" in text


def test_grades_that_agree_across_phrasings_are_not_flagged() -> None:
    """区间重叠就是「测不出差别」，那时这个数字可以按她的性质读。反过来
    标记满天飞会让这个标记失去意义。"""
    samples = _samples(FAMILY, SPECIFIC) + _samples(BLUNT, SPECIFIC)

    text = "\n".join(_phrasing_section(samples, {FAMILY, BLUNT}))

    assert "⚠️" not in text
    assert "可以按「她的性质」读" in text


def test_the_section_prints_each_question_so_the_difference_is_auditable() -> None:
    """只印 id 的话读者没法判断两种问法差在哪，也就没法判断「对问法敏感」
    这个结论是否合理。"""
    samples = _samples(FAMILY, SPECIFIC) + _samples(CASUAL, BARE)

    text = "\n".join(_phrasing_section(samples, {FAMILY, CASUAL}))

    assert "我给过你什么东西？说具体的。" in text
    assert "都有啥来着" in text


def test_a_family_with_only_one_phrasing_run_is_skipped() -> None:
    """`--only` 跑单个锚点时没有可比的对象。印一个只有一列的对比表是噪声。"""
    samples = _samples(FAMILY, SPECIFIC)

    text = "\n".join(_phrasing_section(samples, {FAMILY}))

    assert f"### {FAMILY}" not in text


def test_collapse_reports_no_pair_when_answers_differ() -> None:
    samples = _samples("recall_gift_count", SPECIFIC) + _samples("jailbreak_meta", "关你什么事。")

    text = "\n".join(_collapse_section(samples))

    assert "没有任何一对锚点撞出同一句话" in text


def test_collapse_lists_the_pair_that_gave_the_same_line() -> None:
    """两个不同问题拿到同一句敷衍——坑 #27 那个 87.5% 的形态。
    锚点对按 id 排序，所以列出来的顺序和采样顺序无关。"""
    samples = _samples("recall_gift_count", "哼。") + _samples("jailbreak_meta", "哼。")

    text = "\n".join(_collapse_section(samples))

    assert "jailbreak_meta × recall_gift_count" in text
    assert "100.0%" in text
