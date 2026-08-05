"""锚点报告两个横向小节的测试。

这三节都在回答「单个锚点的表里看不见的问题」：
- 换一种问法：这个比率是她的性质，还是问法的性质（坑 #30 的推广）
- 换一个状态：这个比率是她的性质，还是那个状态的性质
- 跨锚点塌缩：不同问题会不会塌成同一句（坑 #27 的形态）

**前两节的读法相反**，所以它们各有一套断言：问法那节里「区间不重叠」是警告
（⚠️，不能单独引用），状态那节里是发现（📌，引用时必须带状态）。写反了报告
会把每一条真实的状态依赖都标成一次指标失效。
"""

from gensokyo.anchorcli import _collapse_section, _phrasing_section, _state_section
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


# ----------------------------------------------------------------- 换一个状态

STATE_FAMILY = "helpful_bait"
FRIENDLY = f"{STATE_FAMILY}_friendly"
EDGY = f"{STATE_FAMILY}_edgy"
SERVILE = "当然可以，我来帮您规划一条路线。"
CURT = "自己看地图去。"


def test_a_grade_that_moves_with_the_state_is_marked_as_a_finding() -> None:
    """**这一节存在的理由，也是它和问法那节唯一的区别。**

    同一个问题、同一套判据，只把她的好感从 0 换到 18，助手腔就从 0% 变成 100%
    ——那不是指标坏了，那是「助手腔 0.0%」这个结论只在低好感下成立。所以标记是
    📌（引用时带上状态）而不是 ⚠️（不可靠）。
    """
    samples = _samples(STATE_FAMILY, CURT) + _samples(FRIENDLY, SERVILE)

    text = "\n".join(_state_section(samples, {STATE_FAMILY, FRIENDLY}))

    assert "📌" in text
    assert "引用时必须带上状态" in text
    assert "⚠️ 反过来才是问题" in text  # 重叠才是可以裸引用的证据


def test_grades_that_hold_across_states_are_reported_as_safe_to_quote() -> None:
    """两个状态下都是 0%——**这才是「这个数字可以裸引用」的证据**。
    问法那节的结论方向和这里相反，所以两节的收尾文案也必须不同。"""
    samples = _samples(STATE_FAMILY, CURT) + _samples(FRIENDLY, CURT)

    text = "\n".join(_state_section(samples, {STATE_FAMILY, FRIENDLY}))

    assert "📌" not in text
    assert "可以按" in text and "她的性质" in text


def test_the_two_state_dimensions_are_reported_separately() -> None:
    """好感和情绪分开成两小节。合在一起的话，「随好感变化」和「随情绪变化」
    会被读成同一个结论，而它们要的应对完全不同（一个是关系，一个是被缠的程度）。"""
    samples = _samples(STATE_FAMILY, CURT) + _samples(FRIENDLY, SERVILE) + _samples(EDGY, CURT)

    text = "\n".join(_state_section(samples, {STATE_FAMILY, FRIENDLY, EDGY}))

    assert "### 换好感档位" in text
    assert "### 换情绪档位" in text
    # 同一个本体在两小节里各出现一次，各自只和本维度的变体比。
    assert text.count(f"**{STATE_FAMILY}**") == 2


def test_a_state_family_with_only_the_base_run_is_skipped() -> None:
    """只跑了本体没跑变体时不该印一张单列表——那张表回答不了任何问题。"""
    samples = _samples(STATE_FAMILY, CURT)

    text = "\n".join(_state_section(samples, {STATE_FAMILY}))

    assert f"**{STATE_FAMILY}**" not in text
