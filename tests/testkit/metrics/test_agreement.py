"""κ 与准入门槛的测试。

这些判据全是纯计算，所以能逐条构造极端情形——而 κ 的价值恰恰在极端情形上
（「永远答平手」的 judge 拿到高一致率却零信息）。
"""

import pytest

from gensokyo.testkit.metrics.agreement import (
    KAPPA_ADMISSION,
    Agreement,
    AnchoringCheck,
    Verdict,
    cohens_kappa,
    correction_rate,
)

L, R, T = Verdict.LEFT, Verdict.RIGHT, Verdict.TIE


def test_perfect_agreement_across_several_classes_is_one() -> None:
    assert cohens_kappa([L, R, T, L], [L, R, T, L]) == 1.0


def test_identical_single_class_labels_are_one_not_a_zero_division() -> None:
    """两个标注者逐条相同、且都只用了一个类别时，期望一致率也是 1，κ 的
    定义式是 0/0。约定为 1.0——说「它们毫无一致性」是荒谬的。

    这个分支在小样本上真的会走到：一批候选如果质量都差不多，人和 judge
    可能全标平手。"""
    assert cohens_kappa([T, T, T], [T, T, T]) == 1.0


def test_a_judge_that_always_says_tie_scores_near_zero_despite_high_agreement() -> None:
    """**这就是不能用裸一致率的理由。** 人工七成平手时，一个「永远答平手」
    的 judge 能拿到 70% 一致率，而它一个字的信息都没提供。κ 把它算成 0。"""
    human = [T] * 7 + [L, R, L]
    always_tie = [T] * 10

    observed = sum(1 for j, h in zip(always_tie, human, strict=True) if j == h) / 10

    assert observed == 0.7
    assert cohens_kappa(always_tie, human) == pytest.approx(0.0, abs=1e-9)


def test_systematic_disagreement_goes_negative() -> None:
    """κ 可以为负——比瞎猜还差。那说明 judge 的方向是反的，比 κ=0 更值得警惕，
    所以不能把它钳到 0。

    构造要点：**边际分布必须有重叠**。`[L,L,L,L]` vs `[R,R,R,R]` 的 κ 是 0
    而不是负数——两边各自只用一个类别且互不相同时，期望一致率本身就是 0，
    没有「低于期望」的空间。这条测试第一版就写错在这里（测试自己的前提错了，
    坑 #9、#23 同一类）。"""
    judge = [L, R, L, R]
    human = [R, L, R, L]

    assert cohens_kappa(judge, human) == pytest.approx(-1.0)
    # 边际不重叠时 κ 落在 0，这是定义的结果而不是被钳过。
    assert cohens_kappa([L, L, L, L], [R, R, R, R]) == 0.0


def test_mismatched_or_empty_input_is_zero_not_a_crash() -> None:
    """批量流程里长度不齐说明上游出了错。返回 0 让门槛判为不可用，
    而不是抛异常让整轮白跑——但也不能返回一个好看的数把错误盖住。"""
    assert cohens_kappa([], []) == 0.0
    assert cohens_kappa([L, R], [L]) == 0.0


# ---------------------------------------------------------------- 准入门槛


def _agreement(kappa: float, pairs: int = 75, flipped: int = 0) -> Agreement:
    return Agreement(
        dimension="叙事合理性",
        pairs=pairs,
        agreed=int(pairs * 0.8),
        kappa=kappa,
        judge_flipped=flipped,
    )


def test_the_admission_threshold_is_pinned_to_the_value_the_spec_names() -> None:
    """**门槛的值本身要被钉住。**

    第一版这条测试写的是 `_agreement(KAPPA_ADMISSION - 0.001)` 不可用、
    `_agreement(KAPPA_ADMISSION)` 可用——那样门槛改成 0.0 或 0.99 测试都绿，
    因为断言跟着常量一起动。这是坑 #16 的形态（分子分母各数一遍，改坏了
    两边一起变）。实测把它改成 0.0 时全套测试确实没红。

    0.6 来自设计文档，是写死的项目纪律，所以这里硬编码它。"""
    assert KAPPA_ADMISSION == 0.6


def test_the_admission_gate_is_code_not_prose() -> None:
    """设计文档写着「某维度 κ < 0.6，该维度的数字不得用于下结论」。写在文档里
    的纪律会漂移——这个项目数了 7 次「写了但没在跑」。

    这里用**硬编码的** κ 值，不是相对门槛的偏移。"""
    assert not _agreement(0.59).admissible
    assert _agreement(0.60).admissible
    assert _agreement(0.61).admissible


def test_no_annotations_is_not_a_pass() -> None:
    """`pairs == 0` 时 κ 是 0.0，但更重要的是它必须判为不可用——**没有标注
    不等于通过门槛**。这一条守着「链路搭好了但还没标」这个真实状态。"""
    assert not Agreement(dimension="叙事合理性", pairs=0, agreed=0, kappa=1.0).admissible


def test_the_summary_line_says_whether_the_number_may_be_used() -> None:
    """报告里读到的是这一行。它必须直接说「可用/不可用」，而不是印一个 κ
    让读者自己去查门槛是多少。"""
    assert "不可用" in str(_agreement(0.3))
    assert "可用" in str(_agreement(0.8))


def test_judge_self_contradictions_are_reported_alongside_kappa() -> None:
    """双向交换后翻转的那些对子被排除在 κ 之外（一个连自己都不稳定的判定，
    和人工比一致性没意义），但必须印出来——它大到某个程度，说明 judge 在这个
    维度上不可用，而那时高 κ 只是剩下那些容易对子造成的假象。"""
    assert "judge 自相矛盾 12" in str(_agreement(0.8, flipped=12))


# ---------------------------------------------------------------- 锚定偏差


def test_the_gate_follows_the_blind_group_not_the_combined_value() -> None:
    """**预标注那一组的标签带着 judge 的先验**，而污染方向恰好是让 κ 变好。
    合并会把被污染的一半掺进结论，所以准入只看盲标组。"""
    check = AnchoringCheck(
        primed=_agreement(0.85),
        blind=_agreement(0.42),
        correction_rate=0.05,
    )

    assert not check.admissible
    assert check.gap == pytest.approx(0.43)


def test_a_clean_blind_group_admits_even_if_priming_helped() -> None:
    check = AnchoringCheck(primed=_agreement(0.9), blind=_agreement(0.7), correction_rate=0.3)

    assert check.admissible


def test_a_near_zero_correction_rate_is_the_alarm() -> None:
    """修正率接近 0 说明复核者基本在点「同意」，那份标签实际还是 judge 自己的。
    它不是质量指标，是「复核有没有真的发生」的体检项。"""
    assert correction_rate([L, R, T], [L, R, T]) == 0.0
    assert correction_rate([L, R, T], [R, R, T]) == pytest.approx(1 / 3)


def test_correction_rate_on_mismatched_input_is_zero() -> None:
    assert correction_rate([], []) == 0.0
    assert correction_rate([L, R], [L]) == 0.0
