"""judge 与人工标注的一致性，以及准入门槛。

**这个模块存在的理由是把一条纪律变成代码。** 设计文档写着「某维度 κ < 0.6，
该维度的数字不得用于下结论」——写在文档里的纪律会漂移，写成 `Agreement.admissible`
才不会：报告直接问它，而不是让读者自己去查文档。

**为什么必须有盲标对照组。** 这个项目只有两个可用模型：policy 是本地
`qwen3:8b`，而规格要求 judge ≠ policy，所以 judge 只能是外部模型。于是预标注
（降低人工成本）和 judge 只能是同一个模型，人工复核后的标签就带着 judge 的
先验——**锚定偏差会让 κ 虚高**。

对策是把样本分两半：一半给预标注后复核，一半盲标。两组各算一个 κ，**以盲标
组为准入依据**，并把两者的差报出来当作锚定偏差的量。差得大就说明预标注污染了
标签，那时预标注省下的人力是拿结论的可信度换的。
"""

from enum import StrEnum

from pydantic import BaseModel

KAPPA_ADMISSION = 0.6
"""准入门槛，取自设计文档。**低于它该维度的数字不得用于下结论。**

0.6 是 Landis & Koch 那套惯例里「中等到较好」的分界。这里刻意不调松：judge
判的是叙事合理性与「像不像」，而那两项本来就是这套评测里最软的，门槛松了
等于让最不可靠的数字进结论。
"""


class Verdict(StrEnum):
    """成对比较的结果。**刻意不用 1–5 分**——规格里「分数聚集」那一条的对策。

    绝对分数在 LLM 上会挤在 3~4 之间，方差小到分不出东西；而成对偏好只要求
    「哪个更好」，人和模型都答得稳。
    """

    LEFT = "left"
    RIGHT = "right"
    TIE = "tie"
    """平手是**真实答案之一**，不是弃权。强迫二选一会把「这两条差不多」压成
    一个随机方向，而那种噪声会直接压低 κ——那时低 κ 反映的是被迫猜测，不是
    judge 不准。"""


class Agreement(BaseModel):
    """一个维度上 judge 与人工的一致性。"""

    dimension: str
    pairs: int
    """有效比较对数。judge 自相矛盾的那些不计入——见 `judge_flipped`。"""
    agreed: int
    kappa: float
    judge_flipped: int = 0
    """双向交换后 judge 给出相反结论的次数（位置偏差的直接观测）。

    这些对子被排除在 κ 之外：一个连自己都不稳定的判定，和人工比一致性没有
    意义。**但必须报出来**——它大到某个程度，说明 judge 在这个维度上根本
    不可用，而那时 κ 高也只是剩下那些「容易的」对子造成的假象。
    """

    @property
    def observed(self) -> float:
        return self.agreed / self.pairs if self.pairs else 0.0

    @property
    def admissible(self) -> bool:
        """这个维度的数字能不能用于下结论。

        `pairs == 0` 时返回 False：没有标注不等于通过门槛。
        """
        return self.pairs > 0 and self.kappa >= KAPPA_ADMISSION

    def __str__(self) -> str:
        gate = "可用" if self.admissible else f"**不可用**（门槛 {KAPPA_ADMISSION}）"
        return (
            f"{self.dimension}：κ={self.kappa:.3f}、一致 {self.observed:.1%}"
            f"（n={self.pairs}，judge 自相矛盾 {self.judge_flipped}）→ {gate}"
        )


def cohens_kappa(judge: list[Verdict], human: list[Verdict]) -> float:
    """Cohen's κ：扣掉「碰巧一致」之后的一致性。

    **不能用裸一致率代替。** 三分类里瞎猜也有约 1/3 一致；而如果人工标注
    七成是平手，一个「永远答平手」的 judge 能拿到 70% 一致率却毫无信息——
    κ 会把它算成接近 0。

    完全一致且双方都只用了一个类别时返回 1.0：那时期望一致率也是 1，
    κ 的定义式是 0/0。约定为 1.0 而不是 0.0——两个标注者逐条相同，说它们
    毫无一致性是荒谬的。这个分支在「全部平手」的小样本上真的会走到。
    """
    if not judge or len(judge) != len(human):
        return 0.0

    n = len(judge)
    observed = sum(1 for j, h in zip(judge, human, strict=True) if j == h) / n
    expected = sum((judge.count(v) / n) * (human.count(v) / n) for v in Verdict)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def correction_rate(pre_annotated: list[Verdict], final: list[Verdict]) -> float:
    """人工把预标注改掉的比例。

    **这个数太低才是警报。** 它接近 0 说明复核者基本在点「同意」，那份标签
    实际上还是 judge 自己的——κ 会虚高。它不是质量指标，是「复核有没有真的
    发生」的体检项，和零召回回合是同一类东西。
    """
    if not pre_annotated or len(pre_annotated) != len(final):
        return 0.0
    changed = sum(1 for p, f in zip(pre_annotated, final, strict=True) if p != f)
    return changed / len(pre_annotated)


class AnchoringCheck(BaseModel):
    """预标注组与盲标组的 κ 之差——锚定偏差的量。

    **准入看盲标那一组**，不看合并值。合并会把被污染的那一半掺进结论，而
    污染的方向恰好是让 κ 变好。
    """

    primed: Agreement
    blind: Agreement
    correction_rate: float

    @property
    def gap(self) -> float:
        return self.primed.kappa - self.blind.kappa

    @property
    def admissible(self) -> bool:
        return self.blind.admissible

    def __str__(self) -> str:
        return (
            f"盲标 κ={self.blind.kappa:.3f}（准入依据）、"
            f"预标注后复核 κ={self.primed.kappa:.3f}、差 {self.gap:+.3f}；"
            f"人工修正率 {self.correction_rate:.1%} → "
            f"{'可用' if self.admissible else '**不可用**'}"
        )
