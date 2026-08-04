"""相似度那一路的直接测试。

**四路信号里只有这一路需要「懂意思」**，其余三路是纯规则（时间衰减、显著度、
任务相关性）。所以它是唯一做成可替换接口的一路，默认实现是零依赖的中文字符
bigram 余弦——整个记忆系统因此进得了那个不到 1 秒的测试套件。

这个文件同时记录一条**走不通的路**：见最后那条测试。
"""

from gensokyo.memory.similarity import bigram_cosine
from gensokyo.testkit.personas import CONTRADICTION_PAIRS


def test_a_string_is_maximally_similar_to_itself() -> None:
    assert bigram_cosine("我叫甲，你记住了。", "我叫甲，你记住了。") == 1.0


def test_nothing_shared_scores_zero() -> None:
    assert bigram_cosine("赛钱", "魔法") == 0.0


def test_the_measure_is_symmetric() -> None:
    """检索里查询在左、条目正文在右。不对称的话「查询像条目」和「条目像查询」
    是两个分数，而排序用的是哪一个就成了实现细节。"""
    a, b = "我没去过无缘塚。", "我昨天才从无缘塚回来。"

    assert bigram_cosine(a, b) == bigram_cosine(b, a)


def test_an_empty_string_is_similar_to_nothing() -> None:
    """端点抽风会返回空串。除零必须给 0 而不是崩掉——批量采集里一次崩溃
    会让整轮白跑。"""
    assert bigram_cosine("", "我叫甲") == 0.0
    assert bigram_cosine("", "") == 0.0


def test_a_single_character_has_no_bigrams() -> None:
    """一个字构不成 bigram。这不是缺陷，是这个度量的已知盲区——中文单字
    发言（「嗯」「哦」）在这一路上恒为 0，靠其余三路托着。"""
    assert bigram_cosine("嗯", "嗯") == 0.0


def test_similarity_cannot_tell_a_contradiction_from_two_unrelated_claims() -> None:
    """**这条测试记录的是一条走不通的路，不是一个缺陷。**

    矛盾检出实测只有 5.0%，而坑 #37 之后剩下的方案是：引擎发现「玩家当前这句
    和某句历史发言高度相似但不相同」时给一条【留神】指令，判断留给模型。那样
    就避开了坑 #33、#36 的陷阱（引擎不判定矛盾，只提示）。

    动手之前先量判据有没有区分度——结果是没有：

        对内（声明 ⟷ 它自己的翻供）：0.000  0.094  0.191  0.200  0.228  0.239
        对间（翻供 ⟷ 别对的声明）最大：0.224
          「我叫甲，你记住了。」⟷「我在幻想乡住了十年了。」——只因为都有「我」「了」

    **两个分布重叠，不存在能分开它们的阈值。** 阈值取 0.19 抓到 3/6 且误报一组；
    取 0.24 无误报但只抓到 1/6。而「那些花我一朵都没碰过」和「我摘了一大把花带
    在身上」对内相似度是 **0.000**——判据完全看不见它。

    结论和坑 #33 一样：bigram 相似度证明的是共享字面，不是「同一件事的相反
    陈述」。而误报在这里的代价比漏报更高——她会去指出一个不存在的矛盾。所以
    这条路不做，见取舍 #11。
    """
    within = sorted(bigram_cosine(claim, flip) for claim, flip in CONTRADICTION_PAIRS)
    cross = [
        bigram_cosine(flip, other_claim)
        for i, (_, flip) in enumerate(CONTRADICTION_PAIRS)
        for j, (other_claim, _) in enumerate(CONTRADICTION_PAIRS)
        if i != j
    ]

    # 对间最大值落在对内取值范围之内——两个分布重叠，没有阈值能分开它们。
    assert max(cross) > within[0]
    assert max(cross) > within[len(within) // 2]
