"""语义相似度：可注入接口 + 零依赖默认实现。

四路检索信号里三路是纯规则（时间衰减、显著性、任务相关性），只有这一路
需要「懂意思」。把它做成可注入接口，默认实现零依赖且确定性——于是整个
记忆系统进 2.4 秒的测试套件，不需要本地服务，也不需要 mock 向量。

需要真语义时换一个实现进来即可；换的时候要先量出它在记忆探针上确实更好，
再决定是否值得多一个模型驻留（坑 #1 的教训：先分解测量，别凭直觉优化）。
"""

import math
from collections.abc import Callable

Similarity = Callable[[str, str], float]
"""(query, text) -> [0, 1]。"""


def _grams(text: str) -> set[str]:
    """中文字符 bigram。

    不做分词：多一个分词依赖换来的收益在这个规模的语料上看不出来，而
    bigram 对中文的「赛钱 / 投赛钱 / 赛钱箱」这类局部匹配已经够用。

    不足两个字的输入返回空集合，`bigram_cosine` 于是给 0 分。**这是明确的
    契约而不是漏洞**：query 由引擎拼装（任务阶段关键词 + 玩家原话），条目
    正文是模板生成的散文，两边都不可能只有一个字。补一个 unigram 回退分支
    会引入永远走不到的代码，而单字匹配（「的」命中一切）反而会拉低精度。
    单字线索走 `relevance` 那一路，它做的是子串匹配。
    """
    cleaned = "".join(text.split())
    if len(cleaned) < 2:
        return set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def bigram_cosine(query: str, text: str) -> float:
    """集合余弦：|A∩B| / sqrt(|A|·|B|)。

    用集合而不是词频向量：记忆条目都很短（截到 40 字以内），一个 bigram
    出现两次不代表它更重要，而长条目会因为词频平方项被系统性压低。
    """
    a = _grams(query)
    b = _grams(text)
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if overlap == 0:
        return 0.0
    return overlap / math.sqrt(len(a) * len(b))
