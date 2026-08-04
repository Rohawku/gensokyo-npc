"""检索：四路信号融合，衰减率角色化。

```
score = w1·sim(query, item.content)
      + w2·exp(-λ_npc · (now_seq - item.seq))
      + w3·item.salience
      + w4·relevance(item, focus)
```

基础形态取自 Generative Agents 的 retrieval score，本项目做了两处改动：

**λ 与 salience 角色化。** 原文全局统一。这里芙兰 λ 大（忘得快）、魔理沙
λ 小（记得谁欠她书）、灵梦居中，参数从人设推导而不是调出来的。这是差异化
遗忘的实现点——数值一旦被调平，三个 NPC 的记忆表现就没差别了。

**加入任务相关性项。** 纯语义与情感检索会捞出「有感情但对当前剧情无用」的
条目。NPC 记忆要服务剧情推进，检索目标不只是像人。
"""

import math
from dataclasses import dataclass, replace

from gensokyo.memory.item import MemoryItem, MemoryStore, Tier
from gensokyo.memory.similarity import Similarity, bigram_cosine
from gensokyo.world.defs import CharacterCard

W_SIMILARITY = 1.0
W_RECENCY = 1.0
W_SALIENCE = 1.0
W_RELEVANCE = 1.0
"""四路等权。

刻意不调权重：一旦开始调，就必须说清楚是照着什么指标调的，而 Phase B
的记忆探针（事实召回 / 负例否认 / 矛盾检出）还没跑出基线。等权是可解释的
出发点，调权重之前先有数字——否则调出来的是过拟合当前那几局对话。
"""


@dataclass(frozen=True)
class Scored:
    """带四路分解的检索结果。

    分解是必需的而不是调试装饰：只看总分排序对不对的测试，在「四路里只有
    一路真的起作用」的实现上也会通过。坑 #11 那批空转测试就是这么来的。
    """

    item: MemoryItem
    similarity: float
    recency: float
    salience: float
    relevance: float
    duplicates: int = 0
    """还有多少条同内容的记忆被这一条代表了。

    是**检索结果**的属性而不是记忆的属性：同一件事发生四次就是四条独立
    记忆，只是没必要在 prompt 里说四遍。
    """

    @property
    def total(self) -> float:
        return (
            W_SIMILARITY * self.similarity
            + W_RECENCY * self.recency
            + W_SALIENCE * self.salience
            + W_RELEVANCE * self.relevance
        )


def recency(item: MemoryItem, now_seq: int, lambda_decay: float) -> float:
    """exp(-λ·Δ)，Δ 的单位是**事件**而不是回合。

    未来的条目（now_seq 小于 item.seq）钳到 1.0 而不是让指数爆掉：重建
    存档时会短暂出现这种顺序。
    """
    gap = max(0, now_seq - item.seq)
    return math.exp(-lambda_decay * gap)


def relevance(item: MemoryItem, focus: frozenset[str]) -> float:
    """任务相关性，纯规则：条目正文提到当前阶段关心的东西则为 1。

    `focus` 是**中文名字**而不是内部 id——条目正文里也是中文（坑 #10），
    两边必须用同一种表示才能匹配上。由 agent 层从引擎的任务状态构造。
    """
    return 1.0 if any(key and key in item.content for key in focus) else 0.0


def score_all(
    store: MemoryStore,
    query: str,
    card: CharacterCard,
    now_seq: int,
    focus: frozenset[str] = frozenset(),
    similarity: Similarity = bigram_cosine,
) -> list[Scored]:
    """给所有非沉睡条目打分，不排序不截断。

    沉睡条目排除在外是设计：它们只能被强线索召回（见 `recall_dormant`），
    普通对话检索不到——这是芙兰那条线索的机制基础。
    """
    return [
        Scored(
            item=item,
            similarity=similarity(query, item.content),
            recency=recency(item, now_seq, card.memory.lambda_decay),
            salience=item.salience,
            relevance=relevance(item, focus),
        )
        for item in store.active()
    ]


def retrieve(
    store: MemoryStore,
    query: str,
    card: CharacterCard,
    now_seq: int,
    focus: frozenset[str] = frozenset(),
    k: int = 5,
    similarity: Similarity = bigram_cosine,
) -> list[Scored]:
    """取分数最高的 k 条，**同内容只占一个名额**，并记一次访问。

    去重是必须的而不是优化：实测玩家投了 4 次赛钱，四条记忆的正文一模一样
    （「来访者给了我 1 个赛钱。」），于是 4 个召回名额全塞成同一句话，
    prompt 里那一段等于只说了一件事。这和坑 #19 是同一类错误——**把上下文
    塞进 prompt 的机制，要先看塞进去的内容本身有没有意义。**

    被代表的条数记在 `duplicates` 上：同一件事发生四次是四条独立记忆，
    渲染时会说成「这样的事有 4 次」，信息没丢。

    访问计数在这里记，所以检索**有副作用**。「只是看看」的调用用 `score_all`。

    **相似度最高的那条无条件占一个名额**（坑 #34）。四路等权只在权重上成立，
    在数值上不成立：相似度在中文短句上的典型取值是 0.05~0.15，而最近性在相邻
    条目之间的落差就有 0.27——等权实际是 5:1 压倒，相似度进不了排序。实测形态是
    玩家改口（先说「我叫甲」再说「我叫乙」）时，那条唯一相关的记忆是全场唯一
    相似度非零的条目，却排在 k 之外，于是**要比对的那句话压根没进 prompt**。

    做成保底名额而不是调 `W_SIMILARITY`：调权重得说清照着什么指标调、而且会改变
    所有召回的行为，而这是个局部问题——「最像当前这句话的那条必须在场」。保底
    名额占的是 k 里的一个位，不额外加长 prompt（坑 #1 买到的短 prompt 要守住）。
    """
    scored = score_all(store, query, card, now_seq, focus, similarity)
    # 同分按 id 排，保证全序。依赖 sort 的稳定性（即插入顺序）不算确定：
    # 重建存档时条目的插入顺序由动作日志决定，而记忆库也可能被压缩重排。
    # 排序键里刻意**不**放 seq——recency 是 seq 的严格减函数，同分且 seq
    # 不同意味着另外三路正好补偿到小数点后若干位，那个分支实际不可达。
    scored.sort(key=lambda s: (-s.total, s.item.id))

    seen: dict[str, int] = {}
    order: list[Scored] = []
    for s in scored:
        if s.item.content in seen:
            seen[s.item.content] += 1
            continue
        seen[s.item.content] = 0
        order.append(s)

    picked = order[:k]
    # 相似度冠军。`scored` 已是全序，而 max 取首个最大值，所以相似度并列时拿的
    # 是总分最高的那条——确定的。
    #
    # 这里**不需要**「相似度大于 0 才占位」那个守卫：全场相似度相等（比如全是 0）
    # 时冠军就是总分第一，它本来就在 picked 里，替换不会发生。第一版写了那个
    # 条件，突变验证时发现把它改成恒真、测试全绿——一个不可达的守卫（坑 #4、
    # #20 那一类：写了但不起作用）。
    closest = max(scored, key=lambda s: s.similarity, default=None)
    if closest is not None:
        champion = next(o for o in order if o.item.content == closest.item.content)
        if champion not in picked:
            picked = [*picked[: k - 1], champion]

    top = [replace(s, duplicates=seen[s.item.content]) for s in picked]
    for s in top:
        s.item.access_count += 1
        s.item.last_access_seq = now_seq
    return top


def recall_dormant(store: MemoryStore, cues: frozenset[str], now_seq: int) -> list[MemoryItem]:
    """强线索召回沉睡记忆。命中即转回活跃——想起来了就是想起来了。

    `cues` 是当下出现的线索键（比如玩家刚交给她的物品 id），与条目自己声明的
    `trigger_keys` 做**精确键匹配**而不是相似度：这条路径直接决定一条线索能
    不能拿到（芙兰的那段往事），而可通关性不该依赖一个相似度阈值。坑 #6 是
    「游戏做出来不可通关」，同类风险这里必须避开。
    """
    woken: list[MemoryItem] = []
    for item in store.dormant():
        if cues & set(item.trigger_keys):
            item.tier = Tier.ACTIVE
            item.recalled = True
            item.access_count += 1
            item.last_access_seq = now_seq
            woken.append(item)
    return woken
