"""检索查询由引擎已知的东西拼装，不只用玩家原话。

延续本项目最核心的一条方法论：**引擎已经知道的事，不要让别人重新推导。**
当前任务阶段关心什么、她手上有什么、这里摆着什么，引擎都知道；把这些拼进
查询比指望相似度从一句「你还记得吗」里猜出来可靠得多。

`focus` 走的是另一路信号（精确子串匹配），所以它必须是**中文名字**——条目
正文里也是中文（坑 #10 清了四轮内部标识符）。两边用同一种表示才匹配得上。
"""

from gensokyo.world.observation import Observation

MEMORY_TOP_K = 4
"""每回合召回的条目数上限。

不是越多越好：召回的条目会原样进 prompt，而两阶段拆分买到的正是短 prompt
（坑 #1）。4 条够她说出一件具体的事，又不至于把场景描述挤出注意力。
"""


def build_query(obs: Observation, player_utterance: str) -> str:
    """玩家这句话 + 引擎当前的任务提示。

    只用玩家原话的话，「你还记得我给过你什么吗」这种问法里没有任何和记忆
    条目共享的字面（条目写的是「来访者给了我 3 个赛钱」），bigram 相似度
    几乎为零。把任务提示拼进来能把查询拉到剧情词汇上。
    """
    parts = [player_utterance]
    if obs.quest_hint:
        parts.append(obs.quest_hint)
    return " ".join(p for p in parts if p)


def build_focus(obs: Observation) -> frozenset[str]:
    """当前场景里出现的中文物品名 + 地点名。

    她手上的、地上的东西就是这一刻「相关」的定义。玩家刚把枯萎的花交给
    芙兰时，那条提到花的记忆必须浮上来——这正是任务相关性这一路要买的。
    """
    keys = set(obs.own_inventory) | set(obs.items_here)
    keys.add(obs.location_name)
    return frozenset(k for k in keys if k)
