"""遗忘：三级降级，同时是玩法机制。

不做删除，做降级。删除会让「她忘了」和「这件事没发生过」在数据上无法区分，
于是负例探针（问从未发生的事）测不出任何东西；降级保留源事件指针，所以永远
能回答「这件事发生过，但她想不起来了」。

**降级只读事件日志能推导出的东西。** 保留强度是

```
retention = exp(-λ_npc · (now_seq - item.seq)) · (1 + item.salience)
```

`salience` 在这里是**减缓衰减**而不是给一个下限。做成下限的话，高显著性的
条目永远停在活跃层，第三级就成了只有种子记忆才到得了的死档位——又一条空转
配置。乘法形式让三档都可达，而且排序仍然符合人设。

**刻意没有「常被想起的事不容易忘」。** 那需要把 `last_access_seq` 计入保留
强度，而检索发生在实时对话里、不在动作日志里：存档重建时没人查过记忆库，
重建出的分层会和存档那一刻不同。**一个会改变「她记得什么」的存档比一条简化
的衰减规则糟得多**（坑 #9：存档背叛玩家的实际经历）。`access_count` 与
`last_access_seq` 保留下来只作为可观测性字段，不参与任何判定。
"""

from gensokyo.memory.item import MemoryItem, MemoryStore, Tier
from gensokyo.memory.retrieve import recency
from gensokyo.world.defs import CharacterCard

COMPRESS_BELOW = 0.4
DORMANT_BELOW = 0.1

_ORDER: dict[Tier, int] = {Tier.ACTIVE: 0, Tier.COMPRESSED: 1, Tier.DORMANT: 2}
"""分层的显式顺序。

不靠 `Tier` 作为 StrEnum 的字符串比较——`active < compressed < dormant`
在字母序下**恰好**成立，而这种巧合会在有人把某一档改名时无声断掉。
"""


def retention(item: MemoryItem, now_seq: int, lambda_decay: float) -> float:
    return recency(item, now_seq, lambda_decay) * (1.0 + item.salience)


def _target_tier(strength: float) -> Tier:
    if strength < DORMANT_BELOW:
        return Tier.DORMANT
    if strength < COMPRESS_BELOW:
        return Tier.COMPRESSED
    return Tier.ACTIVE


def demote(store: MemoryStore, card: CharacterCard, now_seq: int) -> list[MemoryItem]:
    """按保留强度重算分层，返回本次降级的条目。

    只降不升：分层是单向的，一条已经压缩的记忆不会因为时间又变回原文。
    被强线索召回过的条目（`recalled`）整个跳过——否则它下一回合就会按保留
    强度睡回去，芙兰那条线索在玩家刚问出口时又消失。
    """
    changed: list[MemoryItem] = []
    for item in store.items:
        if item.recalled:
            continue
        target = _target_tier(retention(item, now_seq, card.memory.lambda_decay))
        if _ORDER[target] > _ORDER[item.tier]:
            item.tier = target
            changed.append(item)
    return changed
