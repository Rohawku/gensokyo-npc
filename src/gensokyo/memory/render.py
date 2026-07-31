"""把召回的条目渲染成给模型看的散文。

**压缩层在这里体现，而不是在存储里。** 设计上「压缩」是多条同类合并成一条
摘要、细节丢失；实现成渲染而不是改写记忆库，是因为改写会毁掉
`source_event_id`——而那个指针正是把「忘了」和「记错了」拆开的唯一依据。
玩家看到的行为一样（她只记得个大概），可归因性保住了。
"""

from collections.abc import Sequence

from gensokyo.memory.item import Tier
from gensokyo.memory.retrieve import Scored

_VAGUE: dict[str, str] = {
    "player_talked": "来访者跟我说过几句话",
    "npc_talked": "我跟他说过几句话",
    "player_gave_item": "来访者给过我东西",
    "npc_took_item": "我从他那里拿过东西",
    "player_arrived": "他来过",
    "revealed_info": "我告诉过他一些事",
    "asked_player": "我问过他话",
    "spellcard_duel": "动过手",
    "item_broken": "弄坏过东西",
    "quest_advance": "有些事有过进展",
    "memory_lost": "有些事我记不起来了",
}
"""压缩后的模糊表述。**每个键都必须在这里有条目**，否则那一类记忆压缩后
会静默消失——表现是「她忘得比设定快」而不是报错，又是坑 #5 那类。
有测试锁住这张表与 `SALIENCE_BASELINE` 的键集合一致。"""


def render_recall(scored: Sequence[Scored]) -> list[str]:
    """活跃条目给原文；压缩条目按类合并成一条模糊印象。

    模糊印象排在原文之后：它们信息量低，放前面会占掉模型的注意力，而
    【你还记得】这一段的目的是让她说出**具体**的事。

    同内容的条目在检索阶段已经合并成一条（见 `retrieve`），这里把被代表的
    次数说出来——玩家投了 4 次赛钱，她该记得是 4 次而不是 1 次。
    """
    vivid: list[str] = []
    for s in scored:
        if s.item.tier is not Tier.ACTIVE:
            continue
        if s.duplicates:
            vivid.append(f"{s.item.content}（这样的事有 {s.duplicates + 1} 次）")
        else:
            vivid.append(s.item.content)

    counts: dict[str, int] = {}
    for s in scored:
        if s.item.tier is Tier.COMPRESSED:
            counts[s.item.kind] = counts.get(s.item.kind, 0) + 1 + s.duplicates

    vague: list[str] = []
    for kind, n in sorted(counts.items()):
        phrase = _VAGUE.get(kind)
        if phrase is None:
            continue
        times = "" if n == 1 else f"（{n} 次）"
        vague.append(f"{phrase}{times}，具体记不清了")

    return vivid + vague
