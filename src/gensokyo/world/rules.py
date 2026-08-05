from gensokyo.world.defs import CharacterCard, RevealConditions
from gensokyo.world.state import NpcState

ATTITUDE_MIN = -100
ATTITUDE_MAX = 100

# 事件对 NPC 态度的影响。刻意做成表驱动，调平衡不用改逻辑。
# 键名按「谁做了这件事」命名，别按「谁受影响」——前者才对得上引擎里真正
# 使用它的那个分支。
ATTITUDE_DELTA: dict[str, int] = {
    "player_gave_item": 6,
    "topic_touched": 4,
}
"""好感增量。**只有玩家的行为能改这个数。**

`player_gave_item` 是**首次**送出某样东西的值，见 `GIFT_ATTITUDE_STEPS`。
`topic_touched` 是 4 而不是 6：聊天比送礼便宜，但不能便宜到让送礼失去意义。

**`npc_took_item: -8` 被删掉了。** 它原本的理由是「态度是单一的关系亲疏轴，
她不问一声就拿走东西，关系自然变差」——听起来对，实测下来是个死循环：

这个数在 prompt 里印成「你对来访者的好感」，并且门槛判的是**她愿不愿意开口**。
让她自己的行为压低它，等于「她偷了你的东西，于是她不喜欢你了」，而玩家既阻止
不了也补不回来（赛钱是有限的，投到底只有 10）。实测芙兰在六个对话回合里
`take_item` 三次，好感从 10 掉到 −2，门槛 12 再也够不到——**而她的线索是通关
必需的**：三局里两局因此卡死在 S2。

**判据：门槛依赖的值只能由玩家的行为驱动。** 偷窃的惩罚保留在玩家那一侧——
东西真的没了，而那些东西正是别的门槛要用的。
"""

GIFT_ATTITUDE_STEPS: tuple[int, ...] = (6, 3, 1)
"""**同一样东西**反复送，第 n 次的好感增量。超出长度记 0。

**这是可玩性指标直接导致的改动。** 在此之前每次送礼都是 +6，于是灵梦门槛 24
就等于「投币四次」——最优策略是敲四次 `/give` 再问一句，说话完全没有机制
价值。实测 21 回合通关里 16 回合在敲指令、NPC 只开口 5 次，而 `topic_touched`
（聊到她在意的话题涨好感）在真实对局里触发 **0 次**：机制写好了、有单测、
变异验证也过了，但它在最优策略里根本用不上。

递减到 0 而不是收敛到 1：留一个正的尾巴等于「刷得久总能刷够」，那还是同一个
磨。同一样东西的上限因此是 6+3+1 = 10。

**门槛必须跟着改**，否则递减会把线索变成拿不到的（那不是可玩性改善，是内容
丢失）。灵梦从 24 降到 16：投币到底只有 10，不够；4 个话题 16，够；2 次投币
（9）+ 2 个话题（8）= 17，够。**纯磨的路被掐死，而通关不依赖任何一条单独的
路**——坑 #6 的红线还在。
"""


def gift_attitude_delta(times_given_before: int) -> int:
    """已经送过 `times_given_before` 次同样的东西，这一次涨多少。"""
    if times_given_before < len(GIFT_ATTITUDE_STEPS):
        return GIFT_ATTITUDE_STEPS[times_given_before]
    return 0


# 情绪增量的兜底值。角色卡的 emotion.event_deltas 若给出同名条目则优先，
# 因为同一事件对不同角色的情绪方向可能相反。
EMOTION_DELTA: dict[str, float] = {
    "player_gave_item": 0.18,
}


def emotion_delta_for(card: CharacterCard, event: str) -> float:
    if event in card.emotion.event_deltas:
        return card.emotion.event_deltas[event]
    return EMOTION_DELTA.get(event, 0.0)


MODE_HYSTERESIS = 0.05
"""模式切换的迟滞带宽（施密特触发器）。

没有它，阈值上的抖动会让模式每回合翻一次。实测：玩家连续搭话时烦躁度每
回合 +0.05、回合末衰减 -0.03，恰好跨在灵梦 0.6 的门槛上来回。玩家屏幕上
于是出现「平常的懒散语气」和「不打算再理你了」同框——她在回合内越过门槛
触发了拒绝，回合末又掉回门槛以下，而面板是在衰减之后画的。

带宽 0.05 表示：进入 irritated 要到 0.65，退出要掉到 0.55 以下。
"""


def resolve_mode(card: CharacterCard, emotion: float, current: str = "") -> str:
    """情绪值 → 模式名。`current` 非空时启用迟滞。

    调用方几乎总该传 `current`：不传等于每次都按裸阈值重算，抖动就回来了。
    只有初始化（还没有「当前模式」）才该省略。
    """
    if current:
        for mode in card.emotion.modes:
            if mode.name == current and mode.contains(emotion, MODE_HYSTERESIS):
                return current
    for mode in card.emotion.modes:
        if mode.contains(emotion):
            return mode.name
    return card.emotion.modes[0].name


def bump_attitude(npc: NpcState, delta: int) -> None:
    npc.attitude = max(ATTITUDE_MIN, min(ATTITUDE_MAX, npc.attitude + delta))


def bump_emotion(npc: NpcState, card: CharacterCard, delta: float) -> None:
    npc.emotion = max(0.0, min(1.0, npc.emotion + delta))
    npc.mode = resolve_mode(card, npc.emotion, npc.mode)


def apply_emotion_decay(npc: NpcState, card: CharacterCard) -> None:
    bump_emotion(npc, card, -card.emotion.decay_per_tick)


def can_reveal(npc: NpcState, cond: RevealConditions) -> bool:
    """多个门槛之间是「与」关系，全部满足才能说出口；
    `traded_item_in` 内部是「或」，收到其中任一样即算达成。"""
    if cond.attitude_gte is not None and npc.attitude < cond.attitude_gte:
        return False
    if cond.traded_item_in:
        return bool(npc.received_items & set(cond.traded_item_in))
    return True
