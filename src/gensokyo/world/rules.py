from gensokyo.world.defs import CharacterCard, RevealConditions
from gensokyo.world.state import NpcState

ATTITUDE_MIN = -100
ATTITUDE_MAX = 100

# 事件对 NPC 态度的影响。刻意做成表驱动，调平衡不用改逻辑。
# 本项目只有一个态度轴，语义是「关系亲疏」：NPC 不问一声就从玩家手里
# 拿走东西，关系自然变差。键名按「谁做了这件事」命名，别按「谁受影响」——
# 前者才对得上引擎里真正使用它的那个分支。
ATTITUDE_DELTA: dict[str, int] = {
    "player_gave_item": 6,
    "npc_took_item": -8,
}

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
