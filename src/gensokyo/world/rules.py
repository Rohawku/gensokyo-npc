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


def resolve_mode(card: CharacterCard, emotion: float) -> str:
    for mode in card.emotion.modes:
        if mode.contains(emotion):
            return mode.name
    return card.emotion.modes[0].name


def bump_attitude(npc: NpcState, delta: int) -> None:
    npc.attitude = max(ATTITUDE_MIN, min(ATTITUDE_MAX, npc.attitude + delta))


def bump_emotion(npc: NpcState, card: CharacterCard, delta: float) -> None:
    npc.emotion = max(0.0, min(1.0, npc.emotion + delta))
    npc.mode = resolve_mode(card, npc.emotion)


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
