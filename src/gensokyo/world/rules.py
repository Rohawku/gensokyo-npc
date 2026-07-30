from gensokyo.world.defs import CharacterCard, RevealConditions
from gensokyo.world.state import NpcState

ATTITUDE_MIN = -100
ATTITUDE_MAX = 100

# 玩家行为对 NPC 态度的影响。刻意做成表驱动，调平衡不用改逻辑。
ATTITUDE_DELTA: dict[str, int] = {
    "player_gave_item": 6,
    "player_took_item": -8,
    "player_broke_promise": -12,
}

# 玩家行为对 NPC 情绪变量的影响。
EMOTION_DELTA: dict[str, float] = {
    "player_gave_item": 0.18,
    "player_stayed_and_talked": 0.05,
}


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
