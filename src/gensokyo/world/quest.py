from gensokyo.world.defs import WorldDefs
from gensokyo.world.ids import LocationId
from gensokyo.world.state import QuestStage, WorldState

ANOMALY_SITE = LocationId("muenzuka")


def compute_stage(state: WorldState, defs: WorldDefs) -> QuestStage:
    """由世界状态推导应处于的阶段。纯函数，无副作用，因此可反复调用。"""
    clues = defs.clue_facts()
    obtained = state.player.known_facts & clues

    if len(obtained) == len(clues) and clues:
        return QuestStage.S3_SOURCE
    if obtained:
        return QuestStage.S2_CLUES
    if state.player.location == ANOMALY_SITE or state.quest.stage >= QuestStage.S1_ANOMALY:
        return QuestStage.S1_ANOMALY
    return QuestStage.S0_UNAWARE
