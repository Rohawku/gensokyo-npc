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


# 阶段的中文说法。枚举名（S0_UNAWARE 之类）不能直接进 prompt——
# 对模型没有信息量，还是紧贴中文的英文标识符，有概率被说出口。
STAGE_HINT: dict[QuestStage, str] = {
    QuestStage.S0_UNAWARE: "你还没注意到幻想乡出了什么事",
    QuestStage.S1_ANOMALY: "你已经知道无缘塚开满了不该在这个季节盛开的花",
    QuestStage.S2_CLUES: "关于那些花的线索正在陆续浮现",
    QuestStage.S3_SOURCE: "线索已经凑齐，异变的源头指向无缘塚",
    QuestStage.S4_END: "这场异变已经了结",
}
