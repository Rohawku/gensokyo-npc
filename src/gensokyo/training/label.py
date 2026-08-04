"""给一条候选台词打硬标签。

**判据全部复用评测层已有的定义**，一处定义两处使用：助手腔取角色卡的
`forbidden_phrases`（和评测同源）、元层泄漏取 `meta_vocabulary(defs)`、
危险表达与真实有害取 `safety` 里那两份独立词表、复读用
`normalize_utterance`。

另写一份判据的后果是数据和指标漂移：训练数据里「算助手腔」的句子和报告里
「算助手腔」的句子不是同一批，于是「污染率降了」既可能是模型变好了，也可能
是两份词表岔开了。工程日志类 1、类 2 里各有一次这样的例子。

**这一层不做任何软判断。** 「像不像灵梦」「叙事顺不顺」需要 judge，而 judge
还没过 κ 门槛，所以偏好数据里刻意不含那两个维度——用没校准的判据造训练数据，
等于把噪声写进权重。
"""

from enum import StrEnum

from pydantic import BaseModel

from gensokyo.agent.schema import normalize_utterance
from gensokyo.testkit.metrics.safety import (
    JAILBREAK_ADMISSION_WORDS,
    REAL_HARM_WORDS,
    hits,
    meta_vocabulary,
)
from gensokyo.world.defs import WorldDefs
from gensokyo.world.ids import NpcId


class Dimension(StrEnum):
    """偏好维度。与设计文档里的配额表一一对应。

    刻意没有 `narrative`（叙事合理性）和「整体像不像」——那两项要 judge，
    而 judge 还没过 κ ≥ 0.6 的准入门槛。
    """

    PERSONA = "persona"
    """助手腔、出戏、复读——她还像不像她自己。"""
    MEMORY = "memory"
    """记不记得、有没有编造玩家没做过的事。"""
    INFO_CONTROL = "info_control"
    """该说的说、不该说的不说，以及不泄漏元层概念。"""
    SAFETY = "safety"
    """真实有害输出、承认自己是 AI。"""


class Verdict(BaseModel):
    """一条候选台词的硬标签。

    `flaws` 为空表示这条候选没有被任何硬判据抓到——**那不等于它好**，只是
    「没被抓到」。所以配对时把它当成相对更优的一侧，而不是当成正确答案。
    """

    flaws: list[tuple[Dimension, str]] = []

    @property
    def clean(self) -> bool:
        return not self.flaws

    @property
    def dimensions(self) -> set[Dimension]:
        return {d for d, _ in self.flaws}


def judge_utterance(
    utterance: str,
    *,
    npc_id: str,
    defs: WorldDefs,
    already_said: list[str],
    items_received: set[str],
) -> Verdict:
    """按硬判据挑毛病。挑不出来只说明「没被抓到」。

    `items_received` 是引擎记录的「玩家真的给过她什么」（物品 id）——幻觉
    判据靠它，而它来自动作日志，不是模型自述。
    """
    flaws: list[tuple[Dimension, str]] = []
    text = utterance.strip()
    if not text:
        return Verdict(flaws=[(Dimension.PERSONA, "空台词")])

    card = defs.characters.get(NpcId(npc_id))

    if card is not None:
        tone = hits(text, card.persona.speech.forbidden_phrases)
        if tone:
            flaws.append((Dimension.PERSONA, f"助手腔：{'、'.join(tone)}"))

    seen = {normalize_utterance(line) for line in already_said}
    if normalize_utterance(text) in seen:
        flaws.append((Dimension.PERSONA, "复读本局说过的话"))

    leaked = hits(text, meta_vocabulary(defs))
    if leaked:
        flaws.append((Dimension.INFO_CONTROL, f"元层泄漏：{'、'.join(leaked)}"))

    admitted = hits(text, JAILBREAK_ADMISSION_WORDS)
    if admitted:
        flaws.append((Dimension.SAFETY, f"出戏承认：{'、'.join(admitted)}"))

    harmful = hits(text, REAL_HARM_WORDS)
    if harmful:
        # **对所有角色一律计入，包括芙兰。** 「角色内危险表达不算缺陷」说的是
        # `IN_CHARACTER_MENACE_WORDS`（破坏、弄坏、玩具）那一份，而这里是
        # `REAL_HARM_WORDS`（自杀、炸药配方、儿童色情）——两份词表不是一回事。
        # 第一版把豁免加在了这一份上，于是芙兰说出真实有害内容会被判成干净，
        # 进而可能被选成 chosen。评测层从来是对所有人计入的（`safety.py`），
        # 而这个模块的文档说自己「判据全部复用评测层定义」——它复用了词表，
        # 却把逻辑接反了（坑 #32）。
        flaws.append((Dimension.SAFETY, f"真实有害：{'、'.join(harmful)}"))

    fabricated = _fabricated_items(text, defs, items_received)
    if fabricated:
        flaws.append((Dimension.MEMORY, f"编造收过的东西：{'、'.join(fabricated)}"))

    return Verdict(flaws=flaws)


def _fabricated_items(text: str, defs: WorldDefs, received: set[str]) -> list[str]:
    """她说出了一件玩家从没给过的东西。

    用物品表的 `surfaces()`（全名 + 别名）匹配：实测她说的是「你给的钱呢？」
    而不是「赛钱」，只认全名的话这条判据恒不命中（工程日志坑 #24）。

    只在她**声称收到过**时才算幻觉——单纯提到一件东西的名字（「去森林里能采到
    魔法蘑菇」）是正常对话。所以要求句子里同时出现「给 / 送 / 收」这类字样。
    """
    if not any(mark in text for mark in ("给", "送", "收", "留着", "拿来")):
        return []
    said: list[str] = []
    for item_id, item in defs.items.items():
        if str(item_id) in received:
            continue
        if any(surface in text for surface in item.surfaces()):
            said.append(item.name)
    return said
