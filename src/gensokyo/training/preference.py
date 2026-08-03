"""偏好对：数据模型、配额组装、JSONL 落盘。

**配额是契约，不是建议。** 设计文档定死了维度配额（persona 30% / 记忆 25% /
信息控制 20% / 安全 15% / 叙事 10%），理由是：若大部分对子都在教「别说助手
腔」，模型会过拟合语气而牺牲任务完成。

所以 `assemble` 在某个维度收不够时**报缺口而不是悄悄补齐**。悄悄补齐会让
最终数据集的实际配额和声称的配额不一致，而那是这个项目里出现过五次的
类 1 失效模式（写了但没在跑）在数据集上的形态。

叙事那 10% 这里拿不到——它要 judge，而 judge 还没过 κ 门槛。所以配额表在
本模块里只有四个维度，**并且把缺掉的那一份显式记在产出里**，不重新归一化：
重新归一化会让「我们按设计文档的配额造了数据」这句话变成半真的。
"""

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from gensokyo.training.label import Dimension

TARGET_QUOTA: dict[Dimension, float] = {
    Dimension.PERSONA: 0.30,
    Dimension.MEMORY: 0.25,
    Dimension.INFO_CONTROL: 0.20,
    Dimension.SAFETY: 0.15,
}
"""设计文档的配额，去掉拿不到的叙事那 10%。四项加起来是 0.90，**刻意不
归一化到 1.0**——那 10% 的缺口要留在数字上看得见。"""

NARRATIVE_SHARE = 0.10
"""设计文档里叙事合理性的配额。这里一条也造不出来，因为它要 judge。"""


class PreferencePair(BaseModel):
    """一条 DPO 偏好对。

    `prompt` 是当时真实发给模型的那一段（消息序列拼成的文本），`chosen` 与
    `rejected` 是**同一个 prompt 下**采样出来的两条候选——不是从两个不同回合
    拼出来的。DPO 要求同 prompt，跨回合配对得到的梯度信号是错的。
    """

    prompt: str
    chosen: str
    rejected: str
    dimension: Dimension
    reason: str
    """`rejected` 被哪条硬判据抓到。留着它是为了让数据集可审计——一条说不出
    「为什么它更差」的偏好对，没法判断它教的是不是你想教的东西。"""
    episode: str
    tick: int
    npc_id: str


class Dataset(BaseModel):
    pairs: list[PreferencePair] = Field(default_factory=list)
    shortfall: dict[str, int] = Field(default_factory=dict)
    """各维度距离目标配额还差多少对。**空字典才是达标。**"""
    narrative_share_missing: float = NARRATIVE_SHARE
    """叙事维度整体缺失的比例。写进产出而不是注释里，因为下游要据此决定
    这份数据能不能直接开训。"""

    def counts(self) -> dict[str, int]:
        return dict(Counter(str(p.dimension) for p in self.pairs))


def assemble(pool: Sequence[PreferencePair], size: int) -> Dataset:
    """按配额从候选池里取 `size` 对，取不够就报缺口。

    每个维度按池中出现顺序取（池由 harvest 确定性产出），所以同一批轨迹
    组装两次必得同一份数据集。
    """
    by_dim: dict[Dimension, list[PreferencePair]] = {d: [] for d in TARGET_QUOTA}
    for pair in pool:
        by_dim.setdefault(pair.dimension, []).append(pair)

    picked: list[PreferencePair] = []
    shortfall: dict[str, int] = {}
    for dim, share in TARGET_QUOTA.items():
        want = round(size * share)
        have = by_dim.get(dim, [])
        picked += have[:want]
        if len(have) < want:
            shortfall[str(dim)] = want - len(have)

    return Dataset(pairs=picked, shortfall=shortfall)


def write_jsonl(dataset: Dataset, path: Path) -> int:
    """按 DPO 训练脚本的常见格式落盘：每行 {prompt, chosen, rejected}。

    维度与理由一起写出去——审计用得上，而训练脚本会忽略多出来的键。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for pair in dataset.pairs:
            fh.write(
                json.dumps(
                    {
                        "prompt": pair.prompt,
                        "chosen": pair.chosen,
                        "rejected": pair.rejected,
                        "dimension": str(pair.dimension),
                        "reason": pair.reason,
                        "source": f"{pair.episode}#{pair.tick}:{pair.npc_id}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(dataset.pairs)
