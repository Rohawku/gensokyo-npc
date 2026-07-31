"""salience：规则基线 × 角色情绪敏感度系数。

不让 LLM 自由打分。每个数字都能从人设推导，可复现、可解释，不是调出来的
超参——而且一个 8B 模型给同一件事打的分本来就不稳定，用它当检索权重等于
往排序里注入噪声。

基线表在 `world/defs.SALIENCE_BASELINE`，因为它同时被角色卡的加载校验用到，
而 `world/` 不许 import 项目内其他模块。一处定义两处使用。
"""

from gensokyo.world.defs import SALIENCE_BASELINE, CharacterCard


def salience_for(card: CharacterCard, event_key: str) -> float:
    """基线 × 角色系数，截到 [0, 1]。

    未登记的事件类型返回 0.0 而不是某个默认基线：没人给过基线说明它还没被
    设计成「值得记住的事」，凭空给分会让它在检索里和真正重要的条目竞争。
    调用方按 0.0 决定不写入。
    """
    base = SALIENCE_BASELINE.get(event_key, 0.0)
    if base == 0.0:
        return 0.0
    factor = card.memory.salience_multipliers.get(event_key, 1.0)
    return max(0.0, min(1.0, base * factor))
