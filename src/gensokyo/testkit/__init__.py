"""评测 harness：玩家模拟器与轨迹记录。

`world/` 层的规则可以零成本单测，但「一局玩下来是什么体验」不行——
它需要一个会做决定的玩家。这个包提供可回放的轨迹格式与批量跑局的 runner。
"""

from gensokyo.testkit.personas import Persona
from gensokyo.testkit.runner import RunConfig, run_episode
from gensokyo.testkit.trajectory import Trajectory, TurnRecord

__all__ = [
    "Persona",
    "RunConfig",
    "Trajectory",
    "TurnRecord",
    "run_episode",
]
