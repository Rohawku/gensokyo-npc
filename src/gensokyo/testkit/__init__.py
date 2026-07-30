"""评测 harness：玩家模拟器与轨迹记录。

`world/` 层的规则可以零成本单测，但「一局玩下来是什么体验」不行——
它需要一个会做决定的玩家。这个包提供四种玩家人格（其中三种 0 次 LLM
调用，因此能进 CI）和一份可回放的轨迹格式。
"""

from gensokyo.testkit.personas import (
    FicklePlayer,
    GameMap,
    HonestPlayer,
    JailbreakPlayer,
    Persona,
    SmoothTalkerPlayer,
)
from gensokyo.testkit.runner import RunConfig, run_episode
from gensokyo.testkit.trajectory import Trajectory, TurnRecord

__all__ = [
    "FicklePlayer",
    "GameMap",
    "HonestPlayer",
    "JailbreakPlayer",
    "Persona",
    "RunConfig",
    "SmoothTalkerPlayer",
    "Trajectory",
    "TurnRecord",
    "run_episode",
]
