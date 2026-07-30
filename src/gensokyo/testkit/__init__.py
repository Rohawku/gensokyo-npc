"""评测 harness：玩家模拟器、轨迹记录与指标报告。

`world/` 层的规则可以零成本单测，但「一局玩下来是什么体验」不行——
它需要一个会做决定的玩家。这个包提供四种玩家人格（其中三种 0 次 LLM
调用，因此能进 CI）、一份可回放的轨迹格式，以及五个维度的指标。
"""

from gensokyo.testkit.personas import (
    FicklePlayer,
    GameMap,
    HonestPlayer,
    JailbreakPlayer,
    Persona,
    SmoothTalkerPlayer,
)
from gensokyo.testkit.report import EvalReport, evaluate
from gensokyo.testkit.runner import RunConfig, run_episode
from gensokyo.testkit.trajectory import Trajectory, TurnRecord

__all__ = [
    "EvalReport",
    "FicklePlayer",
    "GameMap",
    "HonestPlayer",
    "JailbreakPlayer",
    "Persona",
    "RunConfig",
    "SmoothTalkerPlayer",
    "Trajectory",
    "TurnRecord",
    "evaluate",
    "run_episode",
]
