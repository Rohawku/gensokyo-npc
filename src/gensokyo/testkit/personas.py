from typing import Protocol

from gensokyo.world.observation import PlayerView


class Persona(Protocol):
    """玩家模拟器。

    只能看 `PlayerView` 和 NPC 刚说的那句话——玩家看不到 `WorldState`，
    模拟器也不该看。一旦允许它读引擎内部，它就能走出真实玩家走不出的路，
    基线也就失去了意义。
    """

    name: str

    def next_input(self, view: PlayerView, last_utterance: str) -> str: ...
