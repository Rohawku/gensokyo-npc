from gensokyo.agent.policy import run_turn
from gensokyo.agent.schema import NpcTurn
from gensokyo.llm.client import LlmClient
from gensokyo.world.defs import CharacterCard
from gensokyo.world.engine import WorldEngine

HISTORY_WINDOW = 12


class NpcAgent:
    """编排 persona / 观测 / 策略。W1 只有滑动窗口短期记忆，
    W2 会在这里接入分层记忆检索。"""

    def __init__(self, card: CharacterCard, engine: WorldEngine, llm: LlmClient) -> None:
        self.card = card
        self.engine = engine
        self.llm = llm
        self.history: list[str] = []

    def act(self, player_utterance: str) -> NpcTurn:
        self.history.append(f"玩家：{player_utterance}")
        turn = run_turn(self.card, self.engine, self.llm, self.history[-HISTORY_WINDOW:])
        self.history.append(f"{self.card.name}：{turn.utterance}")
        return turn
