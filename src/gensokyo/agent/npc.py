from collections.abc import Callable

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

    def act(self, player_utterance: str, on_chunk: Callable[[str], None] | None = None) -> NpcTurn:
        # 先不写入 history。若 run_turn 抛异常（本地端点超时、限流），
        # 写入过的玩家发言会变成一条没人回应的孤立记录，模型恢复后
        # 看到的对话历史就是错的。两行一起提交，或都不提交。
        said = f"玩家：{player_utterance}"
        window = (self.history + [said])[-HISTORY_WINDOW:]

        turn = run_turn(self.card, self.engine, self.llm, window, on_chunk)

        self.history.append(said)
        self.history.append(f"{self.card.name}：{turn.utterance}")
        return turn
