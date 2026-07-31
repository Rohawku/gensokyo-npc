from collections.abc import Callable

from gensokyo.agent.policy import run_turn
from gensokyo.agent.schema import NpcTurn, normalize_utterance
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
        self.spoken: list[str] = []
        """她本局说过的台词，按标准化去重、保留原文、按首次出现排序。

        刻意**不**从 history 切片得到：history 只有 12 条的窗口，而实测里
        第 10 回合复读的是第 6 回合那句——早就滑出窗口了。去重也是必须的：
        原先直接取最后三句自己的话，那三句本身就可能是同一句，于是
        「别再重复」这条约束的示例正在示范复读。
        """
        self._spoken_keys: set[str] = set()

    def act(self, player_utterance: str, on_chunk: Callable[[str], None] | None = None) -> NpcTurn:
        # 先不写入 history。若 run_turn 抛异常（本地端点超时、限流），
        # 写入过的玩家发言会变成一条没人回应的孤立记录，模型恢复后
        # 看到的对话历史就是错的。两行一起提交，或都不提交。
        said = f"玩家：{player_utterance}"
        window = (self.history + [said])[-HISTORY_WINDOW:]

        turn = run_turn(self.card, self.engine, self.llm, window, self.spoken, on_chunk)

        self.history.append(said)
        self.history.append(f"{self.card.name}：{turn.utterance}")
        key = normalize_utterance(turn.utterance)
        if key and key not in self._spoken_keys:
            self._spoken_keys.add(key)
            self.spoken.append(turn.utterance)
        return turn
