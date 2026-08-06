from collections.abc import Callable

from gensokyo.agent.policy import run_turn
from gensokyo.agent.schema import NpcTurn, normalize_utterance
from gensokyo.llm.client import LlmClient
from gensokyo.memory.item import MemoryStore
from gensokyo.memory.pipeline import now_seq
from gensokyo.memory.query import MEMORY_TOP_K, build_focus, build_query
from gensokyo.memory.render import render_recall
from gensokyo.memory.retrieve import Scored, retrieve
from gensokyo.world.defs import CharacterCard
from gensokyo.world.engine import WorldEngine

HISTORY_WINDOW = 12


class NpcAgent:
    """编排 persona / 观测 / 记忆 / 策略。

    短期记忆是 12 轮滑动窗口（`history`），长期记忆是 `store`——两者刻意
    分开：窗口装原话，记忆库装从事件日志抽出来的结构化条目，后者能被存档
    精确重建（`memory.pipeline`），前者不能也不必。
    """

    def __init__(
        self,
        card: CharacterCard,
        engine: WorldEngine,
        llm: LlmClient,
        store: MemoryStore | None = None,
    ) -> None:
        self.card = card
        self.engine = engine
        self.llm = llm
        self.store = store if store is not None else MemoryStore(npc_id=card.id)
        self.history: list[str] = []
        self.spoken: list[str] = []
        """她本局说过的台词，按标准化去重、保留原文、按首次出现排序。

        刻意**不**从 history 切片得到：history 只有 12 条的窗口，而实测里
        第 10 回合复读的是第 6 回合那句——早就滑出窗口了。去重也是必须的：
        原先直接取最后三句自己的话，那三句本身就可能是同一句，于是
        「别再重复」这条约束的示例正在示范复读。
        """
        self._spoken_keys: set[str] = set()

    def reset_dialogue(self) -> None:
        """清掉本局的短期状态：原话窗口与禁语清单。读档时用。

        长期记忆不在这里清——它由会话整批换成 `rebuild` 的产物。"""
        self.history.clear()
        self.spoken.clear()
        self._spoken_keys.clear()

    def recall(self, player_utterance: str) -> list[Scored]:
        """本回合召回的条目。**有副作用**（记一次访问），每回合只该调一次。"""
        obs = self.engine.observe(self.card.id)
        return retrieve(
            self.store,
            build_query(obs, player_utterance),
            self.card,
            now_seq(self.engine),
            build_focus(obs),
            k=MEMORY_TOP_K,
        )

    def act(self, player_utterance: str, on_chunk: Callable[[str], None] | None = None) -> NpcTurn:
        return self._turn(
            f"玩家：{player_utterance}", player_utterance, on_chunk, asked=player_utterance
        )

    def react(self, deed: str, on_chunk: Callable[[str], None] | None = None) -> NpcTurn:
        """玩家**做了**一件事而不是说了一句话，她主动开口。

        历史里记成 `（来访者把赛钱交给了你）` 而不是 `玩家：…`——记成后者
        她会以为那句旁白是他说出口的话，然后引用它。

        **只说话，不动世界**（`speech_only`）：指令回合不该额外给她一次行动
        机会。实测不这样做时，玩家送礼触发她开口，她在那次免费的决策阶段里
        顺手把东西拿走，于是送礼净 −2——见 `run_turn` 的说明。
        """
        return self._turn(f"（{deed}）", deed, on_chunk, speech_only=True)

    def _turn(
        self,
        history_line: str,
        query: str,
        on_chunk: Callable[[str], None] | None,
        *,
        asked: str = "",
        speech_only: bool = False,
    ) -> NpcTurn:
        # 先不写入 history。若 run_turn 抛异常（本地端点超时、限流），
        # 写入过的玩家发言会变成一条没人回应的孤立记录，模型恢复后
        # 看到的对话历史就是错的。两行一起提交，或都不提交。
        window = (self.history + [history_line])[-HISTORY_WINDOW:]

        recalled = self.recall(query)
        turn = run_turn(
            self.card,
            self.engine,
            self.llm,
            window,
            self.spoken,
            render_recall(recalled),
            on_chunk,
            asked=asked,
            speech_only=speech_only,
        )
        turn.retrieved_memory_ids = [s.item.id for s in recalled]

        self.history.append(history_line)
        self.history.append(f"{self.card.name}：{turn.utterance}")
        key = normalize_utterance(turn.utterance)
        if key and key not in self._spoken_keys:
            self._spoken_keys.add(key)
            self.spoken.append(turn.utterance)
        return turn
