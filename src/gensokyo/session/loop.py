from collections.abc import Callable
from pathlib import Path

from gensokyo.agent.npc import NpcAgent
from gensokyo.agent.schema import NpcTurn
from gensokyo.llm.client import LlmClient
from gensokyo.memory.item import MemoryStore
from gensokyo.memory.pipeline import absorb, new_stores, rebuild
from gensokyo.session.save import load_actions, save_actions
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import LocationId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.observation import PlayerView
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action, ActionResult, ErrorCode


class Session:
    def __init__(
        self,
        engine: WorldEngine,
        agents: dict[NpcId, NpcAgent],
        stores: dict[NpcId, MemoryStore],
    ) -> None:
        self.engine = engine
        self.agents = agents
        self.stores = stores
        """记忆库归会话所有，agent 只持引用。读档要整批换掉（`rebuild`
        产出的是新对象），归属放在这里换起来只有一处。"""
        self.refusals: list[str] = []
        """上一次 `say` 里拒绝搭话的 NPC 各自的那一行。调用方要显示它——
        没有它，「她不理你」和「程序卡住了」在屏幕上分不开。"""

    @classmethod
    def create(cls, scenario_dir: Path, characters_dir: Path, llm: LlmClient) -> "Session":
        defs = load_defs(scenario_dir, characters_dir)
        engine = WorldEngine(build_initial_state(defs), defs)
        stores = new_stores(defs)
        agents = {
            npc_id: NpcAgent(card=card, engine=engine, llm=llm, store=stores[npc_id])
            for npc_id, card in defs.characters.items()
        }
        return cls(engine, agents, stores)

    def view(self) -> PlayerView:
        return self.engine.observe_player()

    def is_over(self) -> bool:
        return self.engine.state.quest.ending is not None

    def save(self, path: Path) -> int:
        """存档只写动作日志——世界状态是它的推导结果，存两份就会不一致。"""
        save_actions(path, self.engine.state.action_log)
        return len(self.engine.state.action_log)

    def load(self, path: Path) -> int:
        """从动作日志重建世界**和记忆**。

        长期记忆是动作日志的推导产物，走 `memory.rebuild`——和实时对局同一条
        码路，所以重建出的记忆库与存档那一刻逐字段相同（有测试锁住）。

        短期记忆（12 轮原话窗口）不在存档里也不重建：它是 agent 层的东西。
        所以读档后她记得**发生过什么**，但不记得原话——这正好是人的样子。
        """
        actions = load_actions(path)
        self.engine, self.stores = rebuild(actions, self.engine.defs)
        for npc_id, agent in self.agents.items():
            agent.engine = self.engine
            agent.store = self.stores[npc_id]
            agent.reset_dialogue()
        return len(actions)

    def _location_id_by_name(self, name: str) -> LocationId | None:
        for loc_id, loc in self.engine.defs.locations.items():
            if loc.name == name or loc_id == name:
                return loc_id
        return None

    def say(self, text: str, on_chunk: Callable[[str], None] | None = None) -> list[NpcTurn]:
        """玩家说话。**发言先入库，再看谁愿意回应。**

        情绪模式声明了 refusal 的 NPC 直接跳过——她的发言不会进 event_log，
        因为她确实没说话。玩家仍然看到一行说明（`panel.refusal`），否则
        「她不理你」和「程序卡住了」在屏幕上分不开。
        """
        self.engine.apply(Action(actor="player", tool="say", args={"text": text}))
        self.refusals = []
        turns: list[NpcTurn] = []
        for panel in self.engine.observe_player().npcs_here:
            if panel.refusal:
                self.refusals.append(panel.refusal)
                continue
            if turns and on_chunk is not None:
                # 多个 NPC 同场时台词会流到同一行上。只有这里知道说话人
                # 换了，所以分隔符必须在这里发出去。
                on_chunk("\n")
            turns.append(self.agents[panel.npc_id].act(text, on_chunk))
        self.engine.tick()
        absorb(self.engine, self.stores)
        return turns

    def _act(self, action: Action) -> ActionResult:
        """施加玩家动作。失败不推进回合——打错一个字不该让 NPC 的情绪衰减一轮。"""
        result = self.engine.apply(action)
        if result.ok:
            self.engine.tick()
            absorb(self.engine, self.stores)
        return result

    def go(self, place: str) -> ActionResult:
        target = self._location_id_by_name(place)
        if target is None:
            return ActionResult.failed(ErrorCode.NO_SUCH_EXIT, f"幻想乡没有叫「{place}」的地方。")
        return self._act(Action(actor="player", tool="move", args={"to": target}))

    def give(self, item: str) -> ActionResult:
        target = self.engine.resolve_item(item)
        if target is None:
            return ActionResult.failed(ErrorCode.BAD_ARGS, f"没有叫「{item}」的东西。")
        return self._act(Action(actor="player", tool="give_item", args={"item": target}))

    def pick(self, item: str) -> ActionResult:
        target = self.engine.resolve_item(item)
        if target is None:
            return ActionResult.failed(ErrorCode.BAD_ARGS, f"没有叫「{item}」的东西。")
        return self._act(Action(actor="player", tool="take_item", args={"item": target}))
