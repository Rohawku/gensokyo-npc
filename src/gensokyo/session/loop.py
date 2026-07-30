from pathlib import Path

from gensokyo.agent.npc import NpcAgent
from gensokyo.agent.schema import NpcTurn
from gensokyo.llm.client import LlmClient
from gensokyo.session.save import load_actions, save_actions
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import LocationId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.observation import PlayerView
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action, ActionResult, ErrorCode


class Session:
    def __init__(self, engine: WorldEngine, agents: dict[NpcId, NpcAgent]) -> None:
        self.engine = engine
        self.agents = agents

    @classmethod
    def create(cls, scenario_dir: Path, characters_dir: Path, llm: LlmClient) -> "Session":
        defs = load_defs(scenario_dir, characters_dir)
        engine = WorldEngine(build_initial_state(defs), defs)
        agents = {
            npc_id: NpcAgent(card=card, engine=engine, llm=llm)
            for npc_id, card in defs.characters.items()
        }
        return cls(engine, agents)

    def view(self) -> PlayerView:
        return self.engine.observe_player()

    def is_over(self) -> bool:
        return self.engine.state.quest.ending is not None

    def save(self, path: Path) -> int:
        """存档只写动作日志——世界状态是它的推导结果，存两份就会不一致。"""
        save_actions(path, self.engine.state.action_log)
        return len(self.engine.state.action_log)

    def load(self, path: Path) -> int:
        """从动作日志重建世界。NPC 的对话历史不在存档里（那是 agent 层的
        短期记忆），所以读档后 NPC 记得世界发生过什么，但不记得原话。"""
        actions = load_actions(path)
        self.engine = WorldEngine.replay(actions, self.engine.defs)
        for agent in self.agents.values():
            agent.engine = self.engine
            agent.history.clear()
        return len(actions)

    def _location_id_by_name(self, name: str) -> LocationId | None:
        for loc_id, loc in self.engine.defs.locations.items():
            if loc.name == name or loc_id == name:
                return loc_id
        return None

    def say(self, text: str) -> list[NpcTurn]:
        self.engine.apply(Action(actor="player", tool="say", args={"text": text}))
        turns: list[NpcTurn] = []
        for panel in self.engine.observe_player().npcs_here:
            turns.append(self.agents[panel.npc_id].act(text))
        self.engine.tick()
        return turns

    def _act(self, action: Action) -> ActionResult:
        """施加玩家动作。失败不推进回合——打错一个字不该让 NPC 的情绪衰减一轮。"""
        result = self.engine.apply(action)
        if result.ok:
            self.engine.tick()
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
