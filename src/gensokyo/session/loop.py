from pathlib import Path

from gensokyo.agent.npc import NpcAgent
from gensokyo.agent.schema import NpcTurn
from gensokyo.llm.client import LlmClient
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import ItemId, LocationId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.observation import PlayerView
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action


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

    def go(self, place: str) -> PlayerView:
        target = self._location_id_by_name(place)
        if target is not None:
            self.engine.apply(Action(actor="player", tool="move", args={"to": target}))
            self.engine.tick()
        return self.view()

    def give(self, item: str) -> PlayerView:
        self.engine.apply(Action(actor="player", tool="give_item", args={"item": ItemId(item)}))
        self.engine.tick()
        return self.view()

    def pick(self, item: str) -> PlayerView:
        self.engine.apply(Action(actor="player", tool="take_item", args={"item": ItemId(item)}))
        self.engine.tick()
        return self.view()
