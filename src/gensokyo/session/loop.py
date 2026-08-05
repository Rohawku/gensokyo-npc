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

VOLUNTEER_DEEDS: dict[str, str] = {
    "move": "来访者刚走进来",
    "give_item": "来访者把{item}交给了你",
    "take_item": "来访者拿走了这里的{item}",
}
"""哪些玩家动作值得她主动开口，以及写给她看的那句旁白。

**`say` 不在里面**——玩家说话走 `say()`，那本来就会得到回应，放进来等于同一句
话回两次。清单是白名单而不是黑名单：以后新加的工具默认不触发主动开口，
要开口得显式写进来（坑 #3 的规矩——新工具默认发给所有人是个陷阱）。
"""


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
        self.volunteered: tuple[NpcId, NpcTurn] | None = None
        """上一个指令回合里主动开口的那个 NPC 和她的发言。调用方要显示它。

        **一个回合最多一个人**：两个人同时主动搭话，屏幕上是两段独白，玩家
        插不上话；轨迹里也会出现「一个指令回合两条 NPC 记录」这种要特殊处理的
        形状。限成一个既是体验也是记录格式的选择。
        """
        self._volunteered: dict[NpcId, set[str]] = {}
        """谁已经对哪种动作主动开口过。见 `VOLUNTEER_DEEDS`。"""

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
        self.volunteered = None
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
        self.volunteered = None
        result = self.engine.apply(action)
        if result.ok:
            self._volunteer(action)
            self.engine.tick()
            absorb(self.engine, self.stores)
        return result

    def _volunteer(self, action: Action) -> None:
        """指令回合里让在场 NPC 主动开口。

        **为什么要有这一步。** 实测一局 28 个回合里 18 个是指令，而指令回合
        全程无人说话——走进神社、投币、从她店里拿走一本书，灵梦和魔理沙一句
        反应都没有。那 18 个回合与 LLM 完全无关，而这是个 LLM 驱动的对话游戏。

        **每个 NPC 对每种动作只开口一次**（`_volunteered`）。理由不是省钱，是
        复读：第二次投币她已经表过态了，再说一遍就是同一句话的变体，而复读率
        是硬指标里守得最紧的一项。限流也顺带把成本压住——一局约 5~6 次。
        """
        if action.tool not in VOLUNTEER_DEEDS:
            return
        deed = self._deed_text(action)
        for panel in self.engine.observe_player().npcs_here:
            npc_id = panel.npc_id
            if panel.refusal or action.tool in self._volunteered.setdefault(npc_id, set()):
                continue
            self._volunteered[npc_id].add(action.tool)
            self.volunteered = (npc_id, self.agents[npc_id].react(deed))
            return

    def _deed_text(self, action: Action) -> str:
        """把动作写成她看得懂的一句旁白。

        物品名要出现在里面：「来访者给了你东西」和「来访者给了你一枚赛钱」
        对生成的台词是两回事，而后者已经在状态里了，没有理由不告诉她。
        """
        template = VOLUNTEER_DEEDS[action.tool]
        item = action.args.get("item")
        name = self.engine.defs.items[item].name if item in self.engine.defs.items else "东西"
        return template.format(item=name)

    def go(self, place: str) -> ActionResult:
        self.volunteered = None
        target = self._location_id_by_name(place)
        if target is None:
            return ActionResult.failed(ErrorCode.NO_SUCH_EXIT, f"幻想乡没有叫「{place}」的地方。")
        return self._act(Action(actor="player", tool="move", args={"to": target}))

    def give(self, item: str, to: str = "") -> ActionResult:
        """`to` 为空时由引擎决定：只有一个人在场就给她，两个人就报错让玩家点名。"""
        self.volunteered = None
        target = self.engine.resolve_item(item)
        if target is None:
            return ActionResult.failed(ErrorCode.BAD_ARGS, f"没有叫「{item}」的东西。")
        args: dict[str, object] = {"item": target}
        if to:
            who = self.engine.resolve_npc(to)
            if who is None:
                return ActionResult.failed(ErrorCode.BAD_ARGS, f"这里没有叫「{to}」的人。")
            args["to"] = who
        return self._act(Action(actor="player", tool="give_item", args=args))

    def pick(self, item: str) -> ActionResult:
        self.volunteered = None
        target = self.engine.resolve_item(item)
        if target is None:
            return ActionResult.failed(ErrorCode.BAD_ARGS, f"没有叫「{item}」的东西。")
        return self._act(Action(actor="player", tool="take_item", args={"item": target}))
