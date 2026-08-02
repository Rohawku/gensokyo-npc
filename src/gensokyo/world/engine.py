import math
from collections import deque
from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from gensokyo.world.defs import EmotionMode, StageDef, WorldDefs
from gensokyo.world.events import Event, EventKind
from gensokyo.world.ids import EventId, ItemId, LocationId, NpcId
from gensokyo.world.observation import FactContext, NpcPanel, Observation, PlayerView
from gensokyo.world.quest import (
    ACTION_LIMIT,
    ANOMALY_SITE,
    OBLIVION_THRESHOLD,
    OBLIVION_WARNING,
    TIMEOUT_ENDING,
    compute_stage,
)
from gensokyo.world.rules import (
    ATTITUDE_DELTA,
    MODE_HYSTERESIS,
    apply_emotion_decay,
    bump_attitude,
    bump_emotion,
    can_reveal,
    emotion_delta_for,
)
from gensokyo.world.state import QuestStage, WorldState
from gensokyo.world.tools import (
    TOOL_REGISTRY,
    Action,
    ActionResult,
    AskPlayerArgs,
    BreakItemArgs,
    ErrorCode,
    GiveItemArgs,
    MoveArgs,
    RevealInfoArgs,
    SayArgs,
    TakeItemArgs,
    ToolSpec,
    TravelToArgs,
    UseSpellcardArgs,
    parse_args,
)

ToolHandler = Callable[[Action, BaseModel], ActionResult]


def _typed[A: BaseModel](model: type[A], fn: Callable[[Action, A], ActionResult]) -> ToolHandler:
    """把「参数类型明确」的处理函数包成统一签名的 `ToolHandler`。

    分发表的值必须是同一个签名，而每个处理函数只认自己那个参数模型。原先
    的做法是让所有处理函数都收 `BaseModel`，再在函数体第一行写
    `assert isinstance(args, SayArgs)` 收窄类型——八个工具就是八行样板，
    而且那是**运行时**断言，注册表里配错了要跑到才知道。

    这里让 `A` 同时从 `model` 和 `fn` 的第二个参数推导，于是
    `_typed(SayArgs, self._do_move)` 在 mypy 下直接报错。**检查从运行时
    搬到了类型检查期**，顺带少了八行。
    """

    def wrapper(action: Action, args: BaseModel) -> ActionResult:
        # cast 而不是 isinstance：调用方唯一的入口是 parse_args，它按
        # TOOL_REGISTRY 里登记的 args_model 构造，所以类型已经由注册表保证。
        return fn(action, cast(A, args))

    return wrapper


WARN_WITHIN_TURNS = 3
"""提前几个回合开始预警。

3 是「够你反应过来」和「不至于一进门就在报警」之间的取舍：灵梦从初始
0.10 涨到门槛要 ~28 个回合，预警只在最后三个回合出现。
"""


class WorldEngine:
    """确定性世界内核。apply() 是唯一的状态变更入口，
    因此任何变更都必然产生 Event，event_log 必然完整。"""

    def __init__(self, state: WorldState, defs: WorldDefs) -> None:
        self.state = state
        self.defs = defs

    # ---------- 事件 ----------

    def _next_event_id(self) -> EventId:
        self.state.seq += 1
        return EventId(f"e{self.state.seq:05d}")

    def _actor_location(self, actor: str) -> LocationId:
        if actor == "player" or actor == "world":
            return self.state.player.location
        return self.state.npcs[NpcId(actor)].location

    def _emit(
        self,
        kind: EventKind,
        actor: str,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        ev = Event(
            id=self._next_event_id(),
            tick=self.state.tick,
            kind=kind,
            actor=actor,
            location=self._actor_location(actor),
            payload=payload or {},
        )
        self.state.event_log.append(ev)
        return ev

    # ---------- 回合推进 ----------

    def tick(self) -> None:
        self.state.tick += 1
        for npc_id, npc in self.state.npcs.items():
            apply_emotion_decay(npc, self.defs.characters[npc_id])

    # ---------- 唯一变更入口 ----------

    def apply(self, action: Action) -> ActionResult:
        if action.tool not in TOOL_REGISTRY:
            return ActionResult.failed(ErrorCode.UNKNOWN_TOOL, f"没有名为 {action.tool} 的动作。")
        try:
            args = parse_args(action.tool, action.args)
        except ValidationError as exc:
            detail = exc.errors()[0]
            return ActionResult.failed(
                ErrorCode.BAD_ARGS,
                f"参数不合法：{detail['loc']} {detail['msg']}",
            )

        denied = self._check_denied(action)
        if denied is not None:
            return denied

        handler = self._handlers().get(action.tool)
        if handler is None:
            return ActionResult.failed(ErrorCode.UNKNOWN_TOOL, f"动作 {action.tool} 尚未实现。")
        result = handler(action, args)
        if result.ok:
            self.state.action_log.append(action)
            self._advance_oblivion(action)
            self.refresh_quest()
            if self.state.quest.ending is None and len(self.state.action_log) >= ACTION_LIMIT:
                self._finish(TIMEOUT_ENDING)
        return result

    @classmethod
    def replay(cls, actions: list[Action], defs: WorldDefs) -> "WorldEngine":
        """从动作日志重建引擎。存档读档、调试重现、离线分析共用这一个入口。"""
        from gensokyo.world.state import build_initial_state

        engine = cls(build_initial_state(defs), defs)
        for action in actions:
            engine.apply(action)
        return engine

    def _advance_oblivion(self, action: Action) -> None:
        """玩家在无缘塚每行动一次就更接近遗忘；离开花田即清零。

        计数挂在动作上而不是 tick 上，这样它进动作日志、能被 replay 精确重现。
        挂在 tick 上会让存档读档时丢掉的线索凭空回来。
        """
        if action.actor != "player":
            return

        player = self.state.player
        if player.location != ANOMALY_SITE:
            player.oblivion_exposure = 0
            return

        player.oblivion_exposure += 1
        if player.oblivion_exposure < OBLIVION_THRESHOLD:
            return

        player.oblivion_exposure = 0
        if not player.known_facts:
            return

        # 取排序后的第一条，保证回放确定性——不能用随机或集合迭代序。
        lost = sorted(player.known_facts)[0]
        player.known_facts.discard(lost)
        self._emit(
            EventKind.MEMORY_LOST,
            "world",
            {"fact": lost, "content": self.defs.facts[lost].content},
        )

    def _finish(self, ending_id: str) -> None:
        self.state.quest.stage = QuestStage.S4_END
        self.state.quest.ending = ending_id
        self._emit(
            EventKind.QUEST_ADVANCE,
            "world",
            {
                "stage": int(QuestStage.S4_END),
                "stage_name": QuestStage.S4_END.name,
                "ending": ending_id,
            },
        )

    def refresh_quest(self) -> None:
        clues = self.defs.clue_facts()
        self.state.quest.clues_obtained = self.state.player.known_facts & clues

        target = compute_stage(self.state, self.defs)
        if target > self.state.quest.stage:
            self.state.quest.stage = target
            self._emit(
                EventKind.QUEST_ADVANCE,
                "world",
                {"stage": int(target), "stage_name": target.name},
            )

    def available_tools(self, npc_id: NpcId) -> list[ToolSpec]:
        """情绪状态机的落点：模式变化会真的改变 NPC 能做什么，
        而不是在 prompt 里描述一句。

        基线只含非受限工具；受限工具（如 break_item）必须由当前模式的
        tools_allow 显式解锁。永久禁用在最后再减一次，优先于情绪解锁——
        否则写错一行 YAML 就能让芙兰逃出地下室。
        """
        card = self.defs.characters[npc_id]

        allowed = {name for name, spec in TOOL_REGISTRY.items() if not spec.restricted}
        allowed -= set(card.tools.deny_always)
        mode = self._current_mode(npc_id)
        allowed -= set(mode.tools_deny)
        allowed |= set(mode.tools_allow)
        allowed -= set(card.tools.deny_always)

        return [TOOL_REGISTRY[name] for name in sorted(allowed)]

    def _check_denied(self, action: Action) -> ActionResult | None:
        if action.actor == "player":
            return None
        npc_id = NpcId(action.actor)
        card = self.defs.characters[npc_id]
        allowed = {spec.name for spec in self.available_tools(npc_id)}
        if action.tool in allowed:
            return None
        reason = card.tools.deny_reasons.get(
            action.tool, f"你现在做不到「{TOOL_REGISTRY[action.tool].description}」"
        )
        return ActionResult.failed(ErrorCode.TOOL_DENIED, reason)

    def _handlers(self) -> dict[str, ToolHandler]:
        return {
            "say": _typed(SayArgs, self._do_say),
            "move": _typed(MoveArgs, self._do_move),
            "travel_to": _typed(TravelToArgs, self._do_travel_to),
            "give_item": _typed(GiveItemArgs, self._do_give_item),
            "take_item": _typed(TakeItemArgs, self._do_take_item),
            "reveal_info": _typed(RevealInfoArgs, self._do_reveal_info),
            "ask_player": _typed(AskPlayerArgs, self._do_ask_player),
            "use_spellcard": _typed(UseSpellcardArgs, self._do_use_spellcard),
            "break_item": _typed(BreakItemArgs, self._do_break_item),
        }

    # ---------- 各工具实现 ----------

    def _do_say(self, action: Action, args: SayArgs) -> ActionResult:
        kind = EventKind.PLAYER_UTTERANCE if action.actor == "player" else EventKind.NPC_UTTERANCE
        if action.actor == "player":
            # 说话本身要动情绪。在此之前情绪只被 give_item 推动，而灵梦收到
            # 赛钱是**消气**——于是她的烦躁度只有向下的路，`irritated` 在真实
            # 玩法里根本到不了：实测连问 40 轮后 emotion 从 0.10 掉到 0.00。
            # 单测直接调 bump_emotion，所以这件事一直没红。
            for npc_id in self._npcs_here():
                card = self.defs.characters[npc_id]
                bump_emotion(
                    self.state.npcs[npc_id], card, emotion_delta_for(card, "player_talked")
                )
        ev = self._emit(kind, action.actor, {"text": args.text})
        return ActionResult.succeeded([ev])

    def _do_move(self, action: Action, args: MoveArgs) -> ActionResult:
        here = self._actor_location(action.actor)
        if args.to not in self.defs.locations[here].exits:
            return ActionResult.failed(
                ErrorCode.NO_SUCH_EXIT, f"从{self.defs.locations[here].name}没法直接过去。"
            )

        if action.actor == "player":
            self.state.player.location = args.to
            kind = EventKind.PLAYER_ACTION
        else:
            self.state.npcs[NpcId(action.actor)].location = args.to
            kind = EventKind.NPC_ACTION

        ev = self._emit(kind, action.actor, {"tool": "move", "to": args.to})
        return ActionResult.succeeded([ev], f"来到了{self.defs.locations[args.to].name}。")

    def _path(self, start: LocationId, goal: LocationId) -> list[LocationId] | None:
        """出口图上的最短路。地图是引擎的知识，不该让一个 8B 模型
        看着出口清单自己规划多跳路径——实测它只会试一步，失败就放弃。"""
        if start == goal:
            return []
        seen = {start}
        queue: deque[tuple[LocationId, list[LocationId]]] = deque([(start, [])])
        while queue:
            here, path = queue.popleft()
            for nxt in self.defs.locations[here].exits:
                if nxt in seen:
                    continue
                if nxt == goal:
                    return [*path, nxt]
                seen.add(nxt)
                queue.append((nxt, [*path, nxt]))
        return None

    def _do_travel_to(self, action: Action, args: TravelToArgs) -> ActionResult:
        if action.actor == "player":
            return ActionResult.failed(
                ErrorCode.TOOL_DENIED, "这个动作只有 NPC 能用，你得自己一步步走。"
            )
        if args.destination not in self.defs.locations:
            return ActionResult.failed(
                ErrorCode.NO_SUCH_EXIT, f"幻想乡没有叫「{args.destination}」的地方。"
            )

        npc_id = NpcId(action.actor)
        npc = self.state.npcs[npc_id]
        path = self._path(npc.location, args.destination)
        if path is None:
            return ActionResult.failed(ErrorCode.UNREACHABLE, "从这儿过不去。")
        if not path:
            return ActionResult.succeeded([], "你已经在那儿了。")

        events = []
        for hop in path:
            npc.location = hop
            events.append(
                self._emit(EventKind.NPC_ACTION, action.actor, {"tool": "move", "to": hop})
            )
        return ActionResult.succeeded(events, f"去了{self.defs.locations[args.destination].name}。")

    def _npcs_here(self) -> list[NpcId]:
        here = self.state.player.location
        return [nid for nid, npc in self.state.npcs.items() if npc.location == here]

    @staticmethod
    def _pop_items(bag: dict[ItemId, int], item: ItemId, count: int) -> bool:
        if bag.get(item, 0) < count:
            return False
        bag[item] -= count
        if bag[item] == 0:
            del bag[item]
        return True

    @staticmethod
    def _push_items(bag: dict[ItemId, int], item: ItemId, count: int) -> None:
        bag[item] = bag.get(item, 0) + count

    def _do_give_item(self, action: Action, args: GiveItemArgs) -> ActionResult:
        if action.actor == "player":
            present = self._npcs_here()
            if not present:
                return ActionResult.failed(ErrorCode.NOT_CO_LOCATED, "这里没有人可以给。")
            target = present[0]
            npc = self.state.npcs[target]
            if not self._pop_items(self.state.player.inventory, args.item, args.count):
                return ActionResult.failed(
                    ErrorCode.INSUFFICIENT_ITEM,
                    f"你身上没有那么多{self._item_name(args.item)}。",
                )
            self._push_items(npc.inventory, args.item, args.count)
            npc.received_items.add(args.item)
            bump_attitude(npc, ATTITUDE_DELTA["player_gave_item"])
            bump_emotion(
                npc,
                self.defs.characters[target],
                emotion_delta_for(self.defs.characters[target], "player_gave_item"),
            )
            ev = self._emit(
                EventKind.PLAYER_ACTION,
                "player",
                {"tool": "give_item", "item": args.item, "count": args.count, "to": target},
            )
            return ActionResult.succeeded(
                [ev],
                f"把{args.count}个{self._item_name(args.item)}"
                f"交给了{self.defs.characters[target].name}。",
            )

        npc_id = NpcId(action.actor)
        npc = self.state.npcs[npc_id]
        if not self._pop_items(npc.inventory, args.item, args.count):
            return ActionResult.failed(
                ErrorCode.INSUFFICIENT_ITEM, f"你手上没有那么多{self._item_name(args.item)}。"
            )
        self._push_items(self.state.player.inventory, args.item, args.count)
        ev = self._emit(
            EventKind.NPC_ACTION,
            action.actor,
            {"tool": "give_item", "item": args.item, "count": args.count, "to": "player"},
        )
        return ActionResult.succeeded([ev], f"给出了{self._item_name(args.item)}。")

    def _do_take_item(self, action: Action, args: TakeItemArgs) -> ActionResult:
        if action.actor == "player":
            here = self.state.locations[self.state.player.location]
            if not self._pop_items(here.items, args.item, args.count):
                return ActionResult.failed(
                    ErrorCode.INSUFFICIENT_ITEM, f"这里没有{self._item_name(args.item)}。"
                )
            self._push_items(self.state.player.inventory, args.item, args.count)
            ev = self._emit(
                EventKind.PLAYER_ACTION,
                "player",
                {"tool": "take_item", "item": args.item, "count": args.count},
            )
            return ActionResult.succeeded([ev], f"捡起了{self._item_name(args.item)}。")
        npc_id = NpcId(action.actor)
        npc = self.state.npcs[npc_id]
        if npc.location != self.state.player.location:
            return ActionResult.failed(ErrorCode.NOT_CO_LOCATED, "对方不在这里。")
        if not self._pop_items(self.state.player.inventory, args.item, args.count):
            return ActionResult.failed(
                ErrorCode.INSUFFICIENT_ITEM, f"对方身上没有那么多{self._item_name(args.item)}。"
            )
        self._push_items(npc.inventory, args.item, args.count)
        # 抢来的也算收到过。不记的话交易门槛会永久打不开而东西已经没了——
        # 魔理沙的 take_item 行为基线是全场最高的 0.30，她自己抢走珍稀魔法书
        # 就会把第二条线索锁死。
        npc.received_items.add(args.item)
        bump_attitude(npc, ATTITUDE_DELTA["npc_took_item"])
        ev = self._emit(
            EventKind.NPC_ACTION,
            action.actor,
            {"tool": "take_item", "item": args.item, "count": args.count},
        )
        return ActionResult.succeeded([ev], f"拿走了{self._item_name(args.item)}。")

    def _current_mode(self, npc_id: NpcId) -> EmotionMode:
        """她当前所在的那档情绪模式。

        「按名字在 modes 里找出那个对象」这段循环曾经散在四个地方（工具集
        过滤、语气提示、观测组装、拒绝搭话），而每加一个模式相关的字段就
        多抄一遍。`observe` 里那一份甚至和 `_mode_hint` 逐行相同——改一处
        忘一处只是时间问题。

        名字对不上时退回第一档：`resolve_mode` 保证模式名总是从这张表里来，
        所以对不上意味着有人绕过它直接赋值，那是 bug 而不是要在玩家面板上
        抛异常的场合。
        """
        name = self.state.npcs[npc_id].mode
        for mode in self.defs.characters[npc_id].emotion.modes:
            if mode.name == name:
                return mode
        return self.defs.characters[npc_id].emotion.modes[0]

    def _mode_hint(self, npc_id: NpcId) -> str:
        return self._current_mode(npc_id).speech_hint

    def resolve_item(self, text: str) -> ItemId | None:
        """把玩家输入的中文物品名或英文 id 解析成 ItemId。
        面板显示中文，输入却只认英文 id 会把玩家训练成敲英文。"""
        for item_id, item in self.defs.items.items():
            if text == item_id or text in item.surfaces():
                return item_id
        return None

    def _named(self, bag: dict[ItemId, int]) -> dict[str, int]:
        return {self._item_name(item): n for item, n in bag.items()}

    def _item_name(self, item: ItemId) -> str:
        known = self.defs.items.get(item)
        return known.name if known else str(item)

    def _do_reveal_info(self, action: Action, args: RevealInfoArgs) -> ActionResult:
        if action.actor == "player":
            return ActionResult.failed(ErrorCode.TOOL_DENIED, "这个动作只有 NPC 能用。")

        npc_id = NpcId(action.actor)
        npc = self.state.npcs[npc_id]
        fact = self.defs.facts.get(args.fact)
        if fact is None or args.fact not in npc.holds_facts:
            return ActionResult.failed(ErrorCode.NOT_FACT_HOLDER, "你并不知道这件事，说不出来。")
        if not can_reveal(npc, fact.reveal_conditions):
            return ActionResult.failed(
                ErrorCode.REVEAL_CONDITION_UNMET,
                "现在还不想告诉对方这件事——对方还没让你觉得值得说。",
            )

        if args.fact in npc.revealed_facts and args.fact in self.state.player.known_facts:
            # 已经说过、而且对方还记得。不算失败——把它当失败会给策略层一个
            # 语义错乱的「你说不出来」。但不再产生事件，否则重复揭示会用同质
            # 条目灌满 event_log，挤占后续 prompt 的近期事件窗口。
            #
            # 「对方还记得」这个条件是必需的：玩家被无缘塚的花吸走记忆后，
            # 只看 revealed_facts 会让她拒绝重讲，线索永久拿不回来。
            return ActionResult.succeeded([], f"这件事你已经告诉过对方了：{fact.content}")

        self.state.player.known_facts.add(args.fact)
        npc.revealed_facts.add(args.fact)
        ev = self._emit(
            EventKind.NPC_ACTION,
            action.actor,
            {"tool": "reveal_info", "fact": args.fact},
        )
        return ActionResult.succeeded([ev], fact.content)

    def _do_ask_player(self, action: Action, args: AskPlayerArgs) -> ActionResult:
        if action.actor == "player":
            return ActionResult.failed(ErrorCode.TOOL_DENIED, "这个动作只有 NPC 能用。")

        ev = self._emit(
            EventKind.NPC_ACTION,
            action.actor,
            {"tool": "ask_player", "question": args.question},
        )
        return ActionResult.succeeded([ev])

    def _do_use_spellcard(self, action: Action, args: UseSpellcardArgs) -> ActionResult:
        if action.actor == "player":
            return ActionResult.failed(ErrorCode.TOOL_DENIED, "这个动作只有 NPC 能用。")

        ev = self._emit(
            EventKind.NPC_ACTION,
            action.actor,
            {"tool": "use_spellcard", "name": args.name},
        )

        # 剧情的最后一步由 NPC 自己走完：玩家说服她来无缘塚，她动手解决。
        npc_id = NpcId(action.actor)
        card_name = self.defs.characters[npc_id].name
        ending = self.defs.ending_by(npc_id)
        if (
            ending is not None
            and self.state.npcs[npc_id].location == ANOMALY_SITE
            and self.state.quest.stage >= QuestStage.S3_SOURCE
        ):
            self._finish(ending.id)
            # 结局正文交给结局块打印，这里只报机械结果，否则玩家会看两遍。
            return ActionResult.succeeded([ev], f"{card_name}动手了。")

        return ActionResult.succeeded([ev], f"发动了符卡「{args.name}」。")

    def _do_break_item(self, action: Action, args: BreakItemArgs) -> ActionResult:
        if action.actor == "player":
            return ActionResult.failed(ErrorCode.TOOL_DENIED, "这个动作只有 NPC 能用。")

        npc_id = NpcId(action.actor)
        npc = self.state.npcs[npc_id]
        if not self._pop_items(npc.inventory, args.item, 1):
            return ActionResult.failed(
                ErrorCode.INSUFFICIENT_ITEM, f"你手上没有{self._item_name(args.item)}。"
            )
        ev = self._emit(
            EventKind.NPC_ACTION,
            action.actor,
            {"tool": "break_item", "item": args.item},
        )
        return ActionResult.succeeded([ev], f"{self._item_name(args.item)}被弄坏了。")

    # ---------- 观测视图 ----------

    def observe(self, npc_id: NpcId) -> Observation:
        card = self.defs.characters[npc_id]
        npc = self.state.npcs[npc_id]
        loc = self.defs.locations[npc.location]

        facts: list[FactContext] = []
        for fact_id in sorted(npc.holds_facts):
            fact = self.defs.facts[fact_id]
            gate = fact.reveal_conditions
            parts: list[str] = []
            if gate.attitude_gte is not None:
                parts.append(f"对方好感需达到 {gate.attitude_gte}（当前 {npc.attitude}）")
            if gate.traded_item_in:
                # gate_hint 会原样进 prompt，必须用中文物品名——
                # 否则 NPC 会在对话里说出 rare_book 这种 id。
                names = "、".join(self._item_name(i) for i in gate.traded_item_in)
                parts.append(f"对方需先给你其中一样东西：{names}")
            facts.append(
                FactContext(
                    fact_id=fact_id,
                    content=fact.content,
                    can_reveal_now=can_reveal(npc, gate),
                    already_revealed=fact_id in npc.revealed_facts,
                    gate_hint="；".join(parts) if parts else "无条件",
                )
            )

        blind = card.knowledge.blind_to_outside

        return Observation(
            tick=self.state.tick,
            npc_id=npc_id,
            npc_name=card.name,
            location_id=npc.location,
            location_name=loc.name,
            location_description=loc.description,
            player_is_here=self.state.player.location == npc.location,
            attitude=npc.attitude,
            emotion_var=npc.emotion_var,
            emotion=npc.emotion,
            mode=npc.mode,
            mode_speech_hint=self._mode_hint(npc_id),
            own_inventory=self._named(npc.inventory),
            items_here=self._named(self.state.locations[npc.location].items),
            received_from_player=sorted(self._item_name(item) for item in npc.received_items),
            others_here=[
                self.defs.characters[other].name
                for other in sorted(self.state.npcs)
                if other != npc_id and self.state.npcs[other].location == npc.location
            ],
            facts=facts,
            quest_hint=None if blind else self._stage().hint,
            suggestion=self._suggestion(npc_id, facts),
        )

    def _suggestion(self, npc_id: NpcId, facts: list[FactContext]) -> str:
        """把引擎已经知道的事直说，别让小模型自己推。"""
        parts: list[str] = []

        ready = [f for f in facts if f.can_reveal_now and not f.already_revealed]
        if ready:
            f = ready[0]
            parts.append(
                f"来访者已经让你觉得值得开口了。他一旦问起，就用 reveal_info 把这件事"
                f"告诉他（fact 参数填 {f.fact_id}），别再打发他走。"
            )

        if (
            self.state.quest.stage >= QuestStage.S3_SOURCE
            and self.defs.ending_by(npc_id) is not None
        ):
            here = self.state.npcs[npc_id].location
            if here == ANOMALY_SITE:
                parts.append("你就在无缘塚。要动手就用 use_spellcard。")
            else:
                parts.append(
                    "线索已经凑齐了，源头在无缘塚。来访者要是叫你一起去，就用 travel_to "
                    "（destination 填 muenzuka）直接过去，路上几步不用你操心。"
                )

        return "".join(parts)

    def _stage(self) -> StageDef:
        return self.defs.stages[self.state.quest.stage.name]

    def _will_talk(self, npc_id: NpcId) -> bool:
        """她现在是否有话可说。机器可读，别让调用方去匹配 objective 的中文
        文案——那是「error 与 error_code 分家」那条教训在面板层的重现。"""
        npc = self.state.npcs[npc_id]
        return any(
            fid not in npc.revealed_facts
            and can_reveal(npc, self.defs.facts[fid].reveal_conditions)
            for fid in npc.holds_facts
        )

    def _refusal(self, npc_id: NpcId) -> str:
        """当前情绪模式若声明了 refusal，她就不搭话，返回给玩家看的那一行。"""
        return self._current_mode(npc_id).refusal

    def _mood_warning(self, npc_id: NpcId) -> str:
        """再被搭话几次她就要翻脸了。没有可预告的惩罚只有陷阱。

        倒计时必须按**回合内的时序**算，不能按「每回合净增多少」除一除。
        一个回合里的顺序是「玩家发言推高情绪 → 判定模式 → 回合末衰减」，
        所以翻脸发生在**发言那一刻**，那一次的衰减还没扣。第一版按净增算
        出「再缠 3 次」，而她下一回合就翻脸了——多算了一次尚未发生的衰减。

        第 t 次发言时她的情绪是 `emotion + (t-1)·net + delta`，令它越过门槛：

            t = 1 + ceil((门槛 − 当前 − delta) / net)
        """
        card = self.defs.characters[npc_id]
        npc = self.state.npcs[npc_id]
        delta = emotion_delta_for(card, "player_talked")
        net = delta - card.emotion.decay_per_tick
        if net <= 0:
            # 她的情绪不会因为说话而上升，这条路走不到，预警也就没有意义。
            return ""
        for mode in card.emotion.modes:
            if not mode.approaching or mode.name == npc.mode:
                continue
            # 进入门槛比裸阈值高一个迟滞带宽，和 resolve_mode 用同一个数。
            threshold = mode.range[0] + MODE_HYSTERESIS
            turns = 1 + math.ceil((threshold - npc.emotion - delta) / net)
            if 0 < turns <= WARN_WITHIN_TURNS:
                return mode.approaching.format(turns=turns)
        return ""

    def _player_objective(self) -> str:
        """玩家可见的当前目标。门槛一开就要变，否则玩家会一直投赛钱
        而不知道该开口问了。"""
        base = self._stage().objective
        openers = [
            self.defs.characters[nid].name for nid in self._npcs_here() if self._will_talk(nid)
        ]
        if openers:
            return f"{'、'.join(openers)}已经愿意开口了——直接问她无缘塚的事。"

        if self.state.quest.stage == QuestStage.S3_SOURCE:
            finishers = [
                self.defs.characters[nid].name
                for nid in self._npcs_here()
                if self.defs.ending_by(nid) is not None
            ]
            if finishers and self.state.player.location == ANOMALY_SITE:
                return f"{'、'.join(finishers)}就在这儿——让她动手。"

        return base

    def _oblivion_warning(self) -> str:
        left = OBLIVION_THRESHOLD - self.state.player.oblivion_exposure
        if self.state.player.oblivion_exposure < OBLIVION_WARNING:
            return ""
        return f"思绪开始模糊了——再在这里待 {left} 步，你会忘掉一件事。"

    def observe_player(self) -> PlayerView:
        loc_id = self.state.player.location
        loc = self.defs.locations[loc_id]
        ending_id = self.state.quest.ending
        ending = self.defs.endings.get(ending_id) if ending_id else None

        return PlayerView(
            tick=self.state.tick,
            location_id=loc_id,
            location_name=loc.name,
            location_description=loc.description,
            exits=[self.defs.locations[e].name for e in loc.exits],
            inventory=self._named(self.state.player.inventory),
            items_here=self._named(self.state.locations[loc_id].items),
            known_facts=[self.defs.facts[f].content for f in sorted(self.state.player.known_facts)],
            known_fact_ids=sorted(self.state.player.known_facts),
            quest_stage=self.state.quest.stage.name,
            quest_hint=self._stage().hint,
            objective=self._player_objective(),
            oblivion_warning=self._oblivion_warning(),
            ending_title=ending.title if ending else "",
            ending_text=ending.text.strip() if ending else "",
            npcs_here=[
                NpcPanel(
                    npc_id=nid,
                    name=self.defs.characters[nid].name,
                    attitude=self.state.npcs[nid].attitude,
                    emotion_var=self.state.npcs[nid].emotion_var,
                    emotion=self.state.npcs[nid].emotion,
                    mode=self.state.npcs[nid].mode,
                    mode_hint=self._mode_hint(nid),
                    will_talk=self._will_talk(nid),
                    mood_warning=self._mood_warning(nid),
                    refusal=self._refusal(nid),
                )
                for nid in self._npcs_here()
            ],
        )
