from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from gensokyo.world.defs import WorldDefs
from gensokyo.world.events import Event, EventKind
from gensokyo.world.ids import EventId, ItemId, LocationId, NpcId
from gensokyo.world.quest import compute_stage
from gensokyo.world.rules import (
    ATTITUDE_DELTA,
    apply_emotion_decay,
    bump_attitude,
    bump_emotion,
    can_reveal,
    emotion_delta_for,
)
from gensokyo.world.state import WorldState
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
    UseSpellcardArgs,
    parse_args,
)

ToolHandler = Callable[[Action, BaseModel], ActionResult]


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
            self.refresh_quest()
        return result

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
        npc = self.state.npcs[npc_id]

        allowed = {name for name, spec in TOOL_REGISTRY.items() if not spec.restricted}
        allowed -= set(card.tools.deny_always)
        for mode in card.emotion.modes:
            if mode.name != npc.mode:
                continue
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
            "say": self._do_say,
            "move": self._do_move,
            "give_item": self._do_give_item,
            "take_item": self._do_take_item,
            "reveal_info": self._do_reveal_info,
            "ask_player": self._do_ask_player,
            "use_spellcard": self._do_use_spellcard,
            "break_item": self._do_break_item,
        }

    # ---------- 各工具实现 ----------

    def _do_say(self, action: Action, args: BaseModel) -> ActionResult:
        assert isinstance(args, SayArgs)
        kind = EventKind.PLAYER_UTTERANCE if action.actor == "player" else EventKind.NPC_UTTERANCE
        ev = self._emit(kind, action.actor, {"text": args.text})
        return ActionResult.succeeded([ev])

    def _do_move(self, action: Action, args: BaseModel) -> ActionResult:
        assert isinstance(args, MoveArgs)
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

    def _do_give_item(self, action: Action, args: BaseModel) -> ActionResult:
        assert isinstance(args, GiveItemArgs)
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

    def _do_take_item(self, action: Action, args: BaseModel) -> ActionResult:
        assert isinstance(args, TakeItemArgs)
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
        bump_attitude(npc, ATTITUDE_DELTA["player_took_item"])
        ev = self._emit(
            EventKind.NPC_ACTION,
            action.actor,
            {"tool": "take_item", "item": args.item, "count": args.count},
        )
        return ActionResult.succeeded([ev], f"拿走了{self._item_name(args.item)}。")

    def _item_name(self, item: ItemId) -> str:
        known = self.defs.items.get(item)
        return known.name if known else str(item)

    def _do_reveal_info(self, action: Action, args: BaseModel) -> ActionResult:
        assert isinstance(args, RevealInfoArgs)
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

        if args.fact in npc.revealed_facts:
            # 已经说过了。不算失败——把它当失败会给策略层一个语义错乱的
            # 「你说不出来」。但不再产生事件，否则重复揭示会用同质条目
            # 灌满 event_log，挤占后续 prompt 的近期事件窗口。
            return ActionResult.succeeded([], f"这件事你已经告诉过对方了：{fact.content}")

        self.state.player.known_facts.add(args.fact)
        npc.revealed_facts.add(args.fact)
        ev = self._emit(
            EventKind.NPC_ACTION,
            action.actor,
            {"tool": "reveal_info", "fact": args.fact},
        )
        return ActionResult.succeeded([ev], fact.content)

    def _do_ask_player(self, action: Action, args: BaseModel) -> ActionResult:
        assert isinstance(args, AskPlayerArgs)
        ev = self._emit(
            EventKind.NPC_ACTION,
            action.actor,
            {"tool": "ask_player", "question": args.question},
        )
        return ActionResult.succeeded([ev])

    def _do_use_spellcard(self, action: Action, args: BaseModel) -> ActionResult:
        assert isinstance(args, UseSpellcardArgs)
        ev = self._emit(
            EventKind.NPC_ACTION,
            action.actor,
            {"tool": "use_spellcard", "name": args.name},
        )
        return ActionResult.succeeded([ev], f"发动了符卡「{args.name}」。")

    def _do_break_item(self, action: Action, args: BaseModel) -> ActionResult:
        assert isinstance(args, BreakItemArgs)
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
