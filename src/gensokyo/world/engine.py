from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from gensokyo.world.defs import WorldDefs
from gensokyo.world.events import Event, EventKind
from gensokyo.world.ids import EventId, LocationId, NpcId
from gensokyo.world.rules import apply_emotion_decay
from gensokyo.world.state import WorldState
from gensokyo.world.tools import (
    TOOL_REGISTRY,
    Action,
    ActionResult,
    ErrorCode,
    MoveArgs,
    SayArgs,
    ToolSpec,
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
        return handler(action, args)

    def available_tools(self, npc_id: NpcId) -> list[ToolSpec]:
        """情绪状态机的落点：模式变化会真的改变 NPC 能做什么，
        而不是在 prompt 里描述一句。"""
        card = self.defs.characters[npc_id]
        npc = self.state.npcs[npc_id]

        allowed = set(TOOL_REGISTRY) - set(card.tools.deny_always)
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
        return {"say": self._do_say, "move": self._do_move}

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
