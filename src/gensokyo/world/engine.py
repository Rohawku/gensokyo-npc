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
    SayArgs,
    parse_args,
)


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

        handler = self._handlers().get(action.tool)
        if handler is None:
            return ActionResult.failed(ErrorCode.UNKNOWN_TOOL, f"动作 {action.tool} 尚未实现。")
        return handler(action, args)

    def _handlers(self) -> dict[str, Any]:
        return {"say": self._do_say}

    # ---------- 各工具实现 ----------

    def _do_say(self, action: Action, args: BaseModel) -> ActionResult:
        assert isinstance(args, SayArgs)
        kind = EventKind.PLAYER_UTTERANCE if action.actor == "player" else EventKind.NPC_UTTERANCE
        ev = self._emit(kind, action.actor, {"text": args.text})
        return ActionResult.succeeded([ev])
