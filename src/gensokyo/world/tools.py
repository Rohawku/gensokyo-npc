from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gensokyo.world.events import Event
from gensokyo.world.ids import FactId, ItemId, LocationId, NpcId


class ErrorCode(StrEnum):
    """机器可读的失败原因。与给 LLM 看的自然语言 error 分开，
    这样改提示文案不会让历史指标断档。"""

    UNKNOWN_TOOL = "unknown_tool"
    BAD_ARGS = "bad_args"
    TOOL_DENIED = "tool_denied"
    NOT_CO_LOCATED = "not_co_located"
    NO_SUCH_EXIT = "no_such_exit"
    UNREACHABLE = "unreachable"
    INSUFFICIENT_ITEM = "insufficient_item"
    NOT_FACT_HOLDER = "not_fact_holder"
    REVEAL_CONDITION_UNMET = "reveal_condition_unmet"


class SayArgs(BaseModel):
    text: str


class MoveArgs(BaseModel):
    to: LocationId


class TravelToArgs(BaseModel):
    destination: LocationId


class GiveItemArgs(BaseModel):
    item: ItemId
    count: int = Field(default=1, ge=1)
    to: NpcId | None = None
    """给谁。**同场有两个人时必填**，只有一个人时可以省略。

    原先没有这个参数，目标取 `_npcs_here()[0]`——那是字母序（芙兰 < 魔理沙 <
    灵梦）。两个人同场时玩家投的赛钱会静悄悄进魔理沙的口袋，而她不收赛钱。
    默认 None 而不是必填：绝大多数回合只有一个人在场，让玩家每次都点名等于
    为一个罕见情况给常见情况加税。
    """


class TakeItemArgs(BaseModel):
    item: ItemId
    count: int = Field(default=1, ge=1)


class RevealInfoArgs(BaseModel):
    fact: FactId


class AskPlayerArgs(BaseModel):
    question: str


class UseSpellcardArgs(BaseModel):
    name: str


class BreakItemArgs(BaseModel):
    item: ItemId


class ToolSpec(BaseModel):
    name: str
    description: str
    args_model: type[BaseModel]
    restricted: bool = False
    """受限工具不默认发放给任何 NPC，只能由角色卡的某个情绪模式用
    tools_allow 显式解锁。若默认全开，每新增一个工具都会自动授予所有
    角色，除非有人记得去 deny——那个默认方向是错的。"""

    def json_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()


TOOL_REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(name="say", description="说话。", args_model=SayArgs),
        ToolSpec(name="move", description="移动到相邻地点。", args_model=MoveArgs),
        ToolSpec(
            name="travel_to",
            description="前往任意地点，路上要走几步不用你操心。",
            args_model=TravelToArgs,
        ),
        ToolSpec(name="give_item", description="把自己的物品交给对方。", args_model=GiveItemArgs),
        ToolSpec(name="take_item", description="从对方手里拿走物品。", args_model=TakeItemArgs),
        ToolSpec(
            name="reveal_info",
            description="把自己知道的某个情报告诉对方。",
            args_model=RevealInfoArgs,
        ),
        ToolSpec(name="ask_player", description="向对方提问。", args_model=AskPlayerArgs),
        ToolSpec(
            name="use_spellcard",
            description="发动符卡，以弹幕决斗解决冲突。",
            args_model=UseSpellcardArgs,
        ),
        ToolSpec(
            name="break_item",
            description="破坏一件物品。",
            args_model=BreakItemArgs,
            restricted=True,
        ),
    ]
}


def parse_args(tool: str, raw: dict[str, Any]) -> BaseModel:
    spec = TOOL_REGISTRY[tool]
    return spec.args_model.model_validate(raw)


class Action(BaseModel):
    model_config = ConfigDict(frozen=True)

    actor: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ActionResult(BaseModel):
    ok: bool
    error: str | None = None
    error_code: ErrorCode | None = None
    events: list[Event] = Field(default_factory=list)
    observation_delta: str = ""

    @classmethod
    def failed(cls, code: ErrorCode, message: str) -> "ActionResult":
        return cls(ok=False, error=message, error_code=code)

    @classmethod
    def succeeded(cls, events: list[Event], observation: str = "") -> "ActionResult":
        return cls(ok=True, events=events, observation_delta=observation)
