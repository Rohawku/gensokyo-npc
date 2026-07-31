from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class TurnRecord(BaseModel):
    """一个玩家输入及其后果。

    一次 say 若有多个 NPC 在场会产生多条记录（每个说话人一条），
    所以 `tick` 不是主键——它只是这条输入发生在第几回合。
    """

    tick: int
    player_input: str
    kind: Literal["say", "command"]
    """command 指 /go /give /pick 这类；say 是对 NPC 说话。"""

    command_ok: bool | None = None
    command_error_code: str | None = None
    """指令的失败原因。多数是 ErrorCode 的值，也有 runner 自己的
    unknown_command / missing_arg——两者都是稳定字符串，指标能统计。"""

    npc_id: str | None = None
    utterance: str = ""
    thought: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    """{"tool":..., "args":...}，本回合实际下发的每一次调用（含重试时
    失败的那次），与 tool_results 等长同序。指标靠下标把调用和结果配对。"""
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    """{"ok":..., "error_code":..., "observation":...}"""
    llm_calls: int = 0
    persona_llm_calls: int = 0
    """玩家模拟器自己消耗的调用次数。只有套话玩家非零；
    算全局成本时不能只看 NPC 那一侧。"""
    latency_ms: int = 0
    """墙上时钟，不参与确定性比对。"""
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    """本回合召回的记忆条目。记忆探针要能区分「没召回到」和「召回到了但
    她没用」——这两种失败一个调检索、一个调生成，修法完全不同。"""
    mode_before: str = ""
    mode_after: str = ""
    view_after: dict[str, Any] = Field(default_factory=dict)
    """PlayerView 的 model_dump。玩家能看到的全部信息，指标只该看这里。"""


class Trajectory(BaseModel):
    """一局的完整记录。

    存 `action_log` 而不只是存最终状态，是为了让任何一条轨迹都能被
    `WorldEngine.replay` 重建——指标算错了可以重算，不用重跑模型。
    """

    persona: str
    seed: int
    turns: list[TurnRecord] = Field(default_factory=list)
    finished: bool = False
    ending: str | None = None
    final_stage: str = ""
    action_log: list[dict[str, Any]] = Field(default_factory=list)
    """Action 的 model_dump，能重建世界。"""
    event_log: list[dict[str, Any]] = Field(default_factory=list)
    """Event 的 model_dump(mode="json")。EventKind 是 StrEnum，
    不指定 mode 会落盘成枚举对象，JSON 序列化时才炸。"""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Trajectory":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
