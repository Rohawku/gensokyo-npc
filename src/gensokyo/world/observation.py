from pydantic import BaseModel, Field

from gensokyo.world.ids import FactId, ItemId, LocationId, NpcId


class FactContext(BaseModel):
    fact_id: FactId
    content: str
    can_reveal_now: bool
    already_revealed: bool
    gate_hint: str


class Observation(BaseModel):
    """NPC 视角的世界快照。只包含她该知道的东西——
    信息可见性是世界规则，不能靠在 prompt 里叮嘱模型别乱说。"""

    tick: int
    npc_id: NpcId
    npc_name: str
    location_id: LocationId
    location_name: str
    location_description: str
    player_is_here: bool
    attitude: int
    emotion_var: str
    emotion: float
    mode: str
    mode_speech_hint: str
    own_inventory: dict[str, int] = Field(default_factory=dict)
    items_here: dict[str, int] = Field(default_factory=dict)
    """以中文物品名为键。Observation 只服务 prompt 组装，
    英文 id 泄漏进去就会被 NPC 说出口；面板数据走 PlayerView。"""
    others_here: list[str] = Field(default_factory=list)
    facts: list[FactContext] = Field(default_factory=list)
    quest_hint: str | None = None
    """剧情进展的中文说法。为 None 表示该 NPC 被信息隔离，不知道外面的事。"""


class NpcPanel(BaseModel):
    npc_id: NpcId
    name: str
    attitude: int
    emotion_var: str
    emotion: float
    mode: str


class PlayerView(BaseModel):
    """右栏面板数据。玩家必须能立刻看到自己行为的后果，
    否则记忆与情绪系统做得再好也感知不到。"""

    tick: int
    location_id: LocationId
    location_name: str
    location_description: str
    exits: list[str] = Field(default_factory=list)
    inventory: dict[ItemId, int] = Field(default_factory=dict)
    items_here: dict[ItemId, int] = Field(default_factory=dict)
    known_facts: list[str] = Field(default_factory=list)
    quest_stage: str = ""
    npcs_here: list[NpcPanel] = Field(default_factory=list)
