"""记忆的推进与重建：**实时与读档走同一条码路**。

这个模块存在的唯一理由是防止两条码路分叉。若实时对话用一套逻辑摄入、读档
用另一套重建，两者迟早给出不同的记忆库——而那意味着存档改变了「她记得
什么」，比丢一条线索更糟（坑 #9）。所以 `absorb` 只有一个实现，
`Session` 每回合调它，`rebuild` 重放动作后也调它。

能做到这一点是因为链条上每一环都是事件日志的纯函数：

    动作日志 --replay--> 事件日志 --ingest--> 记忆条目 --demote--> 分层

`ingest` 自己从事件流推导她当时在哪，`demote` 只读 `seq` 与 `salience`，
两者都不看「调用时刻」的任何状态。
"""

from gensokyo.memory.decay import demote
from gensokyo.memory.ingest import ingest
from gensokyo.memory.item import MemoryStore
from gensokyo.world.defs import WorldDefs
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import NpcId
from gensokyo.world.tools import Action


def new_stores(defs: WorldDefs) -> dict[NpcId, MemoryStore]:
    return {npc_id: MemoryStore(npc_id=npc_id) for npc_id in defs.characters}


def now_seq(engine: WorldEngine) -> int:
    """当前的事件序号。空日志返回 0。"""
    log = engine.state.event_log
    return int(str(log[-1].id)[1:]) if log else 0


def absorb(engine: WorldEngine, stores: dict[NpcId, MemoryStore]) -> None:
    """把日志里每个 NPC 感知到的部分摄入，然后重算分层。

    整段日志喂进去而不是只喂增量：`ingest` 靠 `ingested_events` 去重，而
    整段喂让这个函数无状态——调用频率不影响结果，于是「每回合调一次」和
    「重放时每个动作调一次」必然一致。
    """
    seq = now_seq(engine)
    for npc_id, store in stores.items():
        card = engine.defs.characters[npc_id]
        ingest(store, engine.state.event_log, card, engine.defs)
        demote(store, card, seq)


def rebuild(actions: list[Action], defs: WorldDefs) -> tuple[WorldEngine, dict[NpcId, MemoryStore]]:
    """从动作日志重建世界与记忆。

    存档里仍然只有动作日志（取舍 #2）——记忆不进存档，它是推导产物。存两份
    就会不一致，而不一致的那一份是玩家的记忆，最难发现。
    """
    engine = WorldEngine.replay(actions, defs)
    stores = new_stores(defs)
    absorb(engine, stores)
    return engine, stores
