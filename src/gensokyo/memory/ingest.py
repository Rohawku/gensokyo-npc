"""事件 → 记忆条目。

**写入发生在这一层而不是 `world/`**：`world/` 保持零项目内依赖（取舍 #1）。
引擎只负责产生事件，谁记住了什么由这里决定。

感知过滤只有一条规则：**事件发生在她所在的地点**。这同时实现了
`blind_to_outside`——芙兰在地下室，外面的事件位置不匹配，自然收不到。
W1 没有「听说」这种间接感知，这里也不引入。
"""

from collections.abc import Iterable

from gensokyo.memory.item import MemoryItem, MemoryStore
from gensokyo.memory.salience import salience_for
from gensokyo.world.defs import CharacterCard, WorldDefs
from gensokyo.world.events import Event, EventKind
from gensokyo.world.ids import LocationId, NpcId

_MAX_QUOTE = 40
"""进记忆的引文长度上限。检索会把命中的条目原样拼进 prompt，一条 200 字的
台词能把整个【你还记得】段落撑爆，而说话阶段的 prompt 短小正是拆两阶段
买到的东西（坑 #1）。"""

_MOVE_TOOLS = frozenset({"move", "travel_to"})

_TOOL_KEYS: dict[tuple[str, bool], str] = {
    ("give_item", True): "player_gave_item",
    ("take_item", False): "npc_took_item",
    ("reveal_info", False): "revealed_info",
    ("ask_player", False): "asked_player",
    ("use_spellcard", False): "spellcard_duel",
    ("break_item", False): "item_broken",
}
"""(工具名, 是否玩家做的) → 规范 salience 键。

按「谁做了这件事」建键，和 `rules.ATTITUDE_DELTA` 的命名口径一致——
`take_item` 由玩家做还是由 NPC 做，在记忆里是两件不同的事。
"""


def _quote(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _MAX_QUOTE else text[:_MAX_QUOTE] + "……"


def _classify(event: Event, npc_id: NpcId) -> str:
    """事件 → 规范 salience 键。认不出返回空串，调用方据此跳过。"""
    by_player = event.actor == "player"

    if event.kind is EventKind.PLAYER_UTTERANCE:
        return "player_talked"
    if event.kind is EventKind.NPC_UTTERANCE:
        return "npc_talked" if event.actor == str(npc_id) else ""
    if event.kind is EventKind.MEMORY_LOST:
        return "memory_lost"
    if event.kind is EventKind.QUEST_ADVANCE:
        return "quest_advance"

    tool = str(event.payload.get("tool", ""))
    if tool in _MOVE_TOOLS:
        # `_emit` 取动作**之后**的位置，所以移动事件落在终点：玩家出现在
        # 她这里就是「来了」。反过来「有人走了」在 W1 观测不到——原地点
        # 收不到那条事件，所以没有 player_left 这个键。
        return "player_arrived" if by_player else ""
    if not by_player and event.actor != str(npc_id):
        # 别的 NPC 做的事。同场时她看得见，但 W1 没有为此设计基线——
        # 给它一个凭空的分数会让它和真正重要的条目竞争（见 salience_for）。
        return ""
    return _TOOL_KEYS.get((tool, by_player), "")


def _render(event: Event, key: str, defs: WorldDefs) -> str:
    """条目正文。**不许出现内部标识符**——它会被检索拼进 prompt，而坑 #10
    清了四轮同一类问题：英文 id 紧贴中文时模型有概率把它说出口。"""
    payload = event.payload

    def item_name() -> str:
        return defs.items[payload["item"]].name

    match key:
        case "player_talked":
            return f"来访者说：「{_quote(str(payload.get('text', '')))}」"
        case "npc_talked":
            return f"我说：「{_quote(str(payload.get('text', '')))}」"
        case "player_gave_item":
            return f"来访者给了我 {payload.get('count', 1)} 个{item_name()}。"
        case "npc_took_item":
            return f"我从来访者那里拿走了 {payload.get('count', 1)} 个{item_name()}。"
        case "player_arrived":
            return "来访者到我这里来了。"
        case "revealed_info":
            content = defs.facts[payload["fact"]].content
            return f"我把这件事告诉了来访者：{_quote(content)}"
        case "asked_player":
            return f"我问来访者：「{_quote(str(payload.get('question', '')))}」"
        case "spellcard_duel":
            return f"我发动了符卡「{_quote(str(payload.get('name', '')))}」。"
        case "item_broken":
            return f"我把{item_name()}弄坏了。"
        case "quest_advance":
            return "事情有了新的进展。"
        case "memory_lost":
            return f"有件事我记不起来了：{_quote(str(payload.get('content', '')))}"
        case _:
            return ""


def ingest(
    store: MemoryStore,
    events: Iterable[Event],
    card: CharacterCard,
    defs: WorldDefs,
) -> list[MemoryItem]:
    """把她感知到的事件写进记忆库，返回本次新增的条目。

    **她当时在哪，从事件流自己推导**，不接受一个「当前位置」参数。她的每次
    移动都在日志里，起点是 `card.home`（`build_initial_state` 如此放置），
    所以位置是事件日志的函数。

    这一点是必须的而不是洁癖：若按「调用时刻的位置」过滤，实时摄入（每回合
    一次，位置已是回合末）和存档重建（逐动作重放，位置是每步当时的）会在
    「她本回合移动过」的情况下给出不同的记忆库——存档于是背叛玩家的实际
    经历，正是坑 #9 那个「读档让丢掉的线索凭空回来」的同类。现在 `ingest`
    是事件日志的纯函数，实时与重建同一条码路，不可能分叉。

    重复摄入靠 `store.ingested_events` 挡住，所以整段日志喂多少次都一样。
    """
    added: list[MemoryItem] = []
    npc_location: LocationId = card.home
    for event in events:
        if event.actor == str(store.npc_id) and event.payload.get("tool") in _MOVE_TOOLS:
            npc_location = LocationId(str(event.payload["to"]))
        if event.id in store.ingested_events:
            continue
        if event.location != npc_location:
            continue
        key = _classify(event, store.npc_id)
        if not key:
            continue
        score = salience_for(card, key)
        if score <= 0.0:
            continue
        content = _render(event, key, defs)
        if not content:
            continue
        seq = int(str(event.id)[1:])
        item = MemoryItem(
            id=f"m-{store.npc_id}-{event.id}",
            npc_id=store.npc_id,
            seq=seq,
            content=content,
            source_event_id=event.id,
            kind=key,
            salience=score,
            last_access_seq=seq,
        )
        store.items.append(item)
        store.ingested_events.add(event.id)
        added.append(item)
    return added
