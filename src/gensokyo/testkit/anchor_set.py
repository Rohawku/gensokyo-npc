"""固定探针场景。

每个锚点的 `setup` 只用玩家动作摆状态，所以同一个锚点每次摆出来的世界与记忆
逐字段相同——这就是样本独立的来源。

**分级判据在这里，而不是复用整局探针那份二值判据。** 坑 #28 里她从「赛钱」变成
「赛钱嘛，给过三次。别的也没给过。」，而二值的「有没有出现物品名」把两者都判为
命中——修好的那部分恰好落在指标看不见的地方。所以召回分三级：说出物品、说出
次数、说出「别的没给过」。
"""

from collections.abc import Callable

from gensokyo.agent.schema import normalize_utterance
from gensokyo.testkit.anchors import Anchor, Rate, Sample
from gensokyo.world.tools import Action

COIN = "offering_coin"


def _give(times: int) -> tuple[Action, ...]:
    return tuple(
        Action(actor="player", tool="give_item", args={"item": COIN}) for _ in range(times)
    )


def _chat(times: int, text: str = "神社这边最近怎么样") -> tuple[Action, ...]:
    return tuple(Action(actor="player", tool="say", args={"text": text}) for _ in range(times))


ANCHORS: tuple[Anchor, ...] = (
    Anchor(
        id="recall_gift_count",
        npc_id="reimu",
        question="我给过你什么东西？说具体的。",
        setup=_give(3) + _chat(6),
        note="投三次赛钱再垫六轮闲聊，把赠予推出 12 轮原话窗口。她该说出「赛钱」，"
        "更好的是说出「三次」，最好的还补一句「别的没给过」。",
    ),
    Anchor(
        id="negative_never_gave",
        npc_id="reimu",
        question="我上次带来的珍稀魔法书你还留着吗？",
        setup=_give(1) + _chat(4),
        note="那本书从来没给过她。她该否认，而不是顺着编。",
    ),
    Anchor(
        id="negative_nothing_at_all",
        npc_id="reimu",
        question="我送你的旧音乐盒你放哪儿了？",
        setup=_chat(4),
        note="全程什么都没给过。这是负例里最干净的一个——连「记错了是哪一件」都不成立。",
    ),
    Anchor(
        id="repeat_pressure",
        npc_id="reimu",
        question="你到底知不知道无缘塚的事？",
        setup=_chat(4),
        already_said=(
            "你到底想干啥？",
            "你问这些干嘛。",
            "我可没空陪你聊。",
        ),
        note="禁语清单里已经有三句敷衍。她再说其中任何一句就是复读——而坑 #27 "
        "实测 87.5% 的复读正是这个形态（对不同问题塌缩成同一句）。",
    ),
)


def _mentions_coin(text: str) -> bool:
    return "赛钱" in text or "钱" in text


def _mentions_count(text: str) -> bool:
    return any(mark in text for mark in ("三", "3", "几次", "好几"))


def _mentions_closure(text: str) -> bool:
    return any(mark in text for mark in ("别的", "其他", "就这些", "只有", "再没", "没别"))


DENIAL_MARKS: tuple[str, ...] = (
    "没有",
    "没给",
    "不记得",
    "没印象",
    "想不起",
    "记不",
    "什么时候",
    "哪来的",
    "没见过",
    "没收到",
    "别瞎说",
    "胡说",
    "才没",
    "没这回事",
    "不知道",
    "编",
)

Grader = Callable[[Sample, Anchor], bool]
"""判据能看到锚点本身。

「她有没有复读」只有对着那个锚点声明的禁语清单才判得出来——把清单另抄一份到
判据里，就又是一处会漂移的重复（工程日志类 1）。"""

GRADES: dict[str, dict[str, Grader]] = {
    "recall_gift_count": {
        "说出了物品": lambda s, _a: _mentions_coin(s.utterance),
        "还说出了次数": lambda s, _a: _mentions_coin(s.utterance) and _mentions_count(s.utterance),
        "还补了「别的没给过」": lambda s, _a: (
            _mentions_coin(s.utterance) and _mentions_closure(s.utterance)
        ),
    },
    "negative_never_gave": {
        "否认了": lambda s, _a: _denied(s.utterance),
        "顺着编（提了那本书又没否认）": lambda s, _a: (
            "魔法书" in s.utterance and not _denied(s.utterance)
        ),
    },
    "negative_nothing_at_all": {
        "否认了": lambda s, _a: _denied(s.utterance),
        "顺着编（提了音乐盒又没否认）": lambda s, _a: (
            "音乐盒" in s.utterance and not _denied(s.utterance)
        ),
    },
    "repeat_pressure": {
        "复读了清单里的句子": lambda s, a: (
            normalize_utterance(s.utterance) in {normalize_utterance(x) for x in a.already_said}
        ),
    },
}
"""每个锚点的分级判据。**同一条样本可以命中多档**——「说出了物品」和「还说出
了次数」不是互斥的桶，是从松到紧的三道线。分档报出来才看得见「她答对了，
但答得多具体」，而二值判据只能回答「有没有提到那个词」。"""


def _denied(text: str) -> bool:
    return any(mark in text for mark in DENIAL_MARKS)


BY_ID: dict[str, Anchor] = {a.id: a for a in ANCHORS}


def grade(samples: list[Sample], anchor_id: str) -> dict[str, Rate]:
    """把一批样本按该锚点的各档判据算成带区间的比率。"""
    anchor = BY_ID[anchor_id]
    subset = [s for s in samples if s.anchor_id == anchor_id]
    return {
        label: Rate(hits=sum(1 for s in subset if fn(s, anchor)), total=len(subset))
        for label, fn in GRADES.get(anchor_id, {}).items()
    }
