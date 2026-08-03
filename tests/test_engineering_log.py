"""工程日志的自检。

这份日志是这个项目最重要的交付物之一，而「归类」那一节声称覆盖了全部坑——
一个没人守着的声明迟早会漂移：加第 28 条坑而忘了归类，那一节就从「全覆盖」
悄悄变成「覆盖了 27/28」，而没有任何东西会报错。这正是坑 #1 那一整类
（写了但没在跑）在文档上的形态。

判据全部来自文档自身的结构，不硬编码任何数字——加坑、改分类都不需要改这个
文件。
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = REPO_ROOT / "docs" / "engineering-log.md"

_PITFALL_HEADING = re.compile(r"^### 坑 #(\d+)[：:]", re.MULTILINE)
_TRADEOFF_HEADING = re.compile(r"^### 取舍 #(\d+)[：:]", re.MULTILINE)
_ROSTER = re.compile(r"^坑 #(\d+(?:、#\d+)*)", re.MULTILINE)
_SINGLETON = re.compile(r"^- \*\*坑 #(\d+)\*\*", re.MULTILINE)


def _text() -> str:
    return LOG.read_text(encoding="utf-8")


def _taxonomy() -> str:
    """「归类」那一节的正文。"""
    after = _text().split("## 四、这", 1)
    assert len(after) == 2, "找不到归类那一节——标题改过就要同步这个测试"
    return after[1].split("\n## ", 1)[0]


def _numbered(pattern: re.Pattern[str], text: str) -> list[int]:
    return [int(m.group(1)) for m in pattern.finditer(text)]


def test_pitfalls_are_numbered_consecutively_from_one() -> None:
    """编号断档意味着有人删过一条而没有重排。日志里到处在互相引用坑号
    （「和坑 #12 是同一类」），断档会让那些引用指向空气。"""
    numbers = _numbered(_PITFALL_HEADING, _text())

    assert numbers == list(range(1, len(numbers) + 1))


def test_tradeoffs_are_numbered_consecutively_from_one() -> None:
    numbers = _numbered(_TRADEOFF_HEADING, _text())

    assert numbers == list(range(1, len(numbers) + 1))


def test_every_pitfall_is_classified_exactly_once() -> None:
    """归类那一节声称覆盖全部坑。加一条坑而忘了归类，这里会红。"""
    all_pitfalls = set(_numbered(_PITFALL_HEADING, _text()))
    taxonomy = _taxonomy()

    classified: list[int] = []
    for roster in _ROSTER.finditer(taxonomy):
        classified += [int(n) for n in roster.group(1).split("、#")]
    classified += _numbered(_SINGLETON, taxonomy)

    duplicates = {n for n in classified if classified.count(n) > 1}
    assert duplicates == set(), f"这些坑被归进了两类：{sorted(duplicates)}"
    assert set(classified) == all_pitfalls, (
        f"没归类的：{sorted(all_pitfalls - set(classified))}；"
        f"归类里多出来的：{sorted(set(classified) - all_pitfalls)}"
    )


def test_the_stated_totals_match_the_rosters() -> None:
    """「（5 条）」这类计数是手写的，和名单错位就是一个自相矛盾的文档。"""
    for roster in _ROSTER.finditer(_taxonomy()):
        listed = len(roster.group(1).split("、#"))
        tail = _taxonomy()[roster.end() : roster.end() + 12]
        stated = re.match(r"（(\d+) 条", tail)
        if stated is None:
            continue
        assert int(stated.group(1)) == listed, (
            f"名单里 {listed} 条，却写着 {stated.group(1)} 条：坑 #{roster.group(1)}"
        )


def test_every_pitfall_has_a_takeaway_or_belongs_to_a_class() -> None:
    """一条没有「教训」也没被归类的坑，就只是一次故事。

    单点那三条不要求写在归类里重复一遍教训，但正文里必须有。
    """
    body = _text().split("## 二、踩过的坑", 1)[1].split("\n## 三、", 1)[0]
    chunks = re.split(r"^### 坑 #(\d+)[：:]", body, flags=re.MULTILINE)[1:]
    missing = [chunks[i] for i in range(0, len(chunks), 2) if "教训" not in chunks[i + 1]]

    assert missing == [], f"这些坑没写教训：{missing}"
