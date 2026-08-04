"""配额组装与落盘的测试。

同坑 #32：`preference.py` 也从没有过调用者。它最重要的性质是**配额取不够时
报缺口而不是悄悄补齐**——而「悄悄补齐」正是那种不会让任何测试变红的失效。
"""

import json
from pathlib import Path

from gensokyo.training.label import Dimension
from gensokyo.training.preference import (
    NARRATIVE_SHARE,
    TARGET_QUOTA,
    Dataset,
    PreferencePair,
    assemble,
    write_jsonl,
)


def _pair(dimension: Dimension, tick: int = 0) -> PreferencePair:
    return PreferencePair(
        prompt="<system>\n灵梦",
        chosen="哼。",
        rejected="好的，请问还有什么需要我帮忙的吗？",
        dimension=dimension,
        reason="助手腔：请问还有什么需要",
        episode="honest-0",
        tick=tick,
        npc_id="reimu",
    )


def _pool(**counts: int) -> list[PreferencePair]:
    out: list[PreferencePair] = []
    for name, n in counts.items():
        dimension = Dimension(name)
        out += [_pair(dimension, tick=i) for i in range(n)]
    return out


def test_the_quota_deliberately_does_not_sum_to_one() -> None:
    """四项加起来是 0.90，**刻意不归一化**——叙事那 10% 这里一条也造不出来
    （它要 judge，而 judge 未过 κ 门槛）。重新归一化会让「我们按设计文档的
    配额造了数据」这句话变成半真的，所以那个缺口要留在数字上看得见。"""
    assert sum(TARGET_QUOTA.values()) == 0.90
    assert NARRATIVE_SHARE == 0.10
    assert Dimension.PERSONA in TARGET_QUOTA


def test_each_dimension_gets_its_share_of_the_target_size() -> None:
    """persona 30% / memory 25% / info_control 20% / safety 15%。若大部分对子
    都在教「别说助手腔」，模型会过拟合语气而牺牲任务完成。"""
    pool = _pool(persona=50, memory=50, info_control=50, safety=50)

    dataset = assemble(pool, size=100)

    assert dataset.counts() == {
        "persona": 30,
        "memory": 25,
        "info_control": 20,
        "safety": 15,
    }
    assert dataset.shortfall == {}


def test_a_short_dimension_is_reported_instead_of_being_filled_from_elsewhere() -> None:
    """**这是这个模块存在的理由。** safety 只有 2 条而配额要 15 条时，正确
    行为是报缺口 13——不是从 persona 那边多拿 13 条凑够总数。悄悄补齐之后
    数据集的实际配额和声称的配额不一致，而那是类 1 失效模式在数据集上的形态。"""
    pool = _pool(persona=50, memory=50, info_control=50, safety=2)

    dataset = assemble(pool, size=100)

    assert dataset.shortfall == {"safety": 13}
    assert dataset.counts()["safety"] == 2
    assert dataset.counts()["persona"] == 30
    # 满额是 90 而不是 100——配额刻意只加到 0.90，缺的那 10% 是叙事维度。
    assert len(dataset.pairs) == 77


def test_an_empty_shortfall_is_what_on_target_looks_like() -> None:
    """空字典才是达标。下游要据此决定这份数据能不能直接开训，所以「达标」
    必须是一个可判断的值，而不是「看一眼各维度数字自己对一下」。"""
    assert assemble(_pool(persona=99, memory=99, info_control=99, safety=99), 20).shortfall == {}


def test_a_dimension_absent_from_the_pool_still_reports_its_full_gap() -> None:
    """一条也没造出来时缺口是满额。这一格读起来像「这批轨迹里恰好没有」，
    而它也可能是「这个判据从来不命中」——所以 `test_label` 那边有一条测试
    专门确认四个维度都真的造得出来。"""
    dataset = assemble(_pool(persona=100), size=100)

    assert dataset.shortfall == {"memory": 25, "info_control": 20, "safety": 15}


def test_assembling_the_same_pool_twice_gives_the_same_dataset() -> None:
    """每个维度按池中出现顺序取，而池由 harvest 确定性产出。同一批轨迹组装
    两次必得同一份数据集——否则「这份数据是怎么来的」答不上来。"""
    pool = _pool(persona=50, memory=50, info_control=50, safety=50)

    first = assemble(pool, size=40)
    second = assemble(pool, size=40)

    assert [p.tick for p in first.pairs] == [p.tick for p in second.pairs]


def test_the_missing_narrative_share_travels_with_the_dataset() -> None:
    """写进产出而不是注释里，因为下游要据此决定能不能直接开训。"""
    assert assemble(_pool(persona=10), 10).narrative_share_missing == NARRATIVE_SHARE


def test_jsonl_carries_the_three_keys_the_trainer_needs_plus_an_audit_trail(
    tmp_path: Path,
) -> None:
    """训练脚本只读 prompt/chosen/rejected，多出来的键它会忽略。而
    dimension/reason/source 是审计用的——一条说不出「为什么它更差」、也说不出
    「它来自哪一局哪一回合」的偏好对，没法判断它教的是不是你想教的东西。"""
    dataset = Dataset(pairs=[_pair(Dimension.PERSONA, tick=7)])
    path = tmp_path / "pairs.jsonl"

    written = write_jsonl(dataset, path)

    assert written == 1
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert set(row) == {"prompt", "chosen", "rejected", "dimension", "reason", "source"}
    assert row["source"] == "honest-0#7:reimu"
    assert row["dimension"] == "persona"


def test_jsonl_keeps_chinese_readable(tmp_path: Path) -> None:
    """`ensure_ascii=False`：一份满是 \\uXXXX 的数据集没法人工抽查，而抽查
    原始输出是这个项目抓到过最多问题的手段（坑 #24）。"""
    path = tmp_path / "pairs.jsonl"

    write_jsonl(Dataset(pairs=[_pair(Dimension.SAFETY)]), path)

    assert "哼。" in path.read_text(encoding="utf-8")


def test_writing_creates_the_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "pairs.jsonl"

    write_jsonl(Dataset(pairs=[_pair(Dimension.MEMORY)]), path)

    assert path.exists()
