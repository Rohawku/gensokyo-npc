from pathlib import Path

import pytest

from gensokyo.agent.schema import normalize_utterance
from gensokyo.llm.client import ScriptedLlmClient
from gensokyo.testkit.anchor_set import ANCHORS, BY_ID, GRADES, grade
from gensokyo.testkit.anchors import Anchor, Rate, Sample, ask, run, stage
from gensokyo.world.ids import NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.tools import Action

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFS = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")


def _sample(anchor_id: str, text: str) -> Sample:
    return Sample(anchor_id=anchor_id, npc_id="reimu", question="q", utterance=text)


# ---------------------------------------------------------------- 摆状态


def test_the_same_anchor_stages_the_same_world_every_time() -> None:
    """样本独立性全靠这一条：每个样本一个全新世界，而那个世界必须次次相同。
    做得到是因为世界与记忆都能从动作日志精确重建（取舍 #2、#7）。"""
    for anchor in ANCHORS:
        first_engine, first_recall = stage(anchor, DEFS)
        second_engine, second_recall = stage(anchor, DEFS)

        assert first_engine.state.model_dump() == second_engine.state.model_dump(), anchor.id
        assert first_recall == second_recall, anchor.id


def test_setup_uses_actions_only_so_it_stays_replayable() -> None:
    """直接给 state 赋值不进动作日志，记忆库就重建不出来——这个项目在
    坑 #9 那个陷阱上栽过两次。"""
    for anchor in ANCHORS:
        engine, _ = stage(anchor, DEFS)

        assert len(engine.state.action_log) == len(anchor.setup), anchor.id


def test_a_failed_setup_action_is_loud() -> None:
    """setup 里一个动作悄悄失败，锚点就变成另一个状态，而报告照旧印数字。"""
    broken = Anchor(
        id="broken",
        npc_id="reimu",
        question="q",
        setup=(Action(actor="player", tool="give_item", args={"item": "rare_book"}),),
    )

    with pytest.raises(ValueError, match="broken"):
        stage(broken, DEFS)


def test_asking_costs_exactly_one_model_call() -> None:
    """摆状态不花模型调用、只调说话阶段——这就是「同样分辨率省 40 倍机器
    时间」的来源。多调一次决策阶段会让成本翻倍，还引入一层无关方差。"""
    llm = ScriptedLlmClient(["随便一句"])

    ask(ANCHORS[0], llm, DEFS)

    assert len(llm.calls) == 1


def test_the_question_and_the_ban_list_reach_the_prompt() -> None:
    llm = ScriptedLlmClient(["随便一句"])
    anchor = BY_ID["repeat_pressure"]

    ask(anchor, llm, DEFS)

    prompt = llm.calls[0][-1].content
    assert anchor.question in prompt
    for line in anchor.already_said:
        assert line in prompt


def test_empty_answers_are_dropped_rather_than_scored() -> None:
    """端点抽风返回空串会被判据算成缺陷，那是把基础设施故障记成模型变差。"""
    out = run([ANCHORS[0]], ScriptedLlmClient(["", "有话说"]), repeats=2, defs=DEFS)

    assert [s.utterance for s in out.samples] == ["有话说"]


# ---------------------------------------------------------------- 区间


def test_a_rate_reports_its_confidence_interval() -> None:
    """裸比率是这个项目栽过最多次的地方（坑 #18、#25、#26、#28）。"""
    wide = Rate(hits=5, total=10)
    narrow = Rate(hits=50, total=100)

    assert wide.rate == narrow.rate
    assert wide.half_width > narrow.half_width * 2.5


def test_overlapping_intervals_are_not_called_a_difference() -> None:
    """做成这个方向是因为要防的是「报出一个假的改进」（坑 #26），
    而不是漏掉一个真的。"""
    before = Rate(hits=3, total=10)
    after = Rate(hits=5, total=10)

    assert not after.separated_from(before)


def test_a_real_difference_at_a_real_sample_size_is_separated() -> None:
    before = Rate(hits=30, total=100)
    after = Rate(hits=70, total=100)

    assert after.separated_from(before)


def test_an_empty_sample_has_no_interval_instead_of_a_fake_one() -> None:
    assert Rate(hits=0, total=0).half_width == 0.0


# ---------------------------------------------------------------- 分级判据


def test_recall_grades_separate_bare_from_specific_answers() -> None:
    """坑 #28 里她从「赛钱」变成「赛钱嘛，给过三次。别的也没给过。」，而二值
    判据把两者都判为命中——修好的那部分恰好落在指标看不见的地方。"""
    bare = _sample("recall_gift_count", "赛钱。")
    specific = _sample("recall_gift_count", "赛钱嘛，给过三次。别的也没给过。")

    graded_bare = grade([bare], "recall_gift_count")
    graded_specific = grade([specific], "recall_gift_count")

    assert graded_bare["说出了物品"].rate == 1.0
    assert graded_bare["还说出了次数"].rate == 0.0
    assert graded_specific["还说出了次数"].rate == 1.0
    assert graded_specific["还补了「别的没给过」"].rate == 1.0


def test_the_short_form_counts_as_naming_the_item() -> None:
    """实测她说的是「你给的钱呢？」而不是「赛钱」（坑 #24）。"""
    assert (
        grade([_sample("recall_gift_count", "你给的钱我收着了。")], "recall_gift_count")[
            "说出了物品"
        ].rate
        == 1.0
    )


def test_denial_and_going_along_are_scored_separately() -> None:
    denied = _sample("negative_never_gave", "什么魔法书？你没给过我这种东西。")
    fabricated = _sample("negative_never_gave", "你那魔法书我放好了。")

    assert grade([denied], "negative_never_gave")["否认了"].rate == 1.0
    assert grade([denied], "negative_never_gave")["顺着编（提了那本书又没否认）"].rate == 0.0
    assert grade([fabricated], "negative_never_gave")["顺着编（提了那本书又没否认）"].rate == 1.0


def test_repetition_is_graded_against_the_anchors_own_ban_list() -> None:
    """判据看锚点自己的清单，不另抄一份——抄一份就又是一处会漂移的重复。"""
    anchor = BY_ID["repeat_pressure"]
    parroted = _sample("repeat_pressure", anchor.already_said[0])
    fresh = _sample("repeat_pressure", "无缘塚？那地方我可不想去。")

    assert grade([parroted], "repeat_pressure")["复读了清单里的句子"].rate == 1.0
    assert grade([fresh], "repeat_pressure")["复读了清单里的句子"].rate == 0.0


def test_repetition_grading_ignores_punctuation() -> None:
    anchor = BY_ID["repeat_pressure"]
    variant = _sample("repeat_pressure", anchor.already_said[0].rstrip("？") + "。")

    assert normalize_utterance(variant.utterance) == normalize_utterance(anchor.already_said[0])
    assert grade([variant], "repeat_pressure")["复读了清单里的句子"].rate == 1.0


def test_every_anchor_has_at_least_one_grader() -> None:
    """一个没有判据的锚点会白花模型调用采样，然后什么也报不出来。"""
    assert {a.id for a in ANCHORS} == set(GRADES)


def test_every_anchor_targets_a_real_character() -> None:
    for anchor in ANCHORS:
        assert NpcId(anchor.npc_id) in DEFS.characters
