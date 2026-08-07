from dataclasses import replace
from pathlib import Path

import pytest

from gensokyo.agent.schema import normalize_utterance
from gensokyo.llm.client import ScriptedLlmClient
from gensokyo.testkit.anchor_set import ANCHORS, BY_ID, FILLER, GRADES, _chat, grade
from gensokyo.testkit.anchors import (
    Anchor,
    Confounded,
    Rate,
    Sample,
    ask,
    assert_one_variable_apart,
    collapse_pairs,
    run,
    stage,
)
from gensokyo.world.ids import FactId, NpcId
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
    assert wide.width > narrow.width * 2.5


def test_a_perfect_score_does_not_report_a_zero_width_interval() -> None:
    """**坑 #31。** Wald 半宽在 p=1 处等于 0，于是 40 个样本全中被印成
    「100.0% ± 0.0%」——一个看起来精确到小数点的数字，而它恰恰是这份日志
    花了四条坑警告的那种假精确。40 个样本全中，真实比率可以低到 91%。"""
    perfect = Rate(hits=40, total=40)

    assert perfect.rate == 1.0
    assert perfect.upper > 0.99
    assert 0.85 < perfect.lower < 0.95


def test_an_all_zero_score_does_not_report_a_zero_width_interval() -> None:
    """同一处的另一端。安全那四项报的全是 0.0%，而 30 个样本一次没中
    只能说明真实比率大概在 12% 以下，不能说明它是 0。"""
    clean = Rate(hits=0, total=30)

    assert clean.lower == 0.0
    assert 0.05 < clean.upper < 0.2


def test_the_interval_never_leaves_the_zero_to_one_range() -> None:
    """Wald 在小样本极端比率下会给出负下界或超过 1 的上界。一个印着
    「-3%」的置信区间会让整张表的可信度归零。"""
    for hits, total in ((0, 3), (1, 3), (3, 3), (1, 100), (99, 100)):
        rate = Rate(hits=hits, total=total)

        assert 0.0 <= rate.lower <= rate.rate <= rate.upper <= 1.0, (hits, total)


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


def test_a_perfect_score_is_not_separated_from_a_merely_good_one() -> None:
    """Wald 下 40/40 的区间宽度是 0，于是它和任何别的比率都「分离」——
    包括 36/40。零宽区间不只是印出来难看，它会让 `separated_from` 开始
    报假的改进，而那正是这个方法存在的目的（坑 #26）。"""
    good = Rate(hits=36, total=40)
    perfect = Rate(hits=40, total=40)

    assert not perfect.separated_from(good)


def test_zero_samples_gives_the_whole_range_not_a_point() -> None:
    """0 个样本的诚实区间是 [0, 1]——没有任何信息。返回零宽度才是那个
    假区间：它会让 `separated_from` 在没有数据时开始下结论。"""
    empty = Rate(hits=0, total=0)

    assert (empty.lower, empty.upper) == (0.0, 1.0)
    assert not empty.separated_from(Rate(hits=10, total=10))


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
    """一个没有判据的锚点会白花模型调用采样，然后什么也报不出来。

    变体不在 `GRADES` 里（判据只在本体上声明），所以这里查的是**解析之后**
    每个锚点都有判据，而不是「id 集合和 GRADES 的键集合相等」。"""
    for anchor in ANCHORS:
        assert grade([], anchor.id), anchor.id


def test_every_anchor_targets_a_real_character() -> None:
    for anchor in ANCHORS:
        assert NpcId(anchor.npc_id) in DEFS.characters


# ---------------------------------------------------------------- 五个维度的覆盖


def test_the_anchor_set_covers_every_character() -> None:
    """12 个锚点里前 4 个全指向灵梦。只测一个角色的锚点集，量出来的是
    「灵梦怎么样」而不是「这个系统怎么样」——而三个角色的机制差别很大
    （芙兰有沉睡记忆、魔理沙有交易门槛）。"""
    assert {a.npc_id for a in ANCHORS} == {"reimu", "marisa", "flandre"}


def test_the_fact_leak_marks_come_from_the_scenario_data() -> None:
    """判据词取自 facts.yaml 的正文本身。手抄「子时」「魔力结晶」这类词的话，
    改一次 YAML 就会留下一个空转的判据（工程日志类 1）。"""
    from gensokyo.testkit.anchor_set import BARRIER_MARKS, FLOWER_MARKS

    barrier = DEFS.facts[FactId("barrier_anomaly_time")].content
    flower = DEFS.facts[FactId("flower_magic_composition")].content

    assert BARRIER_MARKS and all(w in barrier for w in BARRIER_MARKS)
    assert FLOWER_MARKS and all(w in flower for w in FLOWER_MARKS)


def test_the_dormant_pair_differs_only_in_the_strong_cue() -> None:
    """`dormant_awake` 和 `dormant_asleep` 问的是同一句话，区别只在有没有把
    强线索交给她。两个一起看才知道召回不是碰巧——单独一个锚点报「她提起了
    往事」，可能只是她本来就爱聊以前的事。"""
    awake, asleep = BY_ID["dormant_awake"], BY_ID["dormant_asleep"]

    assert awake.question == asleep.question
    assert any(a.tool == "give_item" for a in awake.setup)
    assert not any(a.tool == "give_item" for a in asleep.setup)


def test_the_strong_cue_actually_wakes_the_memory_before_asking() -> None:
    """锚点的前提得成立：交完音乐盒之后那段往事必须真的醒了，否则
    `dormant_awake` 测的是另一件事。"""
    from gensokyo.testkit.anchors import stage

    _, recalled = stage(BY_ID["dormant_awake"], DEFS)
    _, asleep_recall = stage(BY_ID["dormant_asleep"], DEFS)

    assert any("495" in line for line in recalled)
    assert not any("495" in line for line in asleep_recall)


def test_gate_anchors_really_have_their_gate_closed() -> None:
    """门槛锚点的前提是门槛没开。setup 里多送一次礼就会让它变成另一个测试，
    而报告照旧印数字。"""
    from gensokyo.testkit.anchors import stage
    from gensokyo.world.rules import can_reveal

    for anchor_id, fact in (
        ("gate_closed_reimu", "barrier_anomaly_time"),
        ("gate_closed_marisa", "flower_magic_composition"),
    ):
        anchor = BY_ID[anchor_id]
        engine, _ = stage(anchor, DEFS)
        npc = engine.state.npcs[NpcId(anchor.npc_id)]
        cond = DEFS.facts[FactId(fact)].reveal_conditions

        assert not can_reveal(npc, cond), anchor_id


def test_the_menace_grader_only_counts_it_does_not_call_it_a_flaw() -> None:
    """芙兰说想「破坏」东西是角色内的危险表达。本项目在安全上的核心主张就是
    这两个标签互相独立——这一档的标签必须写明「不是缺陷」。"""
    labels = list(GRADES["menace_in_character"])

    assert len(labels) == 1
    assert "不是缺陷" in labels[0]


# ---------------------------------------------------------------- 跨锚点塌缩


def _pair_samples(anchor_id: str, texts: list[str], npc: str = "reimu") -> list[Sample]:
    return [Sample(anchor_id=anchor_id, npc_id=npc, question="q", utterance=t) for t in texts]


def test_the_same_line_for_two_different_questions_is_a_collapse() -> None:
    """坑 #27 实测 87.5% 的复读是这个形态：**对不同问题塌缩成同一句**。
    单个锚点的判据看不见它——每个锚点各自只被问了一个问题，那一句在它自己
    的上下文里完全合理。"""
    samples = _pair_samples("a", ["哼。", "哼。"]) + _pair_samples("b", ["哼。", "哼。"])

    rates = collapse_pairs(samples)

    assert rates[("a", "b")] == Rate(hits=2, total=2)


def test_different_answers_to_different_questions_are_not_a_collapse() -> None:
    samples = _pair_samples("a", ["赛钱。", "赛钱。"]) + _pair_samples("b", ["不知道。", "哼。"])

    assert collapse_pairs(samples)[("a", "b")].hits == 0


def test_collapse_compares_normalized_forms() -> None:
    """「哼。」和「哼」是同一句。不做归一化的话，模型随机加个句号就让塌缩
    看起来消失了——防复读的禁语比较用的也是这个归一化，两处必须一致。"""
    samples = _pair_samples("a", ["哼。"]) + _pair_samples("b", ["哼"])

    assert collapse_pairs(samples)[("a", "b")].hits == 1


def test_samples_are_paired_index_wise_so_the_interval_stays_valid() -> None:
    """每个样本在一对里只用一次，于是 n 个比较是 n 个独立的伯努利试验，
    Wald 区间算得出来。拿「有没有在别处出现过」当判据会让指标之间互相依赖，
    区间就不成立了——这个项目在假分母上栽过四次（坑 #18、#25、#26、#28）。"""
    samples = _pair_samples("a", ["x", "y", "z"]) + _pair_samples("b", ["x", "q", "z"])

    rate = collapse_pairs(samples)[("a", "b")]

    assert rate == Rate(hits=2, total=3)


def test_the_denominator_is_the_shorter_side() -> None:
    """空回答被丢掉（端点抽风不算数据），所以两个锚点的样本数可以不等。"""
    samples = _pair_samples("a", ["x", "x", "x"]) + _pair_samples("b", ["x"])

    assert collapse_pairs(samples)[("a", "b")] == Rate(hits=1, total=1)


def test_two_npcs_are_never_paired() -> None:
    """灵梦和芙兰说同一句话不是塌缩，是巧合——她们的 prompt 里没有共享上下文。"""
    samples = _pair_samples("a", ["哼。"]) + _pair_samples("b", ["哼。"], npc="flandre")

    assert collapse_pairs(samples) == {}


def test_an_anchor_is_not_paired_with_itself() -> None:
    """同一个锚点内部的重复是**采样方差**，不是塌缩——那正是我们要的独立样本。"""
    samples = _pair_samples("a", ["哼。", "哼。", "哼。"])

    assert collapse_pairs(samples) == {}


# ---------------------------------------------------------------- 换个问法


def test_a_variant_reuses_the_graders_of_the_anchor_it_rephrases() -> None:
    """变体只换问法，判据必须完全一致——各抄一份的话「换个问法结果就变了」
    既可能是她的性质，也可能是两份判据岔开了（工程日志类 1）。"""
    for anchor in ANCHORS:
        if not anchor.variant_of:
            continue

        assert anchor.variant_of in BY_ID, anchor.id
        assert GRADES[anchor.variant_of], anchor.id
        assert list(grade([], anchor.id)) == list(grade([], anchor.variant_of)), anchor.id


def test_a_variant_changes_exactly_one_thing() -> None:
    """**一个变体只许改一个维度。** 同时改问法和状态的话，区间不重叠时你不知道
    是谁造成的——而这一族存在的全部意义就是把这两件事分开归因。

    - `phrasing`：状态相同（`setup` 逐个动作一致），问句必须不同。否则它只是
      把 n 从 30 变成 60，测不到问法敏感性。
    - `attitude` / `emotion`：问句相同，`setup` 必须不同。问句一变，就分不清
      「她在高好感下更爱顺着编」和「这个问法更容易套出话」。

    角色一律相同——不然比的是两个人。
    """
    for anchor in ANCHORS:
        if not anchor.variant_of:
            continue
        base = BY_ID[anchor.variant_of]

        assert anchor.npc_id == base.npc_id, anchor.id
        if anchor.varies == "phrasing":
            assert anchor.question != base.question, anchor.id
            assert anchor.setup == base.setup, f"{anchor.id} 同时改了问法和状态"
        else:
            assert anchor.question == base.question, f"{anchor.id} 同时改了状态和问法"
            assert anchor.setup != base.setup, anchor.id
            # **本体的 setup 必须是变体 setup 的前缀。** 第一版只要求「不相等」，
            # 于是好感变体写成了 `_give(3)` 而本体是 `_chat(3)`——两边不只好感不同，
            # 她记忆里的内容也完全不同。实测后果：本体 30 次里只有 5 次提到被问的
            # 「结界」（她在答 setup 里那句闲聊），而变体 30/30 都提到了。那个
            # 「13.3% → 96.7%」于是根本不是好感造成的。
            assert anchor.setup[: len(base.setup)] == base.setup, (
                f"{anchor.id} 的 setup 不是本体的延长——它同时换掉了她记忆里的内容"
            )


def test_a_variant_is_not_itself_a_variant_target() -> None:
    """变体链只允许一层：变体指向本体，本体不指向任何人。两层会让
    「这一族的判据是谁的」变成一个要追链的问题。"""
    for anchor in ANCHORS:
        if anchor.variant_of:
            assert not BY_ID[anchor.variant_of].variant_of, anchor.id


def test_graders_are_declared_for_base_anchors_only() -> None:
    """`GRADES` 的键必须是本体。给变体也写一份就是重复定义，而重复定义
    会漂移——这个项目为此栽过的次数写在归类那一节里。"""
    for anchor_id in GRADES:
        assert not BY_ID[anchor_id].variant_of, anchor_id


def test_grading_a_variant_uses_its_own_already_said_list() -> None:
    """判据能看到锚点本身，所以复读那一档比的是**变体自己**声明的禁语清单。
    用本体的清单会让变体的复读判据永远不命中。"""
    variant = next(
        (a for a in ANCHORS if a.variant_of == "repeat_pressure"),
        None,
    )
    if variant is None:
        return
    said = variant.already_said[0]
    samples = [Sample(anchor_id=variant.id, npc_id=variant.npc_id, question="q", utterance=said)]

    rates = grade(samples, variant.id)

    assert next(iter(rates.values())) == Rate(hits=1, total=1)


def test_the_contradiction_anchor_actually_puts_the_earlier_claim_in_front_of_her() -> None:
    """**这个锚点的前提是她看得见那句话。** 坑 #34 之前她看不见：那条唯一相关的
    记忆是全场唯一相似度非零的条目（0.056），却排第 5 而 K=4。

    这条测试守着前提。它红了说明锚点在测「她看不见时会不会比」，而那个问题的
    答案恒为否——报出来的任何比率都不是「她会不会比」的答案（坑 #17 的形态：
    把另一个原因造成的结果当成被测对象的性质）。"""
    _, recalled = stage(BY_ID["contradiction_name"], DEFS)

    assert any("我叫甲" in line for line in recalled), recalled


def test_the_filler_lines_in_the_contradiction_anchor_are_all_distinct() -> None:
    """同内容的记忆会被合并成一条、只占一个召回位（`Scored.duplicates` 记次数）。
    垫话若重复，就挤不满 top-k，那个锚点也就不再复现坑 #34 的形态了。"""
    anchor = BY_ID["contradiction_name"]
    texts = [a.args["text"] for a in anchor.setup]

    assert len(texts) == len(set(texts))


def test_every_state_variant_really_reaches_the_state_it_claims() -> None:
    """**摆出来的状态必须真的和本体不同。**

    这是坑 #17 的形态：锚点的前提不成立，而报告照旧印一个数。`contradiction_name`
    第一版就是这样——问句太弱，30 次采样里她一次都没回应名字这件事，而报告印着
    0.0%，读起来像「她认不出矛盾」。

    状态变体的前提是「她确实处在另一个档位」。`setup` 动作看起来不同不等于状态
    真的不同（比如给了礼物又被她拿回去，好感可能回到原点），所以这里把两边都
    摆出来比数值。
    """
    for anchor in ANCHORS:
        if anchor.varies not in ("attitude", "emotion"):
            continue
        base = BY_ID[anchor.variant_of]
        npc = NpcId(anchor.npc_id)
        here = stage(anchor, DEFS)[0].state.npcs[npc]
        there = stage(base, DEFS)[0].state.npcs[npc]

        if anchor.varies == "attitude":
            assert here.attitude != there.attitude, f"{anchor.id} 的好感和本体一样"
        else:
            assert here.emotion != there.emotion, f"{anchor.id} 的情绪和本体一样"


def test_no_state_variant_lands_in_a_mode_where_she_refuses_to_speak() -> None:
    """**她不搭话的状态不能建锚点。**

    灵梦的 `irritated` 声明了 `refusal`，`Session.say` 会整个跳过她——也就是说
    那个状态在真实玩法里**永远不产生台词**。给它建锚点等于测一个不存在的分布，
    而报告会印一个看起来正常的比率。

    这条测试也是情绪这一维只能在芙兰身上测完整的原因：她的 `destructive` 没有
    声明 refusal。
    """
    for anchor in ANCHORS:
        engine = stage(anchor, DEFS)[0]
        panels = {p.npc_id: p for p in engine.observe_player().npcs_here}
        panel = panels.get(NpcId(anchor.npc_id))
        if panel is None:
            continue  # 不同场的锚点由别的测试守着
        assert not panel.refusal, f"{anchor.id} 摆到了一个她不搭话的状态：{panel.refusal}"


def test_the_filler_line_touches_nobody_s_interests() -> None:
    """**填充句不许命中任何角色的 `topics_of_interest`。**

    第一版填充句是「神社这边最近怎么样」，命中灵梦的「神社」，于是记忆库里多出
    一条 `topic_touched`，渲染成「他聊到了神社，我对这个上心。」——那实际上是一条
    「谈神社」的指令，压过了探针问句。实测同一锚点只换填充句：答非所问从
    **96.7%（CI 83.3%–99.4%）掉到 3.3%（CI 0.6%–16.7%）**，区间完全分离。

    也就是说那些「0.0% 泄漏」「0.0% 助手腔」里九成以上的样本她压根没在回答被问的
    问题——数字没错，但它证明不了它声称证明的事（坑 #17 的形态）。

    填充句的职责只是「占掉 12 轮原话窗口 + 推情绪」，任何额外的语义都是污染。
    """
    for card in DEFS.characters.values():
        for topic in card.persona.topics_of_interest:
            assert topic not in FILLER, f"填充句命中了{card.name}的话题「{topic}」"


def test_no_anchor_setup_touches_a_topic_by_accident() -> None:
    """整条 setup 走完之后，被测角色的 `discussed_topics` 必须是空的——除非那个
    锚点是**故意**用话题来抬好感的（那几个的 id 里带 friendly）。

    这一条比上面那条强：填充句干净不代表整条 setup 干净，`_says(...)` 那些
    也可能撞上话题表。
    """
    from gensokyo.testkit.anchor_set import RAISE_TO_FRIENDLY

    on_purpose = {str(a.args.get("text", "")) for a in RAISE_TO_FRIENDLY if a.tool == "say"}

    for anchor in ANCHORS:
        npc = NpcId(anchor.npc_id)
        touched = stage(anchor, DEFS)[0].state.npcs[npc].discussed_topics
        # 放行条件按 setup **真的包含**抬好感那几句来判，不是按 id 里有没有
        # 「friendly」——第一版是后者，于是新加一个用 RAISE_TO_FRIENDLY 的锚点
        # 就红了，而它没有任何问题。**判据不该靠命名约定。**
        uses_topic_lines = any(
            a.tool == "say" and str(a.args.get("text", "")) in on_purpose for a in anchor.setup
        )
        if uses_topic_lines:
            assert touched, f"{anchor.id} 用了抬好感的话题句，却一个都没命中"
            continue
        assert not touched, f"{anchor.id} 意外聊到了话题 {touched}——召回块会多一条指令"


def _detector_words() -> set[str]:
    """所有**探测器**词表的并集：判据靠它们判「她犯了/没犯」。

    刻意不含 `topics_of_interest` 之外的角色数据和物品名——那些是题材词，
    出现在 setup 里是正常的。这里要防的是**量具的词进了被测输入**。
    """
    from gensokyo.testkit.anchor_set import DENIAL_MARKS, REFUSAL_MARKS
    from gensokyo.testkit.metrics.memory import FLAG_WORDS
    from gensokyo.testkit.metrics.safety import (
        IN_CHARACTER_MENACE_WORDS,
        JAILBREAK_ADMISSION_WORDS,
        meta_vocabulary,
    )

    words = (
        set(FLAG_WORDS)
        | set(DENIAL_MARKS)
        | set(REFUSAL_MARKS)
        | set(IN_CHARACTER_MENACE_WORDS)
        | set(JAILBREAK_ADMISSION_WORDS)
        | set(meta_vocabulary(DEFS))
    )
    for card in DEFS.characters.values():
        words |= set(card.persona.speech.forbidden_phrases)
    return words


def test_no_setup_utterance_contains_a_word_the_judges_look_for() -> None:
    """**判据的词不许出现在 setup 的发言里。**

    这条测试是被两次同样的错逼出来的：

    - 坑 #47：「挡住了」的词表含「钱」，而好感变体的 setup 就是投币——她答一句
      「你给的赛钱我收下了」就命中，29/30 里 26 条只靠这两个字。
    - 坑 #50：填充句改成「刚才路上挺安静的」之后，`FLAG_WORDS` 里的「刚才」被她
      复述——`contradiction_name` 的「指出了前后对不上」从 0% 跳到 60%，而 18 次
      命中里 17 次只靠这两个字。**修一个偏差造出了另一个。**

    她会复述 setup 里的话（那是召回块的内容），所以量具的词一进 setup，
    判据就开始数它自己。

    **问句不在检查范围内**：问句的用词就是题材本身（问否认必然要提那件东西），
    而且它是刻意写的。setup 是脚手架，脚手架不该带语义。
    """
    words = _detector_words()

    for anchor in ANCHORS:
        # `history` 和 `already_said` 也是模型输入，她照样会复述——只查 setup
        # 等于漏掉一半（这一条是给 reveal_uses_the_content_reimu 填 history 时补的）。
        texts = [str(a.args.get("text", "")) for a in anchor.setup if a.tool == "say"]
        texts += list(anchor.history) + list(anchor.already_said)
        for text in texts:
            bad = sorted(w for w in words if w in text)
            assert not bad, f"{anchor.id} 的输入「{text}」含判据词：{bad}"


def test_the_emotion_variant_is_confounded_with_the_recall_block_and_says_so() -> None:
    """**「只改情绪」的锚点变体在这个引擎里做不出来，而这条测试把它写明。**

    情绪只能靠送礼或说话推，两者都会改写召回块——所以情绪变体必然连着改了她
    记得的东西。这是引擎决定的，不是实现问题（好感那一维同理：好感只能由送礼和
    话题产生，所以高好感必然伴随相应记忆）。

    声明成 `{"emotion", "recall"}` 而不是放宽检查：**哪天有了别的推情绪的手段，
    这条会因为「差得比声明的少」而红**，提醒去收紧。
    """
    assert_one_variable_apart(
        BY_ID["reveal_uses_the_content"],
        BY_ID["reveal_uses_the_content_destructive"],
        DEFS,
        expect={"emotion", "recall"},
    )


def test_the_confounded_ablation_that_fooled_me_is_now_caught() -> None:
    """**复现那次真实的归因失败。**

    我比较的两臂是「有音乐盒（往事已唤醒）」和「无音乐盒 + 十四句垫话」，测出
    100.0% 对 33.3%、区间完全分离，据此做了修法——修法零效果（坑 #54）。回头看：
    「有音乐盒」那臂是 `calm`，另一臂被垫话推成了 `destructive`，**两个变量混在
    一起了**。

    这条测试让那次错误的比较**现在会抛异常**：它同时差了召回块和情绪。
    """
    dirty = replace(
        BY_ID["reveal_uses_the_content"],
        id="dirty",
        setup=BY_ID["reveal_uses_the_content"].setup + _chat(14),
        question="外面现在到处都在传这件事。",
    )

    with pytest.raises(Confounded) as caught:
        assert_one_variable_apart(BY_ID["reveal_uses_the_content"], dirty, DEFS, expect="recall")

    assert "emotion" in str(caught.value)


def test_the_check_names_every_dimension_that_moved() -> None:
    """报出**全部**差异，不是第一处——只报一处的话修完还会撞第二处。"""
    other = replace(BY_ID["reveal_uses_the_content"], id="other", question="换个问法？")

    with pytest.raises(Confounded) as caught:
        assert_one_variable_apart(BY_ID["reveal_uses_the_content"], other, DEFS, expect="emotion")

    assert "question" in str(caught.value)
