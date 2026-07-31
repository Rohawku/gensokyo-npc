import math

from factories import bad_result, call, defs, episode, ok_result, say_turn

from gensokyo.testkit.metrics.persona import (
    OUT_OF_BOUNDS_WORDS,
    expected_baselines,
    js_divergence,
    persona_library_sizes,
    persona_metrics,
)

DEFS = defs()


# ---------------------------------------------------------------- JS 散度


def test_identical_distributions_have_zero_divergence() -> None:
    p = {"ask_player": 0.3, "take_item": 0.1, "give_item": 0.05}

    assert js_divergence(p, dict(p)) == 0.0


def test_identical_shapes_at_different_scales_have_zero_divergence() -> None:
    """基线写的是频率、实际是计数，两侧量纲不同。不各自归一化的话，
    一个完全符合基线的角色只要多调几次工具就会「偏离」。"""
    expected = {"ask_player": 0.3, "take_item": 0.1}
    observed = {"ask_player": 30.0, "take_item": 10.0}

    assert js_divergence(expected, observed) == 0.0


def test_fully_disjoint_supports_have_divergence_one() -> None:
    """底数取 2 就是为了让这个上界正好是 1。"""
    assert js_divergence({"ask_player": 1.0}, {"break_item": 1.0}) == 1.0


def test_a_tool_missing_from_one_side_is_filled_with_zero_not_dropped() -> None:
    """基线里有 break_item 而实际一次没调，正是最该算成偏离的情况。
    把它当成「这一维不存在」会把偏离整整抹掉——那样这里会是 0.0。

    手算：p=(.5,.5)、q=(1,0)、m=(.75,.25)，
    JSD = ½·[½log₂(.5/.75)+½log₂(.5/.25)] + ½·[log₂(1/.75)] ≈ 0.3113。
    """
    expected = {"ask_player": 0.5, "break_item": 0.5}
    observed = {"ask_player": 1.0}

    assert math.isclose(js_divergence(expected, observed), 0.31127812445913283)


def test_divergence_is_symmetric_and_bounded() -> None:
    p = {"a": 0.7, "b": 0.2, "c": 0.1}
    q = {"a": 0.1, "b": 0.1, "d": 0.8}

    forward = js_divergence(p, q)
    backward = js_divergence(q, p)

    assert math.isclose(forward, backward)
    assert 0.0 < forward < 1.0


def test_two_empty_distributions_are_not_a_divergence() -> None:
    assert js_divergence({}, {}) == 0.0
    assert js_divergence({"a": 1.0}, {}) == 1.0
    assert js_divergence({}, {"a": 1.0}) == 1.0


# ---------------------------------------------------------------- 三种极端


def test_empty_batch_is_all_zeros() -> None:
    metrics = persona_metrics([], DEFS)

    assert metrics.utterances == 0
    assert metrics.assistant_tone_rate == 0.0
    assert metrics.assistant_tone_hits == {}
    assert metrics.behavior_divergence == {}
    assert metrics.behavior_observed == {}
    assert metrics.out_of_bounds_rate == 0.0
    assert metrics.repetition_rate == 0.0


def test_an_in_character_batch_is_clean_on_every_metric() -> None:
    batch = [
        episode(
            [
                say_turn("reimu", "赛钱箱空着呢，你说这算怎么回事。", tick=1),
                say_turn("marisa", "情报是要换的，就是这样。", tick=2),
                say_turn("flandre", "你陪我玩嘛！", tick=3),
            ]
        )
    ]

    metrics = persona_metrics(batch, DEFS)

    assert metrics.utterances == 3
    assert metrics.assistant_tone_rate == 0.0
    assert metrics.out_of_bounds_rate == 0.0
    assert metrics.repetition_rate == 0.0


def test_a_fully_polluted_batch() -> None:
    batch = [
        episode(
            [
                say_turn("reimu", "我很乐意帮您查这件事。", tick=1),
                say_turn("marisa", "有什么可以帮您的吗？", tick=2),
            ]
        )
    ]

    metrics = persona_metrics(batch, DEFS)

    assert metrics.assistant_tone_rate == 1.0


# ------------------------------------------------------------ 助手腔（用角色卡）


def test_assistant_tone_uses_the_phrases_from_the_character_card() -> None:
    """词库直接取角色卡的 forbidden_phrases——一处定义两处使用：同一份清单
    既进系统提示约束生成，又在这里用于检测。两份各写一份的话，「污染率下降」
    可能只是因为检测那份忘了跟着更新。"""
    phrase = "我很乐意"
    assert phrase in DEFS.characters["reimu"].persona.speech.forbidden_phrases

    metrics = persona_metrics([episode([say_turn("reimu", f"{phrase}为您查一下结界的事。")])], DEFS)

    assert metrics.assistant_tone_rate == 1.0
    assert metrics.assistant_tone_hits[phrase] == 1


def test_a_phrase_only_reimu_forbids_is_not_counted_against_marisa() -> None:
    """禁语清单是逐角色的：灵梦多一条「请问还有什么需要」，芙兰和魔理沙没有。
    统一用一份全局清单的话，角色卡就不再是唯一的差异来源了。"""
    only_reimu = "请问还有什么需要"
    assert only_reimu in DEFS.characters["reimu"].persona.speech.forbidden_phrases
    assert only_reimu not in DEFS.characters["marisa"].persona.speech.forbidden_phrases

    batch = [
        episode(
            [
                say_turn("reimu", f"{only_reimu}？", tick=1),
                say_turn("marisa", f"{only_reimu}？", tick=2),
            ]
        )
    ]

    metrics = persona_metrics(batch, DEFS)

    assert metrics.assistant_tone_rate == 0.5
    assert metrics.assistant_tone_hits == {only_reimu: 1}


# ---------------------------------------------------------------- 行为分布


def test_behaviour_matching_the_card_baseline_has_low_divergence() -> None:
    """魔理沙的基线是 take_item 最高（0.30）。照着基线的形状调工具，
    散度就该接近 0。"""
    expected = expected_baselines(DEFS)["marisa"]
    counts = {tool: round(freq * 100) for tool, freq in expected.items()}
    turns = [
        say_turn(
            "marisa",
            "就是这样。",
            tick=i,
            tool_calls=[call(tool)],
            tool_results=[ok_result()],
        )
        for i, (tool, n) in enumerate(counts.items())
        for _ in range(n)
    ]

    metrics = persona_metrics([episode(turns)], DEFS)

    assert metrics.behavior_divergence["marisa"] < 0.01


def test_behaviour_off_the_baseline_shows_up_as_divergence() -> None:
    """整局只会问问题的魔理沙：她的基线里 take_item 是全场最高的 0.30，
    一次不伸手就该被这个数抓住。"""
    turns = [
        say_turn(
            "marisa",
            "你说说看。",
            tick=i,
            tool_calls=[call("ask_player", question="嗯？")],
            tool_results=[ok_result()],
        )
        for i in range(5)
    ]

    metrics = persona_metrics([episode(turns)], DEFS)

    assert metrics.behavior_divergence["marisa"] > 0.4
    assert metrics.behavior_observed["marisa"] == {"ask_player": 1.0}


def test_an_npc_with_no_tool_calls_gets_no_divergence_number() -> None:
    """一次工具都没调不是「完全符合基线」（0.0）也不是「完全跑偏」（1.0），
    是没有数据。给个数字就是在没有依据时下结论；报告里那一格印「—」。"""
    metrics = persona_metrics([episode([say_turn("marisa", "就是这样。")])], DEFS)

    assert "marisa" not in metrics.behavior_divergence
    assert metrics.behavior_observed["marisa"] == {}


def test_failed_tool_calls_still_count_toward_the_observed_distribution() -> None:
    """行为一致性问的是「她想做什么」。只统计成功的调用，会让被情绪 gate
    或前置条件拦下的意图凭空消失——而那正是最该看到的行为。"""
    turns = [
        say_turn(
            "flandre",
            "我想弄坏它！",
            tick=i,
            tool_calls=[call("break_item", item="old_music_box")],
            tool_results=[bad_result("tool_denied")],
        )
        for i in range(3)
    ]

    metrics = persona_metrics([episode(turns)], DEFS)

    assert metrics.behavior_observed["flandre"] == {"break_item": 1.0}


# ---------------------------------------------------------------- 越界知识


def test_modern_technology_words_are_out_of_bounds() -> None:
    """这一项用的是独立词表，不是角色卡的 forbidden_knowledge——后者是中文
    散文（「外界的科技」），拿它做子串匹配等于问她有没有把这五个字念出来。"""
    batch = [
        episode(
            [
                say_turn("reimu", "你说的那个手机，我没见过。", tick=1),
                say_turn("marisa", "情报是要换的。", tick=2),
            ]
        )
    ]

    assert persona_metrics(batch, DEFS).out_of_bounds_rate == 0.5


def test_the_out_of_bounds_list_is_not_the_character_card_prose() -> None:
    """这处不一致是刻意的，也必须是可见的：角色卡里那句散文本身
    不该命中任何东西。"""
    prose = DEFS.characters["flandre"].knowledge.forbidden_knowledge
    assert "外界的科技" in prose
    assert "外界的科技" not in OUT_OF_BOUNDS_WORDS


# ---------------------------------------------------------------- 复读


def test_repeating_the_same_line_is_measured() -> None:
    """工程日志坑 #2：她说过一次「你管的太多了」，这句话进了历史，模型看到
    这个模式就继续敷衍，reveal_info 命中率从 3/5 掉到 1/5。复读是自我强化的，
    而且它同时压低工具调用率——这个数是那件事的量化。"""
    line = "你管的太多了。"
    turns = [say_turn("reimu", line, tick=i) for i in range(4)]

    metrics = persona_metrics([episode(turns)], DEFS)

    assert metrics.repetition_rate == 0.75


def test_different_lines_are_not_repetition() -> None:
    turns = [say_turn("reimu", f"第{i}句不一样。", tick=i) for i in range(4)]

    assert persona_metrics([episode(turns)], DEFS).repetition_rate == 0.0


def test_the_same_line_from_two_different_npcs_is_not_repetition() -> None:
    """复读是「同一个人在同一局里反复说同一句」。跨角色撞句是两条独立
    历史的巧合，算进去会让同场的两个 NPC 互相拉高对方的复读率。"""
    turns = [say_turn("reimu", "……哼。", tick=1), say_turn("marisa", "……哼。", tick=1)]

    assert persona_metrics([episode(turns)], DEFS).repetition_rate == 0.0


def test_the_same_line_in_two_different_episodes_is_not_repetition() -> None:
    """两局之间没有共享历史，说同一句话是采样巧合而不是复读。"""
    batch = [
        episode([say_turn("reimu", "有事说事。")], seed=0),
        episode([say_turn("reimu", "有事说事。")], seed=1),
    ]

    assert persona_metrics(batch, DEFS).repetition_rate == 0.0


def test_library_size_is_reported() -> None:
    assert persona_library_sizes()["out_of_bounds"] == len(OUT_OF_BOUNDS_WORDS)


def test_plot_tools_do_not_inflate_behaviour_divergence() -> None:
    """行为偏离度衡量性格表达，不该被剧情进度污染。

    实测未排除时芙兰散度虚高到 1.000——因为她的基线只列了 4 个性格性工具，
    而 reveal_info 必然出现在实际分布里，于是「走了剧情」被读成「人设崩了」。
    """
    baseline = DEFS.characters["flandre"].behavior_baseline["tool_frequency"]
    on_baseline = max(baseline, key=lambda k: baseline[k])

    def traj(tools: list[str]) -> object:
        return episode(
            turns=[
                say_turn(
                    npc_id="flandre",
                    tool_calls=[call(t) for t in tools],
                    tool_results=[ok_result() for _ in tools],
                )
            ]
        )

    clean = persona_metrics([traj([on_baseline] * 8)], DEFS).behavior_divergence["flandre"]
    polluted = persona_metrics(
        [traj([on_baseline] * 8 + ["reveal_info"] * 8)], DEFS
    ).behavior_divergence["flandre"]

    assert polluted == clean, "剧情工具不该改变行为偏离度"


def test_punctuation_only_variants_count_as_the_same_line() -> None:
    """第一份基线里「你到底想干啥？」19 次、「你到底想干啥。」12 次被算成
    两句不同的话，于是测出来的复读率**低于**真实值。口径改成标准化比较后
    那组 43.1% / 56.7% 的数字不可与之后的数字直接比较。"""
    turns = [
        say_turn("reimu", "你到底想干啥？", tick=0),
        say_turn("reimu", "你到底想干啥。", tick=1),
        say_turn("reimu", "你到底想干啥", tick=2),
        say_turn("reimu", "你 到底想干啥？", tick=3),
    ]

    assert persona_metrics([episode(turns)], DEFS).repetition_rate == 0.75
