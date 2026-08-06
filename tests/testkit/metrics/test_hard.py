from factories import (
    CLUES,
    bad_result,
    call,
    cleared_episode,
    command_turn,
    defs,
    episode,
    forgotten_episode,
    ok_result,
    say_turn,
)

from gensokyo.testkit.metrics.hard import (
    failure_turn_count,
    reveal_delivery_metrics,
    task_metrics,
    tool_metrics,
)
from gensokyo.testkit.trajectory import TurnRecord

DEFS = defs()


# ------------------------------------------------------------------ 三种极端


def test_empty_batch_yields_zeros_and_no_crash() -> None:
    """一批都没跑成时报告仍然要能生成——否则端点挂了就连「全挂了」
    这个结论都印不出来。"""
    task = task_metrics([], DEFS)
    tools = tool_metrics([])

    assert task.episodes == 0
    assert task.completion_rate == 0.0
    assert task.failure_rate == 0.0
    assert task.unfinished_rate == 0.0
    assert task.mean_turns_to_finish is None
    assert task.stage_histogram == {}
    assert task.ending_histogram == {}
    assert set(task.clue_rate) == {str(f) for f in DEFS.clue_facts()}
    assert set(task.clue_rate.values()) == {0.0}

    assert tools.total_calls == 0
    assert tools.self_heal_rate is None
    assert tools.error_code_histogram == {}


def test_all_cleared_batch() -> None:
    task = task_metrics([cleared_episode(), cleared_episode(seed=1)], DEFS)
    tools = tool_metrics([cleared_episode(), cleared_episode(seed=1)])

    assert task.completion_rate == 1.0
    assert task.failure_rate == 0.0
    assert task.unfinished_rate == 0.0
    assert task.mean_turns_to_finish == 3.0
    assert task.stage_histogram == {"S4_END": 2}
    assert task.ending_histogram == {"kirisame_burn": 2}
    assert task.clue_rate == {str(c): 1.0 for c in CLUES}

    assert tools.total_calls == 6
    assert tools.schema_valid_rate == 1.0
    assert tools.denied_rate == 0.0
    assert tools.precondition_fail_rate == 0.0
    assert tools.redundant_rate == 0.0
    assert tools.self_heal_rate is None
    assert tools.per_tool_counts == {"reveal_info": 6}


def test_all_failed_batch() -> None:
    batch = [forgotten_episode(), forgotten_episode(seed=1)]

    task = task_metrics(batch, DEFS)
    tools = tool_metrics(batch)

    assert task.completion_rate == 0.0
    assert task.failure_rate == 1.0
    assert task.unfinished_rate == 0.0
    assert task.mean_turns_to_finish is None
    assert task.ending_histogram == {"forgotten": 2}
    assert set(task.clue_rate.values()) == {0.0}

    assert tools.total_calls == 6
    assert tools.precondition_fail_rate == 1.0
    assert tools.error_code_histogram == {"reveal_condition_unmet": 6}
    # 每回合只有一次调用且它失败了，六个回合全都无从自愈。
    assert tools.self_heal_rate == 0.0
    assert failure_turn_count(batch) == 6


# ------------------------------------------------------- 完成 / 失败 / 未完成


def test_unfinished_is_not_the_same_as_failed() -> None:
    """跑到回合上限被砍断和走到失败结局是两回事：改 max_turns 只该动前者。
    合成一个「未通关率」的话，调 max_turns 会让指标动而系统没变。"""
    batch = [
        cleared_episode(),
        forgotten_episode(seed=1),
        episode([say_turn()], seed=2, finished=False, ending=None, final_stage="S2_CLUES"),
    ]

    task = task_metrics(batch, DEFS)

    assert task.completion_rate == 1 / 3
    assert task.failure_rate == 1 / 3
    assert task.unfinished_rate == 1 / 3
    assert task.stage_histogram == {"S4_END": 2, "S2_CLUES": 1}
    assert task.ending_histogram == {"kirisame_burn": 1, "forgotten": 1}


def test_a_finished_episode_with_the_timeout_ending_does_not_count_as_completion() -> None:
    task = task_metrics([episode([say_turn()], ending="forgotten")], DEFS)

    assert task.completion_rate == 0.0
    assert task.failure_rate == 1.0


def test_mean_turns_counts_only_the_cleared_episodes() -> None:
    """把没打完的局混进分母，「平均多少回合能通关」就退化成
    「平均一局多长」，而后者只反映 max_turns 设了多少。"""
    long_unfinished = episode(
        [say_turn(tick=i) for i in range(40)], finished=False, ending=None, final_stage="S1_ANOMALY"
    )

    task = task_metrics([cleared_episode(), long_unfinished], DEFS)

    assert task.mean_turns_to_finish == 3.0


def test_clue_rate_counts_a_clue_that_was_obtained_then_forgotten() -> None:
    """无缘塚的花会吸走已到手的线索。只看结局那一帧会把「拿到过又丢了」
    记成「从没拿到」——那是两个完全不同的问题。"""
    traj = episode(
        [
            say_turn(tick=1, known_fact_ids=[CLUES[0]]),
            say_turn(tick=2, known_fact_ids=[]),
        ],
        ending="forgotten",
    )

    task = task_metrics([traj], DEFS)

    assert task.clue_rate[CLUES[0]] == 1.0
    assert task.clue_rate[CLUES[1]] == 0.0


def test_clue_rate_lists_every_clue_in_the_scenario() -> None:
    """一条谁都没拿到的线索必须以 0.0 出现。缺项和 0 在读报告时
    长得完全不一样——前者会被当成「这个指标没测」。"""
    task = task_metrics([episode([say_turn(known_fact_ids=[CLUES[0]])])], DEFS)

    assert set(task.clue_rate) == {str(f) for f in DEFS.clue_facts()}
    assert task.clue_rate[CLUES[2]] == 0.0


# ---------------------------------------------------------------- 工具调用


def test_error_codes_are_bucketed_by_kind() -> None:
    """bad_args（模型不会填参数）、tool_denied（人设/情绪 gate 拦住了）、
    前置条件不满足（世界规则在生效）是三类完全不同的事，不能合成一个失败率。"""
    traj = episode(
        [
            say_turn(
                tool_calls=[
                    call("reveal_info", fact="barrier_anomaly_time"),
                    call("move", to="muenzuka"),
                    call("give_item"),
                    call("ask_player", question="你要干嘛"),
                ],
                tool_results=[
                    bad_result("reveal_condition_unmet"),
                    bad_result("tool_denied"),
                    bad_result("bad_args"),
                    ok_result(),
                ],
            )
        ]
    )

    tools = tool_metrics([traj])

    assert tools.total_calls == 4
    assert tools.schema_valid_rate == 0.75
    assert tools.denied_rate == 0.25
    assert tools.precondition_fail_rate == 0.25
    assert tools.error_code_histogram == {
        "reveal_condition_unmet": 1,
        "tool_denied": 1,
        "bad_args": 1,
    }
    assert tools.per_tool_counts == {
        "reveal_info": 1,
        "move": 1,
        "give_item": 1,
        "ask_player": 1,
    }


def test_commands_are_not_tool_calls() -> None:
    """/go /give 是玩家的指令，不是 NPC 的工具调用。混进来会让
    工具调用量随玩家人格的走路习惯浮动。"""
    tools = tool_metrics(
        [episode([command_turn(), command_turn(ok=False, error_code="no_such_exit")])]
    )

    assert tools.total_calls == 0
    assert tools.error_code_histogram == {}


def test_redundant_counts_only_repeats_inside_the_same_turn() -> None:
    """跨回合再问一次同一件事是正常玩法；同一回合里把同一个 (tool, args)
    发两遍才是空转。"""
    repeated = say_turn(
        tick=1,
        tool_calls=[
            call("reveal_info", fact="barrier_anomaly_time"),
            call("reveal_info", fact="barrier_anomaly_time"),
        ],
        tool_results=[bad_result("reveal_condition_unmet")] * 2,
    )
    later = say_turn(
        tick=2,
        tool_calls=[call("reveal_info", fact="barrier_anomaly_time")],
        tool_results=[ok_result()],
    )

    tools = tool_metrics([episode([repeated, later])])

    assert tools.total_calls == 3
    assert tools.redundant_rate == 1 / 3


def test_argument_order_does_not_make_two_calls_look_different() -> None:
    turn = say_turn(
        tool_calls=[
            call("give_item", item="offering_coin", count=1),
            {"tool": "give_item", "args": {"count": 1, "item": "offering_coin"}},
        ],
        tool_results=[ok_result(), ok_result()],
    )

    assert tool_metrics([episode([turn])]).redundant_rate == 0.5


# ---------------------------------------------------------------- 自愈


def test_self_heal_rate_is_none_when_no_turn_ever_failed() -> None:
    """分母为 0 时必须是 None，不是 0.0。0.0 会让「本批没有失败回合」
    和「每次自愈都失败」在报告里长得一模一样，而它们一个是好消息、
    一个是最坏的消息。"""
    tools = tool_metrics([cleared_episode()])

    assert tools.self_heal_rate is None
    assert failure_turn_count([cleared_episode()]) == 0


def test_a_different_successful_call_after_a_failure_counts_as_healing() -> None:
    """自愈的实现点是 ActionResult.error 回灌：芙兰 move 被拒后应该
    改成求玩家帮忙，而不是重复撞墙。"""
    turn = say_turn(
        npc_id="flandre",
        tool_calls=[
            call("move", to="hakurei_shrine"),
            call("ask_player", question="你帮我去看看？"),
        ],
        tool_results=[bad_result("tool_denied"), ok_result()],
    )

    tools = tool_metrics([episode([turn])])

    assert tools.self_heal_rate == 1.0
    assert failure_turn_count([episode([turn])]) == 1


def test_repeating_the_same_call_until_it_works_is_not_healing() -> None:
    """同一个 (tool, args) 重发到蒙对，是引擎或采样的抖动，
    不是模型读懂了回灌的错误原因。"""
    turn = say_turn(
        tool_calls=[
            call("reveal_info", fact="barrier_anomaly_time"),
            call("reveal_info", fact="barrier_anomaly_time"),
        ],
        tool_results=[bad_result("reveal_condition_unmet"), ok_result()],
    )

    assert tool_metrics([episode([turn])]).self_heal_rate == 0.0


def test_a_success_before_the_failure_is_not_healing() -> None:
    """自愈是「失败之后改招」。把失败前就成功的调用算进去，
    只要一个回合里恰好有一次成功就永远是 1.00。"""
    turn = say_turn(
        tool_calls=[
            call("ask_player", question="有事？"),
            call("reveal_info", fact="barrier_anomaly_time"),
        ],
        tool_results=[ok_result(), bad_result("reveal_condition_unmet")],
    )

    assert tool_metrics([episode([turn])]).self_heal_rate == 0.0


def test_self_heal_denominator_is_failing_turns_not_all_turns() -> None:
    """分母写成总回合数会把自愈率稀释成「一批里有多少回合发生过自愈」，
    数值随着顺利的回合变多而单调下降——越顺利看起来越差。"""
    healed = say_turn(
        tick=1,
        tool_calls=[call("move", to="muenzuka"), call("travel_to", destination="muenzuka")],
        tool_results=[bad_result("no_such_exit"), ok_result()],
    )
    clean = [
        say_turn(
            tick=i + 2, tool_calls=[call("ask_player", question="嗯？")], tool_results=[ok_result()]
        )
        for i in range(9)
    ]
    batch = [episode([healed, *clean])]

    assert failure_turn_count(batch) == 1
    assert tool_metrics(batch).self_heal_rate == 1.0


# ------------------------------------------------- 情报有没有真的说出口


def test_every_clue_declares_marks_that_really_appear_in_its_content() -> None:
    """`marks` 逐字取自 `content`。抄错一个字就留下一个永远不命中的判据——
    而它会报出一个漂亮的 0%（类 1 的形态：写了但从来不生效）。"""
    for fact in DEFS.facts.values():
        if not fact.is_clue:
            continue
        assert fact.marks, f"{fact.id} 是线索却没声明 marks"
        for mark in fact.marks:
            assert mark in fact.content, f"{fact.id} 的 marks 里「{mark}」不在正文里"


def _episode(turn: TurnRecord):
    return episode([turn])


def _reveal_turn(npc: str, fact: str, utterance: str) -> TurnRecord:
    return TurnRecord(
        tick=1,
        player_input="你知道些什么吗",
        kind="say",
        npc_id=npc,
        utterance=utterance,
        tool_calls=[{"tool": "reveal_info", "args": {"fact": fact}}],
        tool_results=[{"ok": True, "error_code": None, "observation": "说了"}],
    )


def test_a_reveal_whose_utterance_omits_the_content_is_not_delivered() -> None:
    """**这一格是为一次真实缺陷补的。** 实测 9 次成功揭示里只有 1 次台词里有情报：
    她说「妖怪？你倒是说说看，我倒要听听。」，而引擎那一刻把「结界在三天前的子时
    出现了一次异常波动」记进了 `known_facts`。工具成功 ≠ 玩家听到了。
    """
    said = _reveal_turn("reimu", "barrier_anomaly_time", "结界在子时抖过一下，方向正对无缘塚。")
    mute = _reveal_turn("reimu", "barrier_anomaly_time", "妖怪？你倒是说说看，我倒要听听。")

    m = reveal_delivery_metrics([_episode(said), _episode(mute)], DEFS)

    assert m.reveals == 2
    assert m.delivered == 1
    assert m.delivery_rate == 0.5
    assert m.by_fact["barrier_anomaly_time"] == (1, 2)


def test_a_failed_reveal_is_not_in_the_denominator() -> None:
    """门槛没开时她说不出来，那不是「没说出口」，是压根没揭示成功。
    混进分母会让这个比率随门槛难度变化。"""
    turn = _reveal_turn("reimu", "barrier_anomaly_time", "我可不知道。")
    turn.tool_results = [{"ok": False, "error_code": "reveal_condition_unmet", "observation": "x"}]

    m = reveal_delivery_metrics([_episode(turn)], DEFS)

    assert m.reveals == 0
    assert m.delivery_rate == 0.0


def test_delivery_is_split_by_fact() -> None:
    """三条线索的持有者是三个不同性格的 NPC。聚合成一个数会把「芙兰从来不说」
    和「魔理沙偶尔不说」混在一起——而前者是角色问题，后者是采样波动。"""
    batch = [
        _episode(_reveal_turn("marisa", "flower_magic_composition", "那花里有魔力结晶。")),
        _episode(_reveal_turn("flandre", "ancient_oblivion_memory", "破坏！破坏！")),
    ]

    m = reveal_delivery_metrics(batch, DEFS)

    assert m.by_fact["flower_magic_composition"] == (1, 1)
    assert m.by_fact["ancient_oblivion_memory"] == (0, 1)
