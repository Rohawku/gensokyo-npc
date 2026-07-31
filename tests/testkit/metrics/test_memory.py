from pathlib import Path

from gensokyo.testkit.metrics.memory import DENIAL_WORDS, memory_metrics
from gensokyo.testkit.personas import FLIP_LINES, MEMORY_PROBES, MemoryProbePlayer
from gensokyo.testkit.trajectory import Trajectory, TurnRecord
from gensokyo.world.ids import ItemId
from gensokyo.world.loader import load_defs
from gensokyo.world.observation import PlayerView

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFS = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")

RECALL = next(p for p in MEMORY_PROBES if p.kind == "recall")
NEGATIVE = next(p for p in MEMORY_PROBES if p.kind == "negative")


def test_an_alias_uniquely_identifies_one_item() -> None:
    """别名撞车会让指标把两件东西混成一件——「书」既像珍稀魔法书又出现在
    无关句子里，那种别名会把「她在看书」判成幻觉。所以别名必须唯一，
    而且不能是这个游戏的高频词（「花」「书」刻意没进别名表）。"""
    owners: dict[str, list[str]] = {}
    for item_id, item in DEFS.items.items():
        for surface in item.surfaces():
            owners.setdefault(surface, []).append(str(item_id))

    clashes = {s: ids for s, ids in owners.items() if len(ids) > 1}
    assert clashes == {}


def test_she_can_be_credited_for_saying_the_short_form() -> None:
    """实测 166 次召回探针里她一次都没说出「赛钱」三个字，说的是「你给的钱
    呢？」。只认全名的话事实召回率恒为 0——那是尺子看不见，不是她记不住。"""
    traj = _episode(_gave("赛钱"), _asked(RECALL.question, "你给的钱呢？我收着了。"))

    m = memory_metrics([traj], DEFS)

    assert m.fact_recall_rate == 1.0
    assert m.fact_hallucination_rate == 0.0


def test_a_short_form_also_counts_as_going_along_with_a_fabrication() -> None:
    """「你那魔法书呢？」是在顺着一本从没收到过的书往下说。第一版指标要求
    出现「珍稀魔法书」全名，于是这种明显的编造被判成 0。"""
    traj = _episode(_asked(NEGATIVE.question, "你那魔法书呢？我放好了。"))

    assert memory_metrics([traj], DEFS).false_affirmation_rate == 1.0


def _gave(item: str = "赛钱", ok: bool = True) -> TurnRecord:
    return TurnRecord(tick=0, player_input=f"/give {item}", kind="command", command_ok=ok)


def _asked(question: str, answer: str, memories: int = 2) -> TurnRecord:
    return TurnRecord(
        tick=1,
        player_input=question,
        kind="say",
        npc_id="reimu",
        utterance=answer,
        retrieved_memory_ids=[f"m-{i}" for i in range(memories)],
    )


def _episode(*turns: TurnRecord) -> Trajectory:
    return Trajectory(persona="memory_probe", seed=0, turns=list(turns))


# ---------------------------------------------------------------- 事实召回


def test_naming_a_truly_given_item_counts_as_recall() -> None:
    traj = _episode(_gave("赛钱"), _asked(RECALL.question, "你投了赛钱进箱子，我记得。"))

    m = memory_metrics([traj], DEFS)

    assert m.recall_probes == 1
    assert m.fact_recall_rate == 1.0
    assert m.fact_hallucination_rate == 0.0


def test_naming_something_never_given_counts_as_hallucination() -> None:
    """召回率和幻觉率必须一起看：只测召回会奖励「把所有物品名都念一遍」
    的模型——念得越多召回率越高。"""
    traj = _episode(_gave("赛钱"), _asked(RECALL.question, "你给过我赛钱，还有那本珍稀魔法书。"))

    m = memory_metrics([traj], DEFS)

    assert m.fact_recall_rate == 1.0
    assert m.fact_hallucination_rate == 1.0


def test_forgetting_shows_up_as_a_recall_miss_not_as_an_error() -> None:
    """「她忘了」是一个正常结果而不是失败——芙兰忘得快是设定。指标只负责
    把它量出来，不负责判它对错。"""
    traj = _episode(_gave("赛钱"), _asked(RECALL.question, "谁知道啊，我不记得了。"))

    m = memory_metrics([traj], DEFS)

    assert m.fact_recall_rate == 0.0
    assert m.fact_hallucination_rate == 0.0


def test_a_failed_give_is_not_ground_truth() -> None:
    """失败的 `/give`（身上没那么多）不构成真值。按 player_input 一律计入的话，
    「她记得一件没送成的东西」会被算成正确召回。"""
    traj = _episode(_gave("赛钱", ok=False), _asked(RECALL.question, "你给过我赛钱。"))

    m = memory_metrics([traj], DEFS)

    assert m.fact_recall_rate == 0.0
    assert m.fact_hallucination_rate == 1.0


def test_picking_something_up_is_not_giving_it_to_her() -> None:
    """`/pick 赛钱` 是玩家从地上捡起来，不是交给她。把所有成功的指令都算成
    赠予的话，「她记得你给过赛钱」在你根本没给的情况下也算召回正确。"""
    picked = TurnRecord(tick=0, player_input="/pick 赛钱", kind="command", command_ok=True)
    traj = _episode(picked, _asked(RECALL.question, "你给过我赛钱。"))

    m = memory_metrics([traj], DEFS)

    assert m.fact_recall_rate == 0.0
    assert m.fact_hallucination_rate == 1.0


def test_the_pay_alias_counts_as_giving() -> None:
    """别名表里 `pay` 映射到 give。真值判定必须跟着别名表走，另写一份很快
    就会漏掉新加的别名——第一版就漏了 pay，还多写了两个不存在的中文别名。"""
    paid = TurnRecord(tick=0, player_input="/pay 赛钱", kind="command", command_ok=True)
    traj = _episode(paid, _asked(RECALL.question, "你给过我赛钱。"))

    assert memory_metrics([traj], DEFS).fact_recall_rate == 1.0


def test_ground_truth_does_not_leak_across_episodes() -> None:
    """两局之间没有共享历史。第一局送过赛钱不能让第二局的召回算对。"""
    first = _episode(_gave("赛钱"))
    second = _episode(_asked(RECALL.question, "你给过我赛钱。"))

    m = memory_metrics([first, second], DEFS)

    assert m.fact_recall_rate == 0.0
    assert m.fact_hallucination_rate == 1.0


# ---------------------------------------------------------------- 负例


def test_going_along_with_a_fabricated_event_is_counted() -> None:
    traj = _episode(_asked(NEGATIVE.question, "你的珍稀魔法书我收好了，放心。"))

    m = memory_metrics([traj], DEFS)

    assert m.negative_probes == 1
    assert m.false_affirmation_rate == 1.0


def test_denying_a_fabricated_event_is_not_counted() -> None:
    traj = _episode(_asked(NEGATIVE.question, "什么魔法书？你没给过我这种东西。"))

    m = memory_metrics([traj], DEFS)

    assert m.false_affirmation_rate == 0.0


def test_not_mentioning_the_fabricated_thing_at_all_is_not_counted() -> None:
    """她压根没提那件东西，谈不上「顺着编」。把它算进去会让「敷衍一句」
    和「编造记忆」混成一个数。"""
    traj = _episode(_asked(NEGATIVE.question, "你管的太多了。"))

    m = memory_metrics([traj], DEFS)

    assert m.false_affirmation_rate == 0.0


def test_denial_words_are_reported_as_a_library_size() -> None:
    assert len(DENIAL_WORDS) > 5


# ---------------------------------------------------------------- 检索通路


def test_zero_recall_turns_are_counted() -> None:
    """不是质量指标，是体检项：全是零召回说明记忆层根本没接上，而那种
    情况下召回率为 0 会被误读成「模型记性差」。"""
    traj = _episode(
        _asked(RECALL.question, "不记得。", memories=0),
        _asked(RECALL.question, "赛钱。", memories=3),
    )

    m = memory_metrics([traj], DEFS)

    assert m.zero_recall_turns == 1
    assert m.recalled_per_turn == 1.5


def test_non_probe_turns_do_not_touch_the_probe_rates() -> None:
    traj = _episode(
        _gave("赛钱"),
        _asked("随便聊聊吧。", "哼。"),
        _asked(RECALL.question, "你给过我赛钱。"),
    )

    m = memory_metrics([traj], DEFS)

    assert m.recall_probes == 1
    assert m.negative_probes == 0


# ---------------------------------------------------------------- 探针人格


def _view(coins: int = 8) -> PlayerView:
    return PlayerView(
        tick=0,
        location_id="hakurei_shrine",
        location_name="博丽神社",
        location_description="幻想乡东端的神社。",
        inventory={"赛钱": coins} if coins else {},
        quest_stage="S0_UNAWARE",
    )


def test_the_prober_sets_up_ground_truth_before_asking() -> None:
    """先投赛钱造真值，再垫几个回合把它推出 12 轮原话窗口——不垫的话
    「记得」和「刚才说过」区分不开，测的就不是长期记忆了。"""
    prober = MemoryProbePlayer.from_dirs(REPO_ROOT / "scenario", REPO_ROOT / "characters")

    inputs = [prober.next_input(_view(), "") for _ in range(12)]

    gifts = [i for i in inputs if i.startswith("/give")]
    probes = [i for i in inputs if i in {p.question for p in MEMORY_PROBES}]
    assert len(gifts) == 3
    assert inputs.index(probes[0]) > inputs.index(gifts[-1]) + 1


def test_the_prober_never_gives_the_items_it_asks_about_in_negatives() -> None:
    """负例探针的前提就是那件东西从没给过。人格若真给了，「她说记得」
    就不再是幻觉，整类探针失去意义。"""
    prober = MemoryProbePlayer.from_dirs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    forbidden = {
        surface
        for p in MEMORY_PROBES
        if p.kind == "negative"
        for surface in DEFS.items[ItemId(p.subject_item)].surfaces()
    }

    inputs = [prober.next_input(_view(), "") for _ in range(40)]

    for given in (i.partition(" ")[2] for i in inputs if i.startswith("/give")):
        assert given not in forbidden


def test_the_prober_costs_no_llm_calls() -> None:
    prober = MemoryProbePlayer.from_dirs(REPO_ROOT / "scenario", REPO_ROOT / "characters")

    for _ in range(20):
        prober.next_input(_view(), "")

    assert prober.llm_calls == 0


def test_the_effective_denominator_is_episodes_not_probes() -> None:
    """同一局里的探针结果高度相关——她整局要么认真回答、要么整局敷衍，实测
    每局召回率在 0.00 和 0.76 之间。16 次探针不是 16 个独立样本，是 1 个样本
    重复观测 16 次。按探针次数报分母等于把置信区间凭空缩小 4 倍（坑 #25）。"""
    one_episode = _episode(
        _gave("赛钱"),
        *[_asked(RECALL.question, "你给的钱呢？") for _ in range(8)],
    )

    m = memory_metrics([one_episode], DEFS)

    assert m.recall_probes == 8
    assert m.probe_episodes == 1


def test_episodes_without_probes_do_not_inflate_the_denominator() -> None:
    """honest / jailbreak 那些局里没有探针。把它们算进探针分母会让比率的
    分母比真实观测多出几十——正是坑 #18 那个错误。"""
    probed = _episode(_gave("赛钱"), _asked(RECALL.question, "你给的钱呢？"))
    unprobed = _episode(_gave("赛钱"), _asked("随便聊聊吧。", "哼。"))

    m = memory_metrics([probed, unprobed], DEFS)

    assert m.probe_episodes == 1


# ---------------------------------------------------------------- 矛盾检出

FLIP = next(iter(FLIP_LINES))


def _flip(answer: str, npc: str = "reimu") -> TurnRecord:
    return TurnRecord(tick=2, player_input=FLIP, kind="say", npc_id=npc, utterance=answer)


def test_pointing_out_the_change_of_story_is_counted() -> None:
    traj = _episode(_flip("你刚才明明说的是另一套，到底哪句是真的？"))

    m = memory_metrics([traj], DEFS)

    assert m.contradiction_probes == 1
    assert m.contradiction_flag_rate["reimu"] == 1.0


def test_letting_it_slide_is_not_counted() -> None:
    traj = _episode(_flip("哦。"))

    assert memory_metrics([traj], DEFS).contradiction_flag_rate["reimu"] == 0.0


def test_the_rate_is_split_by_character_because_the_right_answer_differs() -> None:
    """灵梦该直接怼；芙兰大概率根本没记住前一句，而没记住在她身上不算失败
    ——她的 λ 是全场最大的，忘得快是设定。跨角色聚合会把「符合人设的遗忘」
    和「被绕晕了」加成一个数，正是坑 #18 那个错误。"""
    traj = _episode(
        _flip("你刚才说的不是这个。", npc="reimu"),
        _flip("是吗？我们来玩吧！", npc="flandre"),
    )

    rates = memory_metrics([traj], DEFS).contradiction_flag_rate

    assert rates["reimu"] == 1.0
    assert rates["flandre"] == 0.0


def test_the_contradiction_denominator_is_also_reported_in_episodes() -> None:
    """六个矛盾对在一局里出现，效果和记忆探针一样高度相关（坑 #25）。"""
    traj = _episode(*[_flip("你刚才说的不是这个。") for _ in range(6)])

    m = memory_metrics([traj], DEFS)

    assert m.contradiction_probes == 6
    assert m.contradiction_episodes == 1


def test_non_flip_turns_are_not_contradiction_probes() -> None:
    traj = _episode(_asked("随便聊聊吧。", "你刚才说什么？"))

    m = memory_metrics([traj], DEFS)

    assert m.contradiction_probes == 0
    assert m.contradiction_flag_rate == {}
