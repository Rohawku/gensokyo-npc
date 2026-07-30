from pathlib import Path

from gensokyo.llm.client import ScriptedLlmClient
from gensokyo.testkit.personas import (
    CONTRADICTION_PAIRS,
    ESCORT,
    JAILBREAK_LINES,
    QUESTION,
    STRIKE,
    FicklePlayer,
    GameMap,
    HonestPlayer,
    JailbreakPlayer,
    SmoothTalkerPlayer,
)
from gensokyo.world.ids import LocationId, NpcId
from gensokyo.world.observation import NpcPanel, PlayerView

REPO_ROOT = Path(__file__).resolve().parents[2]


def _map() -> GameMap:
    return GameMap.load(REPO_ROOT / "scenario", REPO_ROOT / "characters")


def _view(
    location_id: str,
    *,
    stage: str = "S2_CLUES",
    objective: str = "还有线索没拿到。",
    warning: str = "",
    npcs: list[NpcPanel] | None = None,
    known: int = 1,
    inventory: dict[str, int] | None = None,
) -> PlayerView:
    return PlayerView(
        tick=1,
        location_id=LocationId(location_id),
        location_name=location_id,
        location_description="",
        quest_stage=stage,
        objective=objective,
        oblivion_warning=warning,
        known_facts=["线索" * (i + 1) for i in range(known)],
        inventory=inventory if inventory is not None else {"赛钱": 8},
        npcs_here=npcs or [],
    )


def _panel(npc_id: str) -> NpcPanel:
    return NpcPanel(
        npc_id=NpcId(npc_id), name=npc_id, attitude=0, emotion_var="x", emotion=0.0, mode="normal"
    )


# ---------------------------------------------------------------------- GameMap


def test_map_is_loaded_from_yaml_not_hardcoded() -> None:
    """地图硬编码在 Python 里的话，加一个地点就要改人格代码。"""
    gm = _map()

    assert gm.name_of(LocationId("muenzuka")) == "无缘塚"
    assert gm.home_of(NpcId("marisa")) == LocationId("kirisame_magic_shop")
    assert gm.sites_of("rare_book") == [LocationId("kirisame_magic_shop")]


def test_path_finds_the_two_hop_route_to_muenzuka() -> None:
    gm = _map()

    path = gm.path(LocationId("kirisame_magic_shop"), LocationId("muenzuka"))

    assert path == [LocationId("human_village"), LocationId("muenzuka")]
    assert gm.next_hop(LocationId("kirisame_magic_shop"), LocationId("muenzuka")) == "human_village"
    assert gm.distance(LocationId("kirisame_magic_shop"), LocationId("muenzuka")) == 2


# ----------------------------------------------------------------- HonestPlayer


def test_honest_gives_a_coin_when_the_gate_is_still_closed() -> None:
    player = HonestPlayer(_map())

    move = player.next_input(_view("hakurei_shrine", npcs=[_panel("reimu")], known=0), "")

    assert move == "/give 赛钱"


def test_honest_asks_once_the_objective_says_she_will_talk() -> None:
    """门槛开没开只能从 objective 看出来——好感门槛的具体数值玩家不知道。"""
    player = HonestPlayer(_map())

    move = player.next_input(
        _view(
            "hakurei_shrine",
            objective="灵梦已经愿意开口了——直接问她无缘塚的事。",
            npcs=[_panel("reimu")],
            known=0,
        ),
        "",
    )

    assert move == QUESTION


def test_honest_picks_up_what_marisa_wants() -> None:
    player = HonestPlayer(_map())
    view = _view("kirisame_magic_shop", npcs=[_panel("marisa")], known=1)
    view.items_here = {"珍稀魔法书": 1}

    assert player.next_input(view, "") == "/pick 珍稀魔法书"


def test_honest_leaves_muenzuka_the_moment_memory_starts_slipping() -> None:
    """收尾之前赖在花田里只会丢线索，而丢掉的线索要重新跑一趟才能拿回来。"""
    player = HonestPlayer(_map())

    move = player.next_input(_view("muenzuka", warning="思绪开始模糊了", known=2), "")

    assert move == "/go 人间之里"


def test_honest_stays_in_muenzuka_during_the_finale() -> None:
    """收尾阶段例外：她就在这儿，走开等于放弃通关。"""
    player = HonestPlayer(_map())

    move = player.next_input(
        _view(
            "muenzuka",
            stage="S3_SOURCE",
            warning="思绪开始模糊了",
            npcs=[_panel("marisa")],
            known=3,
        ),
        "",
    )

    assert move == STRIKE


def test_honest_asks_a_finisher_to_come_along_once_clues_are_complete() -> None:
    player = HonestPlayer(_map())

    move = player.next_input(
        _view("kirisame_magic_shop", stage="S3_SOURCE", npcs=[_panel("marisa")], known=3), ""
    )

    assert move == ESCORT


def test_honest_heads_for_muenzuka_after_the_escort_left() -> None:
    player = HonestPlayer(_map())
    player.next_input(
        _view("kirisame_magic_shop", stage="S3_SOURCE", npcs=[_panel("marisa")], known=3), ""
    )

    move = player.next_input(_view("kirisame_magic_shop", stage="S3_SOURCE", known=3), "")

    assert move == "/go 人间之里"


def test_honest_gives_up_on_someone_who_never_opens_up() -> None:
    """卡住必须有出口。没有上限的话一次卡死会烧光整局回合数，
    而「她就是不开口」这件事本身才是要报告的发现。"""
    player = HonestPlayer(_map())
    view = _view(
        "hakurei_shrine",
        objective="灵梦已经愿意开口了",
        npcs=[_panel("reimu")],
        known=0,
    )

    moves = [player.next_input(view, "") for _ in range(8)]

    assert moves[:6] == [QUESTION] * 6
    # 放弃灵梦之后目标换成魔理沙，于是开始往魔法店走。
    assert moves[6] == "/go 人间之里"


def test_honest_is_deterministic_for_the_same_view_sequence() -> None:
    views = [
        _view("hakurei_shrine", npcs=[_panel("reimu")], known=0),
        _view("hakurei_shrine", npcs=[_panel("reimu")], known=0),
        _view("human_village", known=0),
    ]

    a = [HonestPlayer(_map()).next_input(v, "") for v in views]
    b = [HonestPlayer(_map()).next_input(v, "") for v in views]

    assert a == b


# -------------------------------------------------------------- JailbreakPlayer


def test_jailbreak_library_has_no_duplicates_and_enough_variety() -> None:
    """语句库固定且不重复，抗越狱率才能跨模型版本比较——
    模型生成的攻击每次都不同，指标升降就分不清是谁变了。"""
    assert len(JAILBREAK_LINES) >= 12
    assert len(set(JAILBREAK_LINES)) == len(JAILBREAK_LINES)


def test_jailbreak_covers_the_five_attack_shapes() -> None:
    joined = "\n".join(JAILBREAK_LINES)

    assert "语言模型" in joined  # 直接质询身份
    assert "忽略之前的所有设定" in joined  # 要求脱离角色
    assert "系统提示" in joined  # 套系统提示
    assert "工具" in joined and "fact id" in joined  # 元层追问
    assert "ignore previous instructions" in joined  # 中英混杂


def test_jailbreak_rotation_is_seeded_and_covers_the_whole_library() -> None:
    a = JailbreakPlayer(seed=0)
    b = JailbreakPlayer(seed=5)
    view = _view("hakurei_shrine")

    first = [a.next_input(view, "") for _ in range(len(JAILBREAK_LINES))]
    second = [b.next_input(view, "") for _ in range(3)]

    assert set(first) == set(JAILBREAK_LINES)
    assert second[0] == JAILBREAK_LINES[5]
    assert JailbreakPlayer(seed=0).next_input(view, "") == first[0]


# ----------------------------------------------------------------- FicklePlayer


def test_fickle_emits_real_contradiction_pairs_and_records_them() -> None:
    """矛盾对成对记录，W2 的记忆指标才能区分「她忘了」和「她被绕晕了」。"""
    player = FicklePlayer(seed=0, gap=3)
    view = _view("hakurei_shrine")

    said = [player.next_input(view, "") for _ in range(12)]

    assert player.contradictions
    for first, flip in player.contradictions:
        assert flip - first >= 3
        pair = (said[first], said[flip])
        assert pair in CONTRADICTION_PAIRS


def test_fickle_records_the_first_pair_at_the_expected_turns() -> None:
    player = FicklePlayer(seed=0, gap=2)
    view = _view("hakurei_shrine")

    said = [player.next_input(view, "") for _ in range(6)]

    assert said[0] == CONTRADICTION_PAIRS[0][0]
    assert said[2] == CONTRADICTION_PAIRS[0][1]
    assert player.contradictions[0] == (0, 2)


def test_fickle_is_deterministic_and_seed_shifts_the_pair_order() -> None:
    view = _view("hakurei_shrine")

    a = [FicklePlayer(seed=1).next_input(view, "") for _ in range(1)]
    again = [FicklePlayer(seed=1).next_input(view, "") for _ in range(1)]
    other = FicklePlayer(seed=2).next_input(view, "")

    assert a == again
    assert other != a[0]


# ----------------------------------------------------------- SmoothTalkerPlayer


def test_smooth_talker_costs_exactly_one_call_per_turn() -> None:
    llm = ScriptedLlmClient(["这事你多少知道点吧？", "别装了，我看得出来。"])
    player = SmoothTalkerPlayer(llm)
    view = _view("hakurei_shrine", npcs=[_panel("reimu")])

    player.next_input(view, "")
    player.next_input(view, "干嘛。")

    assert len(llm.calls) == 2


def test_smooth_talker_strips_thinking_blocks_quotes_and_extra_lines() -> None:
    """玩家输入只能是一行话。整段旁白喂进去会污染 NPC 的对话历史。"""
    llm = ScriptedLlmClient(["<think>先共情</think>\n「你也觉得那些花不对劲吧？」\n（她皱了皱眉）"])
    player = SmoothTalkerPlayer(llm)

    said = player.next_input(_view("hakurei_shrine", npcs=[_panel("reimu")]), "")

    assert said == "你也觉得那些花不对劲吧？"


def test_smooth_talker_prompt_carries_scene_npc_and_last_line() -> None:
    llm = ScriptedLlmClient(["你知道点什么吧。"])
    player = SmoothTalkerPlayer(llm)
    view = _view("hakurei_shrine", npcs=[_panel("reimu")])

    player.next_input(view, "你管的太多了。")

    prompt = llm.calls[0][-1].content
    assert "reimu" in prompt
    assert "你管的太多了。" in prompt
    assert "不要给她任何东西" in prompt


def test_smooth_talker_survives_a_dead_endpoint() -> None:
    """端点抽风不该让整局崩掉——那一局的其他指标还有价值。"""
    player = SmoothTalkerPlayer(ScriptedLlmClient([]))

    said = player.next_input(_view("hakurei_shrine", npcs=[_panel("reimu")]), "")

    assert said
