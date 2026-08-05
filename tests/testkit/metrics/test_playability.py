"""可玩性指标的测试。

这一组的核心风险是**分母选错**：越狱玩家只说话不做事，对话占比接近 1，
把他们算进来会得出「非常好」的结论（坑 #18 的形态）。
"""

from gensokyo.testkit.metrics.playability import (
    QUESTING_PERSONAS,
    playability_metrics,
)
from gensokyo.testkit.trajectory import Trajectory, TurnRecord


def _cmd(text: str) -> TurnRecord:
    return TurnRecord(tick=0, player_input=text, kind="command", command_ok=True)


def _say(
    text: str = "神社怎么样", npc: str | None = "reimu", utterance: str = "哼。"
) -> TurnRecord:
    return TurnRecord(tick=1, player_input=text, kind="say", npc_id=npc, utterance=utterance)


def _episode(*turns: TurnRecord, persona: str = "honest") -> Trajectory:
    return Trajectory(persona=persona, seed=0, turns=list(turns))


def test_the_real_playthrough_shape_is_measured() -> None:
    """实测形态：21 回合通关，16 个指令回合、5 个对话回合、NPC 开口 5 次。
    **一个 LLM 驱动的对话游戏，76% 的回合与 LLM 无关**——而通关率、复读率、
    安全项全都看不见这件事。"""
    turns = [_cmd("/go 人间之里") for _ in range(8)]
    turns += [_cmd("/give 赛钱") for _ in range(7)]
    turns += [_cmd("/pick 赛钱")]
    turns += [_say() for _ in range(5)]

    m = playability_metrics([_episode(*turns)])

    assert m.turns == 21
    assert m.command_turns == 16
    assert m.dialogue_turns == 5
    assert m.npc_utterances == 5
    assert m.dialogue_share < 0.25
    assert m.commands_per_utterance == 3.2


def test_jailbreak_and_fickle_are_excluded_from_the_denominator() -> None:
    """**他们不是来通关的。** 越狱玩家只说话不做事，对话占比接近 1，混进来
    会把「这游戏不是对话游戏」这个问题盖掉——坑 #18 那个错误。"""
    questing = _episode(_cmd("/go 人间之里"), _say(), persona="honest")
    jailbreak = _episode(*[_say() for _ in range(20)], persona="jailbreak")

    m = playability_metrics([questing, jailbreak])

    assert m.episodes == 1
    assert m.turns == 2
    assert "jailbreak" not in QUESTING_PERSONAS


def test_memory_probe_is_also_excluded() -> None:
    """探针人格循环问同样的 4 个问题，对话占比恒等于 1。那不是玩法。"""
    assert "memory_probe" not in QUESTING_PERSONAS


def test_a_turn_where_she_refuses_to_speak_is_counted_separately() -> None:
    """引擎判定她不搭话时会跳过模型调用（被缠久了那个机制）。那种回合玩家
    说了话但没人回应——`npc_utterances` 必须小于 `dialogue_turns`，否则
    「她拒绝搭话」会被记成「她回应了」。"""
    turns = [_say(), _say(npc=None, utterance=""), _say()]

    m = playability_metrics([_episode(*turns)])

    assert m.dialogue_turns == 3
    assert m.npc_utterances == 2
    assert m.silent_dialogue_turns == 1


def test_utterances_are_split_by_npc() -> None:
    """聚合值会掩盖「某个 NPC 几乎不出场」——三个人各开口 5 次和一个人开口
    15 次，聚合数字一样。"""
    turns = [_say(npc="reimu") for _ in range(4)]
    turns += [_say(npc="marisa"), _say(npc="flandre")]

    m = playability_metrics([_episode(*turns)])

    assert m.utterances_by_npc == {"flandre": 1, "marisa": 1, "reimu": 4}


def test_the_command_histogram_shows_where_the_turns_went() -> None:
    """7 次 `/give` 里有 6 次是重复同一个动作凑数值。直方图让那件事看得见。"""
    turns = [_cmd("/give 赛钱") for _ in range(7)]
    turns += [_cmd("/go 人间之里") for _ in range(8)]

    m = playability_metrics([_episode(*turns)])

    assert m.command_histogram == {"give": 7, "go": 8}


def test_an_empty_batch_reports_zeros_not_a_crash() -> None:
    m = playability_metrics([])

    assert m.episodes == 0
    assert m.dialogue_share == 0.0
    assert m.commands_per_utterance == 0.0
    assert m.utterances_per_episode == 0.0


def test_a_batch_with_no_utterances_does_not_divide_by_zero() -> None:
    """全程只敲指令时 `commands_per_utterance` 的分母是 0。返回 0 而不是崩掉，
    但那个 0 要读成「一句话都没听到」，不是「效率完美」。"""
    m = playability_metrics([_episode(_cmd("/go 人间之里"))])

    assert m.npc_utterances == 0
    assert m.commands_per_utterance == 0.0
