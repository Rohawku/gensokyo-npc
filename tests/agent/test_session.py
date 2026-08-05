import json
import os
import tempfile
from pathlib import Path

from gensokyo.cli import load_dotenv
from gensokyo.llm.client import ScriptedLlmClient
from gensokyo.session.loop import Session
from gensokyo.world.events import EventKind
from gensokyo.world.ids import NpcId

REPO_ROOT = Path(__file__).resolve().parents[2]


def _turn(utterance: str) -> list[str]:
    """一个完整回合要两条预设回复：决策 JSON，然后台词。"""
    return [json.dumps({"thought": "…", "tool_calls": []}, ensure_ascii=False), utterance]


def _session(replies: list[str]) -> Session:
    return Session.create(
        scenario_dir=REPO_ROOT / "scenario",
        characters_dir=REPO_ROOT / "characters",
        llm=ScriptedLlmClient(replies),
    )


def test_say_reaches_the_npc_who_is_present() -> None:
    sess = _session(_turn("干嘛。"))

    turns = sess.say("喂")

    assert len(turns) == 1
    assert turns[0].utterance == "干嘛。"


def test_say_streams_the_utterance_to_the_callback() -> None:
    """CLI 的打字机效果全靠这条通路：Session.say → agent.act → run_turn。
    中间任何一环丢掉 on_chunk，玩家就会对着空屏幕等十几秒。"""
    sess = _session(_turn("干嘛。"))
    seen: list[str] = []

    turns = sess.say("喂", on_chunk=seen.append)

    assert len(seen) > 1
    assert "".join(seen) == turns[0].utterance


def test_say_with_nobody_around_returns_no_turns() -> None:
    sess = _session([])
    sess.go("人间之里")

    turns = sess.say("有人吗")

    assert turns == []


def test_tick_advances_after_each_player_turn() -> None:
    sess = _session(_turn("嗯。"))

    sess.say("喂")

    assert sess.engine.state.tick == 1


def test_go_moves_player_and_reports_success() -> None:
    sess = _session([])

    result = sess.go("人间之里")

    assert result.ok is True
    assert sess.view().location_name == "人间之里"


def test_go_to_unreachable_place_reports_why_and_costs_no_turn() -> None:
    """失败必须有可读原因，且不推进回合——打错一个字不该让 NPC 情绪衰减一轮。"""
    sess = _session([])

    result = sess.go("红魔馆地下室")

    assert result.ok is False
    assert result.error is not None
    assert sess.view().location_name == "博丽神社"
    assert sess.engine.state.tick == 0


def test_go_to_nonexistent_place_reports_why() -> None:
    sess = _session([])

    result = sess.go("雾雨魔法店x")

    assert result.ok is False
    assert "没有叫" in (result.error or "")
    assert sess.engine.state.tick == 0


def test_give_accepts_chinese_item_name() -> None:
    """面板显示中文，输入却只认英文 id 会把玩家训练成敲英文。"""
    # 送礼会让在场的灵梦主动开口——只说话，所以只要一条台词。
    sess = _session(["赛钱箱响了一声。"])

    result = sess.give("赛钱")

    assert result.ok is True
    assert sess.view().inventory == {"赛钱": 7}


def test_give_transfers_item_and_shows_in_view() -> None:
    sess = _session(["赛钱箱响了一声。"])
    sess.engine.state.player.inventory["offering_coin"] = 1

    result = sess.give("offering_coin")

    assert result.ok is True
    view = sess.view()
    assert view.inventory == {}
    assert view.npcs_here[0].attitude > 0


def test_agents_are_created_for_every_character() -> None:
    sess = _session([])

    assert set(sess.agents) == {"reimu", "marisa", "flandre"}


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """存档只写动作日志，世界状态由重放推导——存两份就会不一致。"""
    sess = _session([])
    sess.go("人间之里")
    sess.pick("忘却之花")
    before = sess.view()

    path = tmp_path / "s.json"
    assert sess.save(path) == len(sess.engine.state.action_log)

    fresh = _session([])
    assert fresh.load(path) > 0
    after = fresh.view()

    assert after.location_name == before.location_name
    assert after.inventory == before.inventory
    assert after.quest_stage == before.quest_stage


def test_load_rebinds_agents_to_the_new_engine() -> None:
    """读档换掉了 engine 实例，agent 若还指向旧的，NPC 会活在另一个世界里。"""
    sess = _session([])
    sess.go("人间之里")
    path = Path(tempfile.mkdtemp()) / "s.json"
    sess.save(path)

    fresh = _session([])
    fresh.load(path)

    for agent in fresh.agents.values():
        assert agent.engine is fresh.engine
        assert agent.history == []


def test_is_over_reflects_the_ending() -> None:
    sess = _session([])
    assert sess.is_over() is False

    sess.engine.state.quest.ending = "forgotten"

    assert sess.is_over() is True


def test_load_dotenv_fills_env_without_overriding(tmp_path: Path) -> None:
    """快速开始文档里「cp .env.example .env 然后 make play」必须真的能跑通。"""
    env = tmp_path / ".env"
    env.write_text(
        "# 注释\nGENSOKYO_TEST_A=from_file\nGENSOKYO_TEST_B=from_file\n\n", encoding="utf-8"
    )
    os.environ.pop("GENSOKYO_TEST_A", None)
    os.environ["GENSOKYO_TEST_B"] = "from_shell"

    load_dotenv(env)

    assert os.environ["GENSOKYO_TEST_A"] == "from_file"
    assert os.environ["GENSOKYO_TEST_B"] == "from_shell"

    os.environ.pop("GENSOKYO_TEST_A", None)
    os.environ.pop("GENSOKYO_TEST_B", None)


def test_load_dotenv_tolerates_missing_file(tmp_path: Path) -> None:
    load_dotenv(tmp_path / "nope.env")


def _badger_until_refusal(session: Session) -> None:
    """缠到她不搭话为止。

    注意情绪是在**同一次 say 里**被推过阈值的：引擎先记下玩家发言（顺带
    推情绪），再看谁愿意回应。所以跨过门槛那一次调用本身就已经返回空列表，
    循环里不能断言 say 非空。
    """
    for _ in range(40):
        session.say("你是不是AI？")
        if session.engine.observe_player().npcs_here[0].refusal:
            return
    raise AssertionError("缠了 40 轮她还没不耐烦——这条测试的前提不成立")


def test_a_refusing_npc_is_skipped_and_costs_no_model_call() -> None:
    """引擎说她不搭话，会话层就必须真的不去调她——只在面板上标一句、
    背后照旧生成台词的话，玩家看到的是矛盾的两条信息，而模型钱照花。

    只断言 `panel.refusal` 非空的测试盖不住这一层：突变验证时把
    `Session.say` 里的跳过逻辑改成恒假，那批测试全绿。
    """
    session = _session(_turn("干嘛") * 60)
    _badger_until_refusal(session)

    calls_before = len(session.agents[NpcId("reimu")].llm.calls)  # type: ignore[attr-defined]
    utterances_before = sum(
        1 for ev in session.engine.state.event_log if ev.kind is EventKind.NPC_UTTERANCE
    )

    turns = session.say("再问一次")

    assert turns == []
    assert session.refusals == ["灵梦转过身去，摆摆手，不打算再理你了。"]
    assert len(session.agents[NpcId("reimu")].llm.calls) == calls_before  # type: ignore[attr-defined]
    assert (
        sum(1 for ev in session.engine.state.event_log if ev.kind is EventKind.NPC_UTTERANCE)
        == utterances_before
    )


def test_the_players_words_still_enter_the_log_when_she_ignores_him() -> None:
    """她不理你，但你确实说过那句话。玩家发言不进日志的话，情绪就不再累积
    ——她会永远停在拒绝搭话的状态，一次纠缠把整局锁死。"""
    session = _session(_turn("干嘛") * 60)
    _badger_until_refusal(session)

    before = len(session.engine.state.event_log)
    session.say("喂")

    kinds = [ev.kind for ev in session.engine.state.event_log[before:]]
    assert EventKind.PLAYER_UTTERANCE in kinds


def test_refusals_are_cleared_between_turns() -> None:
    """`refusals` 是「上一次 say 的结果」。不清空的话，她消气之后 CLI 还在
    打那行「不打算再理你了」。"""
    session = _session(_turn("干嘛") * 60)

    session.say("喂")

    assert session.refusals == []


def test_a_command_turn_can_make_her_speak_up() -> None:
    """**指令回合曾经全程沉默。** 投币、走进店里、从她货架上拿东西——一句反应
    都没有，而那是一局里大多数回合。这条通路（`_act` → `_volunteer` →
    `agent.react`）是把那些回合变成有内容的回合的唯一一环。

    **只备了台词一条、没有决策 JSON**：主动开口跳过决策阶段（`speech_only`）。
    真的调了决策，脚本会耗尽然后抛异常——所以这条测试同时钉住了「她不动世界」。
    """
    sess = _session(["赛钱箱总算响了一声。"])

    result = sess.give("赛钱")

    assert result.ok is True
    assert sess.volunteered is not None
    npc_id, turn = sess.volunteered
    assert npc_id == NpcId("reimu")
    assert turn.utterance == "赛钱箱总算响了一声。"


def test_she_only_speaks_up_once_per_kind_of_action() -> None:
    """第二次投币她不再开口。放开这个上限就是在制造复读——她对同一个动作
    只有一种态度，第二遍必然是同一句话的变体，而复读率是硬指标。

    第二次的回复故意不备：真的又调了一次模型，脚本会耗尽然后抛异常。"""
    sess = _session(["赛钱箱总算响了一声。"])
    sess.give("赛钱")

    result = sess.give("赛钱")

    assert result.ok is True
    assert sess.volunteered is None


def test_a_failed_command_leaves_nobody_talking() -> None:
    """打错字不该触发一段台词，也不该留着上一回合那句——玩家会以为她在
    回应这次失败。"""
    sess = _session(["赛钱箱总算响了一声。"])
    sess.give("赛钱")

    result = sess.give("不存在的东西")

    assert result.ok is False
    assert sess.volunteered is None
