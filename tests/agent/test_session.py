import json
import tempfile
from pathlib import Path

from gensokyo.llm.client import ScriptedLlmClient
from gensokyo.session.loop import Session

REPO_ROOT = Path(__file__).resolve().parents[2]


def _reply(utterance: str) -> str:
    return json.dumps(
        {"thought": "…", "tool_calls": [], "utterance": utterance}, ensure_ascii=False
    )


def _session(replies: list[str]) -> Session:
    return Session.create(
        scenario_dir=REPO_ROOT / "scenario",
        characters_dir=REPO_ROOT / "characters",
        llm=ScriptedLlmClient(replies),
    )


def test_say_reaches_the_npc_who_is_present() -> None:
    sess = _session([_reply("干嘛。")])

    turns = sess.say("喂")

    assert len(turns) == 1
    assert turns[0].utterance == "干嘛。"


def test_say_with_nobody_around_returns_no_turns() -> None:
    sess = _session([])
    sess.go("人间之里")

    turns = sess.say("有人吗")

    assert turns == []


def test_tick_advances_after_each_player_turn() -> None:
    sess = _session([_reply("嗯。")])

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
    sess = _session([])

    result = sess.give("赛钱")

    assert result.ok is True
    assert sess.view().inventory == {"赛钱": 7}


def test_give_transfers_item_and_shows_in_view() -> None:
    sess = _session([])
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
