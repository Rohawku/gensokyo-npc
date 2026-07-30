import json
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


def test_go_moves_player_and_returns_view() -> None:
    sess = _session([])

    view = sess.go("人间之里")

    assert view.location_name == "人间之里"


def test_go_to_unreachable_place_keeps_position() -> None:
    sess = _session([])

    view = sess.go("红魔馆地下室")

    assert view.location_name == "博丽神社"


def test_give_transfers_item_and_shows_in_view() -> None:
    sess = _session([])
    sess.engine.state.player.inventory["offering_coin"] = 1

    view = sess.give("offering_coin")

    assert view.inventory == {}
    assert view.npcs_here[0].attitude > 0


def test_agents_are_created_for_every_character() -> None:
    sess = _session([])

    assert set(sess.agents) == {"reimu", "marisa", "flandre"}
