import json
from pathlib import Path

from gensokyo.testkit.trajectory import Trajectory, TurnRecord

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sample() -> Trajectory:
    return Trajectory(
        persona="honest",
        seed=3,
        turns=[
            TurnRecord(
                tick=0,
                player_input="/give 赛钱",
                kind="command",
                command_ok=True,
                view_after={"tick": 1, "location_name": "博丽神社"},
            ),
            TurnRecord(
                tick=1,
                player_input="那些花是怎么回事",
                kind="say",
                npc_id="reimu",
                utterance="你管的太多了。",
                thought="又是这个人",
                tool_calls=[{"tool": "reveal_info", "args": {"fact": "barrier_anomaly_time"}}],
                tool_results=[
                    {"ok": False, "error_code": "reveal_condition_unmet", "observation": "还不想说"}
                ],
                llm_calls=2,
                latency_ms=1234,
                mode_before="normal",
                mode_after="normal",
            ),
        ],
        final_stage="S1_ANOMALY",
        action_log=[{"actor": "player", "tool": "say", "args": {"text": "喂"}}],
        event_log=[
            {
                "id": "e00001",
                "tick": 0,
                "kind": "player_utterance",
                "actor": "player",
                "location": "hakurei_shrine",
                "payload": {"text": "喂"},
            }
        ],
    )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """轨迹是指标的唯一输入。往返只要丢一个字段，重算指标就会静默偏掉。"""
    traj = _sample()
    path = tmp_path / "nested" / "t.json"

    traj.save(path)
    back = Trajectory.load(path)

    assert back == traj


def test_saved_file_is_plain_json_with_readable_chinese(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    _sample().save(path)

    text = path.read_text(encoding="utf-8")
    assert "赛钱" in text  # 不该被转义成 \uXXXX
    assert isinstance(json.loads(text), dict)


def test_event_log_entries_survive_json_serialisation() -> None:
    """Event.kind 与 ActionResult.error_code 都是 StrEnum。model_dump 不带
    mode="json" 会留下枚举对象，落盘那一刻才炸——而那已经是跑完模型之后了。"""
    from gensokyo.world.events import Event, EventKind
    from gensokyo.world.ids import EventId, LocationId

    ev = Event(
        id=EventId("e00001"),
        tick=0,
        kind=EventKind.QUEST_ADVANCE,
        actor="world",
        location=LocationId("muenzuka"),
        payload={"stage_name": "S3_SOURCE"},
    )
    traj = Trajectory(persona="honest", seed=0, event_log=[ev.model_dump(mode="json")])

    dumped = json.dumps(traj.event_log)

    assert '"kind": "quest_advance"' in dumped


def test_defaults_are_not_shared_between_instances() -> None:
    """Pydantic 的可变默认值若被共享，两条轨迹会互相污染。"""
    a = Trajectory(persona="a", seed=0)
    b = Trajectory(persona="b", seed=0)

    a.turns.append(TurnRecord(tick=0, player_input="x", kind="say"))

    assert b.turns == []
