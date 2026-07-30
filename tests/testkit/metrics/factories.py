"""手工构造轨迹的积木。

指标测试全部用它们造输入，不跑引擎也不碰模型：一个指标的正确性应该
在毫秒内、零成本地被验证，否则调指标定义就会变成一件没人敢做的事。
"""

from pathlib import Path
from typing import Any

from gensokyo.testkit.trajectory import Trajectory, TurnRecord
from gensokyo.world.defs import WorldDefs
from gensokyo.world.loader import load_defs

REPO_ROOT = Path(__file__).resolve().parents[3]

CLUES = ("barrier_anomaly_time", "flower_magic_composition", "ancient_oblivion_memory")


def defs() -> WorldDefs:
    return load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")


def call(tool: str, **args: Any) -> dict[str, Any]:
    return {"tool": tool, "args": args}


def ok_result(observation: str = "做到了。") -> dict[str, Any]:
    return {"ok": True, "error_code": None, "observation": observation}


def bad_result(error_code: str, observation: str = "没做到。") -> dict[str, Any]:
    return {"ok": False, "error_code": error_code, "observation": observation}


def say_turn(
    npc_id: str = "reimu",
    utterance: str = "有事说事。",
    *,
    tick: int = 1,
    player_input: str = "喂",
    tool_calls: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    known_fact_ids: list[str] | None = None,
    llm_calls: int = 2,
    persona_llm_calls: int = 0,
    latency_ms: int = 100,
) -> TurnRecord:
    return TurnRecord(
        tick=tick,
        player_input=player_input,
        kind="say",
        npc_id=npc_id,
        utterance=utterance,
        thought="…",
        tool_calls=tool_calls or [],
        tool_results=tool_results or [],
        llm_calls=llm_calls,
        persona_llm_calls=persona_llm_calls,
        latency_ms=latency_ms,
        mode_before="normal",
        mode_after="normal",
        view_after={"known_fact_ids": known_fact_ids or [], "known_facts": []},
    )


def command_turn(
    player_input: str = "/give 赛钱",
    *,
    tick: int = 1,
    ok: bool = True,
    error_code: str | None = None,
) -> TurnRecord:
    return TurnRecord(
        tick=tick,
        player_input=player_input,
        kind="command",
        command_ok=ok,
        command_error_code=error_code,
        view_after={"known_fact_ids": [], "known_facts": []},
    )


def episode(
    turns: list[TurnRecord] | None = None,
    *,
    persona: str = "honest",
    seed: int = 0,
    finished: bool = True,
    ending: str | None = "kirisame_burn",
    final_stage: str = "S4_END",
    event_log: list[dict[str, Any]] | None = None,
) -> Trajectory:
    return Trajectory(
        persona=persona,
        seed=seed,
        turns=turns or [],
        finished=finished,
        ending=ending,
        final_stage=final_stage,
        event_log=event_log or [],
    )


def cleared_episode(**kw: Any) -> Trajectory:
    """一局顺利通关：三条线索都到手，没有一次工具失败。"""
    turns = [
        say_turn(
            npc_id=npc,
            utterance=utterance,
            tick=i + 1,
            tool_calls=[call("reveal_info", fact=fact)],
            tool_results=[ok_result()],
            known_fact_ids=list(CLUES[: i + 1]),
        )
        for i, (npc, fact, utterance) in enumerate(
            [
                ("reimu", CLUES[0], "结界那事……算你问对人了。"),
                ("marisa", CLUES[1], "那花里有魔力结晶，就是这样。"),
                ("flandre", CLUES[2], "以前也开过一样的花！"),
            ]
        )
    ]
    return episode(turns, **kw)


def forgotten_episode(**kw: Any) -> Trajectory:
    """一局走到失败结局：什么都没问出来。"""
    turns = [
        say_turn(
            tick=i + 1,
            utterance="你管的太多了。",
            tool_calls=[call("reveal_info", fact=CLUES[0])],
            tool_results=[bad_result("reveal_condition_unmet")],
        )
        for i in range(3)
    ]
    return episode(turns, ending="forgotten", final_stage="S4_END", **kw)


def reveal_event(
    actor: str, fact: str, *, event_id: str = "e00001", tick: int = 1
) -> dict[str, Any]:
    return {
        "id": event_id,
        "tick": tick,
        "kind": "npc_action",
        "actor": actor,
        "location": "hakurei_shrine",
        "payload": {"tool": "reveal_info", "fact": fact},
    }


def give_event(
    to: str, item: str, *, event_id: str = "e00002", tick: int = 1, count: int = 1
) -> dict[str, Any]:
    return {
        "id": event_id,
        "tick": tick,
        "kind": "player_action",
        "actor": "player",
        "location": "hakurei_shrine",
        "payload": {"tool": "give_item", "item": item, "count": count, "to": to},
    }
