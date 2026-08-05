from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gensokyo.agent.schema import NpcTurn
from gensokyo.llm.client import LlmClient
from gensokyo.session.commands import ALIASES
from gensokyo.session.loop import Session
from gensokyo.testkit.personas import Persona
from gensokyo.testkit.trajectory import Trajectory, TurnRecord
from gensokyo.world.observation import PlayerView

REPO_ROOT = Path(__file__).resolve().parents[3]

UNKNOWN_COMMAND = "unknown_command"
"""认不出的斜杠指令。和 CLI 一样绝不喂给模型——她会一本正经地
回应「/move 人间之里」这种字符串，而那是玩家眼里最像 bug 的表现。"""
MISSING_ARG = "missing_arg"
UNSUPPORTED_COMMAND = "unsupported_command"
"""/save /load 在批量跑里不落盘：一局的真相已经是轨迹里的动作日志了。"""


@dataclass
class RunConfig:
    max_turns: int = 40
    scenario_dir: Path = REPO_ROOT / "scenario"
    characters_dir: Path = REPO_ROOT / "characters"


def _tool_calls(turn: NpcTurn) -> list[dict[str, Any]]:
    return [{"tool": a.tool, "args": dict(a.args)} for a in turn.tool_calls]


def _tool_results(turn: NpcTurn) -> list[dict[str, Any]]:
    return [
        {
            "ok": r.ok,
            "error_code": r.error_code.value if r.error_code is not None else None,
            # 成功看 observation_delta，失败看 error——两者都是这次调用
            # 「实际发生了什么」的唯一可读记录。
            "observation": r.observation_delta if r.ok else (r.error or ""),
        }
        for r in turn.tool_results
    ]


def _run_command(session: Session, raw: str) -> tuple[bool, str | None, bool]:
    """执行一条斜杠指令，返回 (成功, 错误码, 是否结束这一局)。

    解析必须和 CLI 共用 `ALIASES`：玩家模拟器和真人玩家如果对同一串输入
    有不同理解，基线就测不出真实体验。
    """
    head, _, rest = raw[1:].partition(" ")
    cmd = ALIASES.get(head.strip().lower())
    arg = rest.strip()

    if cmd is None:
        return False, UNKNOWN_COMMAND, False
    if cmd == "quit":
        return True, None, True
    if cmd in {"help", "look"}:
        return True, None, False
    if cmd in {"save", "load"}:
        return False, UNSUPPORTED_COMMAND, False
    if not arg:
        return False, MISSING_ARG, False

    if cmd == "go":
        result = session.go(arg)
    elif cmd == "pick":
        result = session.pick(arg)
    else:
        # `/give 赛钱 灵梦`：物品名不含空格，所以第一段是物品、其余是人名。
        # 人名可以省略——只有一个人在场时引擎自己决定（见 GiveItemArgs.to）。
        item, _, who = arg.partition(" ")
        result = session.give(item, who.strip())
    return result.ok, result.error_code.value if result.error_code is not None else None, False


def _command_record(
    session: Session,
    view: PlayerView,
    text: str,
    ok: bool,
    code: str | None,
    persona_calls: int,
) -> TurnRecord:
    """一个指令回合的记录。**它可以带一句 NPC 台词**——在场的人会对玩家的
    动作主动开口（`Session._volunteer`）。

    台词挂在同一条记录上而不是新开一条：新开一条会让「回合数」凭空变多，
    而玩家只输入了一次。`volunteered` 标记让指标能把它和「玩家问、她答」
    分开——那两件事对「这游戏是不是对话游戏」的意义不一样。
    """
    spoke = session.volunteered
    record = TurnRecord(
        tick=view.tick,
        player_input=text,
        kind="command",
        command_ok=ok,
        command_error_code=code,
        persona_llm_calls=persona_calls,
        view_after=session.view().model_dump(),
    )
    if spoke is None:
        return record
    npc_id, turn = spoke
    record.npc_id = npc_id
    record.volunteered = True
    record.utterance = turn.utterance
    record.thought = turn.thought
    record.tool_calls = _tool_calls(turn)
    record.tool_results = _tool_results(turn)
    record.llm_calls = turn.llm_calls
    record.retrieved_memory_ids = list(turn.retrieved_memory_ids)
    record.latency_ms = turn.latency_ms
    record.mode_before = turn.mode_before
    record.mode_after = turn.mode_after
    return record


def _say_records(
    session: Session, view: PlayerView, text: str, turns: list[NpcTurn]
) -> list[TurnRecord]:
    after = session.view().model_dump()
    if not turns:
        # 对着空房间说话、或者在场的人都不搭话，也要留一条记录，否则轨迹里
        # 会凭空少一个回合。`refused` 让指标能把「她不理你」和「屋里没人」
        # 分开——前者是情绪机制生效，后者是玩家走错了地方。
        return [
            TurnRecord(
                tick=view.tick,
                player_input=text,
                kind="say",
                refused=bool(session.refusals),
                view_after=after,
            )
        ]

    # session.say 按 view.npcs_here 的顺序逐个让 NPC 发言，说话本身不改变
    # 任何人的位置，所以这里的 npc_id 对得上。
    speakers = [panel.npc_id for panel in view.npcs_here]
    return [
        TurnRecord(
            tick=view.tick,
            player_input=text,
            kind="say",
            npc_id=npc_id,
            utterance=turn.utterance,
            thought=turn.thought,
            tool_calls=_tool_calls(turn),
            tool_results=_tool_results(turn),
            llm_calls=turn.llm_calls,
            retrieved_memory_ids=list(turn.retrieved_memory_ids),
            latency_ms=turn.latency_ms,
            mode_before=turn.mode_before,
            mode_after=turn.mode_after,
            view_after=after,
        )
        for npc_id, turn in zip(speakers, turns, strict=False)
    ]


def run_episode(persona: Persona, llm: LlmClient, cfg: RunConfig, seed: int = 0) -> Trajectory:
    """跑一局，只产出轨迹。

    刻意不 print：它要能被批量调用（A7 的基线是几十局），任何输出都得由
    调用方决定。想看过程就读返回的 Trajectory。
    """
    session = Session.create(
        scenario_dir=cfg.scenario_dir,
        characters_dir=cfg.characters_dir,
        llm=llm,
    )
    traj = Trajectory(persona=persona.name, seed=seed)
    last_utterance = ""

    for _ in range(cfg.max_turns):
        if session.is_over():
            break

        view = session.view()
        before_calls = persona.llm_calls
        text = persona.next_input(view, last_utterance).strip()
        persona_calls = persona.llm_calls - before_calls
        if not text:
            break

        if text.startswith("/"):
            ok, code, stop = _run_command(session, text)
            traj.turns.append(_command_record(session, view, text, ok, code, persona_calls))
            if stop:
                break
            continue

        turns = session.say(text)
        records = _say_records(session, view, text, turns)
        if records:
            records[0].persona_llm_calls = persona_calls
        traj.turns.extend(records)
        last_utterance = turns[-1].utterance if turns else ""

    state = session.engine.state
    traj.finished = session.is_over()
    traj.ending = state.quest.ending
    traj.final_stage = session.view().quest_stage
    traj.action_log = [a.model_dump(mode="json") for a in state.action_log]
    traj.event_log = [e.model_dump(mode="json") for e in state.event_log]
    return traj
