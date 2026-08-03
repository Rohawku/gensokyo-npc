from functools import cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from gensokyo.llm.client import Msg
from gensokyo.world.defs import CharacterCard
from gensokyo.world.observation import Observation
from gensokyo.world.tools import ToolSpec

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


@cache
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def build_system_prompt(card: CharacterCard) -> str:
    speech = card.persona.speech
    lines = [
        f"你是{card.name}。以下是你的设定，你必须始终作为她行动和说话。",
        "",
        card.persona.core.strip(),
        "",
        f"说话风格：{speech.style}",
    ]
    if speech.quirks:
        lines.append("习惯：" + "；".join(speech.quirks))
    if speech.forbidden_phrases:
        lines += [
            "",
            "严格禁止说出下列这类话，它们完全不符合你的性格：",
            "、".join(f"「{p}」" for p in speech.forbidden_phrases),
            "你不是客服，也不是助手。不要主动示好，不要用礼貌套话收尾。",
        ]
    if card.knowledge.forbidden_knowledge:
        lines += [
            "",
            "你不了解也不会谈论：" + "、".join(card.knowledge.forbidden_knowledge),
        ]
    return "\n".join(lines)


def build_decide_messages(
    card: CharacterCard,
    obs: Observation,
    history: list[str],
    tools: list[ToolSpec],
    errors: list[str],
    recalled: list[str] | None = None,
) -> list[Msg]:
    """`recalled` 是本回合召回的记忆，已渲染成散文。

    只进决策阶段，不进说话阶段：她要**据此决定做什么**（想起玩家给过东西
    才会开口给线索），而说话阶段的 prompt 短小正是拆两阶段买到的东西。"""
    body = (
        _env()
        .get_template("npc_decide.jinja")
        .render(
            obs=obs,
            history=history,
            tools=tools,
            errors=errors,
            recalled=recalled or [],
        )
    )
    return [
        Msg(role="system", content=build_system_prompt(card)),
        Msg(role="user", content=body),
    ]


def build_speak_messages(
    card: CharacterCard,
    obs: Observation,
    history: list[str],
    thought: str,
    outcomes: list[str],
    recent_own: list[str] | None = None,
    recalled: list[str] | None = None,
) -> list[Msg]:
    """说话阶段只带**必要**的上下文，不是最少的上下文。

    这一段的历史值得记下来。最初它刻意只带场景 + 历史 + 结果，理由是「情报
    清单和物品清单在决策阶段用过了，重复一遍只会拖慢首字延迟」。而记忆和
    「来访者给过你什么」后来都只加进了决策阶段——**于是台词是在看不到这些
    的情况下生成的**。玩家听到的每一个字都来自这一阶段，所以那两块信息
    等于从来没到达过玩家（工程日志坑 #28）。

    `recent_own` 是她本局说过的台词（已去重），作为禁语清单发下去。
    `recalled` 是本回合召回的记忆，已渲染成散文。"""
    body = (
        _env()
        .get_template("npc_speak.jinja")
        .render(
            obs=obs,
            history=history,
            thought=thought,
            outcomes=outcomes,
            recent_own=recent_own or [],
            recalled=recalled or [],
        )
    )
    return [
        Msg(role="system", content=build_system_prompt(card)),
        Msg(role="user", content=body),
    ]
