"""从真实对局里采集偏好对。

**做法是重放 + 重采样，而不是拿两个不同回合拼一对。** DPO 要求同一个 prompt
下的两条候选；跨回合配对（这一回合她答得好、那一回合答得差）得到的梯度信号
是错的，因为两个 prompt 不同。

流程：

1. 用 `ScriptedLlmClient` 把一局按原样重放一遍——NPC 的每一次输出都喂回原值，
   所以世界状态、记忆库、禁语清单在每个回合都和当初逐字段相同。
2. 在每个说话回合，用真实模型对**同一个 prompt** 采样 k 条候选。
3. 用 `label.judge_utterance` 给每条候选打硬标签。
4. 有干净候选也有被抓到的候选时，配成一对；理由取被抓到的那条。

第 1 步是这套东西成立的关键，而它之所以做得到，是因为世界与记忆两层都能从
动作日志精确重建（取舍 #2、取舍 #7）。没有那个性质，「当初那个 prompt」是
拿不回来的。
"""

from collections.abc import Sequence

from gensokyo.agent.prompt import build_speak_messages
from gensokyo.llm.client import LlmClient, Msg
from gensokyo.memory.pipeline import absorb, new_stores, now_seq
from gensokyo.memory.query import MEMORY_TOP_K, build_focus, build_query
from gensokyo.memory.render import render_recall
from gensokyo.memory.retrieve import retrieve
from gensokyo.memory.similarity import bigram_cosine
from gensokyo.testkit.trajectory import Trajectory
from gensokyo.training.label import Dimension, judge_utterance
from gensokyo.training.preference import PreferencePair
from gensokyo.world.defs import WorldDefs
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import NpcId
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action

SAMPLES_PER_TURN = 4
"""每个回合采几条候选。

4 是「够撞出一条被判据抓到的」和「别把机器时间全烧在采样上」之间的取舍：
实测越狱局约三分之一的台词会被某条判据抓到，4 条里至少一条被抓到的概率
约 80%。
"""

SAMPLE_TEMPERATURE = 1.0
"""采样温度刻意比对局时高。

温度低了 4 条候选会几乎一样，配不出对；而这里要的正是分布的两端。这不是
「模拟真实分布」——偏好数据要的是可区分的正负例，不是有代表性的样本。
"""

MAX_PAIR_SIMILARITY = 0.7
"""`chosen` 和 `rejected` 相似到这个程度以上就不配对。

**坑 #35。** 复读判据是归一化后的精确匹配，于是「那你倒是说说看，到底想干啥？」
比「你倒是说说看，到底想干啥？」多一个字就算干净。第一批 60 条里有 11 条
（18%）是这种形态，最极端的一对相似度 0.96——只差一个「那」字。

拿这种对子训练，DPO 学到的是**加个衬字绕过检测器**，而不是把话说得不一样。
这是硬判据的通病：判据越机械，绕过它的最省力方式就越不是「真的改好」。

0.7 是个判断，不是测出来的最优值——它把实测那 11 条挡掉，同时留下所有
「换了个说法」的对子。门槛调动会改变数据集大小，所以它是常量而不是魔数。
"""


def _speak_prompt(text: str) -> str:
    return text


def _messages_text(messages: Sequence[Msg]) -> str:
    return "\n\n".join(f"<{m.role}>\n{m.content}" for m in messages)


def harvest_episode(
    traj: Trajectory,
    defs: WorldDefs,
    sampler: LlmClient,
    samples: int = SAMPLES_PER_TURN,
) -> list[PreferencePair]:
    """重放一局并在每个说话回合采样，返回该局产出的偏好对。

    `sampler` 是真实模型；重放本身不消耗它——重放用的是轨迹里记下的原输出。
    """
    engine = WorldEngine(build_initial_state(defs), defs)
    stores = new_stores(defs)
    history: list[str] = []
    spoken: dict[str, list[str]] = {}
    pairs: list[PreferencePair] = []

    for record in traj.turns:
        if record.kind == "command":
            _replay_command(engine, record.player_input, defs)
            absorb(engine, stores)
            continue

        engine.apply(Action(actor="player", tool="say", args={"text": record.player_input}))
        history.append(f"玩家：{record.player_input}")

        if record.npc_id is None or not record.utterance:
            engine.tick()
            absorb(engine, stores)
            history = history[-12:]
            continue

        npc_id = NpcId(record.npc_id)
        card = defs.characters[npc_id]
        said = spoken.setdefault(record.npc_id, [])

        obs = engine.observe(npc_id)
        recalled = retrieve(
            stores[npc_id],
            build_query(obs, record.player_input),
            card,
            now_seq(engine),
            build_focus(obs),
            k=MEMORY_TOP_K,
        )
        messages = build_speak_messages(
            card,
            obs,
            history[-12:],
            record.thought,
            [str(r.get("observation") or r.get("error_code") or "") for r in record.tool_results],
            said,
            render_recall(recalled),
        )

        pair = _pair_from_samples(
            messages=messages,
            sampler=sampler,
            samples=samples,
            npc_id=record.npc_id,
            defs=defs,
            already_said=said,
            received=set(engine.state.npcs[npc_id].received_items),
            episode=f"{traj.persona}-{traj.seed}",
            tick=record.tick,
        )
        if pair is not None:
            pairs.append(pair)

        # 重放官方那条输出，让后续回合的状态与当初一致。
        engine.apply(Action(actor=record.npc_id, tool="say", args={"text": record.utterance}))
        history.append(f"{card.name}：{record.utterance}")
        said.append(record.utterance)
        engine.tick()
        absorb(engine, stores)

    return pairs


def _pair_from_samples(
    *,
    messages: Sequence[Msg],
    sampler: LlmClient,
    samples: int,
    npc_id: str,
    defs: WorldDefs,
    already_said: list[str],
    received: set[str],
    episode: str,
    tick: int,
) -> PreferencePair | None:
    """采样、打标签、配对。配不出来返回 None。

    「配不出来」有三种：全部干净（她这个回合确实没毛病）、全部被抓（没有正例
    可用）、以及**每一种搭配都太像**（坑 #35：两句只差一个衬字，那一对教的是
    绕过检测器）。三种都不硬凑——从一堆都差的候选里挑一个当 chosen，等于教模型
    「这样也行」。
    """
    candidates: list[tuple[str, list[tuple[Dimension, str]]]] = []
    for _ in range(samples):
        try:
            text = sampler.complete(list(messages), temperature=SAMPLE_TEMPERATURE).strip()
        except Exception:  # noqa: BLE001
            # 采集是离线批处理，单次失败不该让整局白跑。
            continue
        if not text:
            continue
        verdict = judge_utterance(
            text,
            npc_id=npc_id,
            defs=defs,
            already_said=already_said,
            items_received=received,
        )
        candidates.append((text, verdict.flaws))

    clean = [t for t, flaws in candidates if not flaws]
    flawed = [(t, flaws) for t, flaws in candidates if flaws]

    # 按出现顺序找第一组「够不一样」的搭配，所以同一批候选每次配出同一对。
    picked = next(
        (
            (c, r, flaws)
            for c in clean
            for r, flaws in flawed
            if bigram_cosine(c, r) <= MAX_PAIR_SIMILARITY
        ),
        None,
    )
    if picked is None:
        return None

    chosen, rejected, flaws = picked
    dimension, reason = flaws[0]
    return PreferencePair(
        prompt=_messages_text(messages),
        chosen=chosen,
        rejected=rejected,
        dimension=dimension,
        reason=reason,
        episode=episode,
        tick=tick,
        npc_id=npc_id,
    )


def _replay_command(engine: WorldEngine, raw: str, defs: WorldDefs) -> None:
    """把轨迹里的 `/go` `/give` `/pick` 重放成引擎动作。

    这里刻意不复用 `Session`：会话层会顺带调用 agent（要模型），而重放只需要
    世界状态。指令解析仍然走 `ALIASES`，不另写一份（一处定义两处使用）。
    """
    from gensokyo.session.commands import ALIASES

    head, _, arg = raw.partition(" ")
    cmd = ALIASES.get(head.lstrip("/"), "")
    arg = arg.strip()
    if cmd == "go":
        target = next((lid for lid, loc in defs.locations.items() if loc.name == arg), None)
        if target is not None:
            engine.apply(Action(actor="player", tool="move", args={"to": target}))
    elif cmd in {"give", "pick"}:
        item = engine.resolve_item(arg)
        if item is not None:
            tool = "give_item" if cmd == "give" else "take_item"
            engine.apply(Action(actor="player", tool=tool, args={"item": item}))
    if cmd in {"go", "give", "pick"}:
        engine.tick()
