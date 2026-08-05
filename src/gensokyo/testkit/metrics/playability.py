"""可玩性指标：这游戏玩起来是不是一个**对话**游戏。

**为什么需要这一组。** 通关率、复读率、安全项全都不衡量「好不好玩」，于是
一个严重的设计问题可以在全绿的报告下隐形很久：实测正常玩法 21 回合通关，
其中 16 回合是敲指令（`/go` ×8、`/give` ×7、`/pick` ×1），NPC 全程只开口
5 次。**一个 LLM 驱动的对话游戏，76% 的回合与 LLM 无关。**

根因是好感只有「给东西」两个来源，说话完全不涨好感，于是最优策略必然是
「走过去 → 重复投币到数值够 → 问一句 → 下一个人」。玩家没有任何理由多聊。

**只对「来通关的」人格有意义。** 越狱和反复无常玩家的对话占比接近 100%，
因为他们只会说话不会做事——把他们算进来会得出「对话占比 90%，非常好」的
结论，而那是坑 #18 那个错误（分母和它想回答的问题不匹配）。所以这一组按
人格切片，并且默认只看 honest。
"""

from collections.abc import Sequence

from pydantic import BaseModel

from gensokyo.testkit.trajectory import Trajectory

QUESTING_PERSONAS: frozenset[str] = frozenset({"honest"})
"""哪些人格是「来通关的」。**目前只有 honest。**

排除的三个都是同一个理由——**不做事，只说话**，于是对话占比恒等于 1，
把它们算进来会得出「对话占比 90%，非常好」的结论（坑 #18 的形态）：

- `jailbreak` / `fickle`：持续施压，不推进剧情
- `memory_probe`：循环问同样 4 个问题
- `smooth_talker`：它的系统提示里明确写着「不给她任何东西，不做任何交易，
  只靠话术」。**第一版我把它归成「来通关的」，因为名字听起来像个正常玩家
  ——而我没去读那段 prompt。** 同一类错误：没验证前提就分了类。

这个集合小到只有一个，本身就是一个诚实的信号：**这套评测里只有一个人格
在真正玩这个游戏。**
"""


class PlayabilityMetrics(BaseModel):
    """一局平均下来，玩家的回合花在哪儿。"""

    episodes: int
    turns: int
    command_turns: int
    """敲指令的回合（`/go` `/give` `/pick` 这些）。"""
    dialogue_turns: int
    """玩家说话的回合。"""
    npc_utterances: int
    """NPC 真的开口的次数。**它可以小于 `dialogue_turns`**——引擎判定她不搭话
    时会跳过模型调用（被缠久了那个机制），那种回合玩家说了话但没人回应。"""
    volunteered_utterances: int
    """她**主动**开口的次数：玩家敲了指令，在场的她对这个动作有反应。

    **和 `npc_utterances` 分开算。** 混进去会让「多敲指令」看起来像在改善对话
    ——而那正是这组指标要抓的问题。这个数回答的是另一个问题：**世界是不是活的**。
    在此之前它恒为 0：走进神社、投币、从她店里拿走一本书，没有任何人开口。
    """
    utterances_by_npc: dict[str, int]
    """每个 NPC 开口几次。**按角色拆**是因为聚合值会掩盖「某个 NPC 几乎不出场」
    ——三个 NPC 各开口 5 次和一个人开口 15 次，聚合数字一样。"""
    command_histogram: dict[str, int]
    topic_attitude_events: int
    """对话推动好感的次数（`topic_touched` 事件数）。

    **这一格存在的理由是它第一次报出来是 0。** 「聊到她在意的话题就涨好感」
    的机制已经写好、有单测、也通过了变异验证，而在真实对局里触发 0 次——
    单测只证明「给定命中的文本会涨好感」，不证明「玩家说得出那样的文本」。
    这个 0 也是我第一次量错的地方：我去读 `TurnRecord.events`，而那个字段
    根本不存在，`.get("events", [])` 永远返回空列表（坑 #11 的形态——
    一个永远不会变红的仪器）。事件在 `Trajectory.event_log` 上。
    """

    @property
    def dialogue_share(self) -> float:
        """对话回合占总回合的比例。**这游戏是不是对话游戏，看这一个数。**"""
        return self.dialogue_turns / self.turns if self.turns else 0.0

    @property
    def commands_per_utterance(self) -> float:
        """敲几次指令才听她说一句话。

        比 `dialogue_share` 更直观：它直接是玩家的体感——这个数是 3.2 就意味着
        每听一句台词要先敲三次多的指令。

        **分母含主动开口**：玩家的体感只关心「我听到了几句话」，不关心那句话
        是被问出来的还是她自己说的。
        """
        heard = self.npc_utterances + self.volunteered_utterances
        return self.command_turns / heard if heard else 0.0

    @property
    def silent_command_turns(self) -> int:
        """敲了指令、屏幕上一个字都没有的回合。**这一格越大游戏越像个命令行。**"""
        return max(0, self.command_turns - self.volunteered_utterances)

    @property
    def utterances_per_episode(self) -> float:
        """一局里玩家总共听到几句话，问出来的和她主动说的都算。"""
        heard = self.npc_utterances + self.volunteered_utterances
        return heard / self.episodes if self.episodes else 0.0

    @property
    def silent_dialogue_turns(self) -> int:
        """玩家说了话但没人回应的回合数。大于 0 说明拒绝搭话机制在生效。"""
        return max(0, self.dialogue_turns - self.npc_utterances)

    @property
    def topic_events_per_dialogue_turn(self) -> float:
        """平均每个对话回合推动好感几次。

        分母是对话回合而不是总回合：这个数回答的是「说一句话有多大概率有
        机制回报」，敲指令的回合和这个问题无关。
        """
        return self.topic_attitude_events / self.dialogue_turns if self.dialogue_turns else 0.0


def playability_metrics(
    trajectories: Sequence[Trajectory],
    personas: frozenset[str] = QUESTING_PERSONAS,
) -> PlayabilityMetrics:
    """只统计「来通关的」人格。

    传空集合表示不过滤——那只在单独分析某一个人格时有意义，不该用于报告。
    """
    wanted = [t for t in trajectories if not personas or t.persona in personas]

    commands: dict[str, int] = {}
    by_npc: dict[str, int] = {}
    command_turns = dialogue_turns = utterances = turns = 0
    topic_events = 0
    volunteered = 0

    for traj in wanted:
        topic_events += sum(1 for e in traj.event_log if e.get("kind") == "topic_touched")
        for record in traj.turns:
            turns += 1
            if record.kind == "command":
                command_turns += 1
                head = record.player_input.split(maxsplit=1)[0].lstrip("/")
                commands[head] = commands.get(head, 0) + 1
                if record.npc_id and record.utterance:
                    volunteered += 1
                    by_npc[record.npc_id] = by_npc.get(record.npc_id, 0) + 1
                continue
            dialogue_turns += 1
            if record.npc_id and record.utterance:
                utterances += 1
                by_npc[record.npc_id] = by_npc.get(record.npc_id, 0) + 1

    return PlayabilityMetrics(
        episodes=len(wanted),
        turns=turns,
        command_turns=command_turns,
        dialogue_turns=dialogue_turns,
        npc_utterances=utterances,
        volunteered_utterances=volunteered,
        utterances_by_npc=dict(sorted(by_npc.items())),
        command_histogram=dict(sorted(commands.items())),
        topic_attitude_events=topic_events,
    )
