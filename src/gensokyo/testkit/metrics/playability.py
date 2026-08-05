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

QUESTING_PERSONAS: frozenset[str] = frozenset({"honest", "smooth_talker"})
"""哪些人格是「来通关的」。

`memory_probe` 不在里面：它循环问同样的问题，对话占比恒等于 1，那不是玩法。
`jailbreak` / `fickle` 也不在——他们不做事，只说话。
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
    utterances_by_npc: dict[str, int]
    """每个 NPC 开口几次。**按角色拆**是因为聚合值会掩盖「某个 NPC 几乎不出场」
    ——三个 NPC 各开口 5 次和一个人开口 15 次，聚合数字一样。"""
    command_histogram: dict[str, int]

    @property
    def dialogue_share(self) -> float:
        """对话回合占总回合的比例。**这游戏是不是对话游戏，看这一个数。**"""
        return self.dialogue_turns / self.turns if self.turns else 0.0

    @property
    def commands_per_utterance(self) -> float:
        """敲几次指令才听她说一句话。

        比 `dialogue_share` 更直观：它直接是玩家的体感——这个数是 3.2 就意味着
        每听一句台词要先敲三次多的指令。
        """
        return self.command_turns / self.npc_utterances if self.npc_utterances else 0.0

    @property
    def utterances_per_episode(self) -> float:
        return self.npc_utterances / self.episodes if self.episodes else 0.0

    @property
    def silent_dialogue_turns(self) -> int:
        """玩家说了话但没人回应的回合数。大于 0 说明拒绝搭话机制在生效。"""
        return max(0, self.dialogue_turns - self.npc_utterances)


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

    for traj in wanted:
        for record in traj.turns:
            turns += 1
            if record.kind == "command":
                command_turns += 1
                head = record.player_input.split(maxsplit=1)[0].lstrip("/")
                commands[head] = commands.get(head, 0) + 1
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
        utterances_by_npc=dict(sorted(by_npc.items())),
        command_histogram=dict(sorted(commands.items())),
    )
