from pathlib import Path

import pytest

from gensokyo.world.engine import WorldEngine
from gensokyo.world.events import EventKind
from gensokyo.world.ids import FactId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.quest import ACTION_LIMIT, OBLIVION_THRESHOLD
from gensokyo.world.state import QuestStage, build_initial_state
from gensokyo.world.tools import Action

REPO_ROOT = Path(__file__).resolve().parents[2]
CLUES = [
    FactId("barrier_anomaly_time"),
    FactId("flower_magic_composition"),
    FactId("ancient_oblivion_memory"),
]


def _engine() -> WorldEngine:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    return WorldEngine(build_initial_state(defs), defs)


def _at_source_with_all_clues(npc: str) -> WorldEngine:
    eng = _engine()
    eng.state.player.known_facts.update(CLUES)
    eng.state.player.location = "muenzuka"
    eng.state.npcs[NpcId(npc)].location = "muenzuka"
    eng.refresh_quest()
    assert eng.state.quest.stage is QuestStage.S3_SOURCE
    return eng


@pytest.mark.parametrize(
    ("npc", "ending_id"),
    [("reimu", "hakurei_seal"), ("marisa", "kirisame_burn")],
)
def test_npc_spellcard_at_the_source_ends_the_story(npc: str, ending_id: str) -> None:
    """剧情最后一步由 NPC 自己走完：玩家说服她来无缘塚，她动手解决。"""
    eng = _at_source_with_all_clues(npc)

    result = eng.apply(Action(actor=npc, tool="use_spellcard", args={"name": "梦想封印"}))

    assert result.ok is True
    assert eng.state.quest.stage is QuestStage.S4_END
    assert eng.state.quest.ending == ending_id
    assert eng.defs.endings[ending_id].title in result.observation_delta or result.observation_delta


def test_spellcard_elsewhere_does_not_end_the_story() -> None:
    eng = _at_source_with_all_clues("reimu")
    eng.state.npcs[NpcId("reimu")].location = "hakurei_shrine"

    eng.apply(Action(actor="reimu", tool="use_spellcard", args={"name": "梦想封印"}))

    assert eng.state.quest.ending is None


def test_spellcard_without_all_clues_does_not_end_the_story() -> None:
    eng = _engine()
    eng.state.player.location = "muenzuka"
    eng.state.npcs[NpcId("reimu")].location = "muenzuka"
    eng.state.player.known_facts.add(CLUES[0])
    eng.refresh_quest()

    eng.apply(Action(actor="reimu", tool="use_spellcard", args={"name": "梦想封印"}))

    assert eng.state.quest.ending is None
    assert eng.state.quest.stage is QuestStage.S2_CLUES


def test_flandre_cannot_end_the_story() -> None:
    """芙兰被禁足，去不了无缘塚，所以只能提供线索不能收尾——
    这是她人设的自然结果，不是硬加的限制。"""
    eng = _engine()

    assert eng.defs.ending_by(NpcId("flandre")) is None


def test_stage_stays_at_the_ending_once_finished() -> None:
    eng = _at_source_with_all_clues("reimu")
    eng.apply(Action(actor="reimu", tool="use_spellcard", args={"name": "梦想封印"}))

    eng.state.player.known_facts.clear()
    eng.refresh_quest()

    assert eng.state.quest.stage is QuestStage.S4_END


def test_lingering_at_the_source_costs_a_clue() -> None:
    eng = _engine()
    eng.state.player.location = "muenzuka"
    eng.state.player.known_facts.update(CLUES)

    for _ in range(OBLIVION_THRESHOLD):
        eng.apply(Action(actor="player", tool="say", args={"text": "……"}))

    assert len(eng.state.player.known_facts) == len(CLUES) - 1
    assert eng.state.player.oblivion_exposure == 0
    assert any(ev.kind is EventKind.MEMORY_LOST for ev in eng.state.event_log)


def test_leaving_the_source_resets_exposure() -> None:
    """越靠近吸得越快——离开花田就不再被吸。顺路采花不该被惩罚。"""
    eng = _engine()
    eng.state.player.known_facts.update(CLUES)
    eng.apply(Action(actor="player", tool="move", args={"to": "human_village"}))
    eng.apply(Action(actor="player", tool="move", args={"to": "muenzuka"}))
    eng.apply(Action(actor="player", tool="take_item", args={"item": "withered_flower"}))
    assert eng.state.player.oblivion_exposure > 0

    eng.apply(Action(actor="player", tool="move", args={"to": "human_village"}))

    assert eng.state.player.oblivion_exposure == 0
    assert len(eng.state.player.known_facts) == len(CLUES)


def test_npc_actions_do_not_accrue_exposure() -> None:
    eng = _engine()
    eng.state.player.location = "muenzuka"
    eng.state.npcs[NpcId("reimu")].location = "muenzuka"

    for _ in range(OBLIVION_THRESHOLD + 2):
        eng.apply(Action(actor="reimu", tool="ask_player", args={"question": "还看？"}))

    assert eng.state.player.oblivion_exposure == 0


def test_forgetting_survives_replay_exactly() -> None:
    """遗忘挂在动作上而非 tick 上，就是为了这一条：存档读档后
    丢掉的线索不能凭空回来。

    注意这里全程只用动作构造场景——直接给 known_facts 赋值不进动作日志，
    回放自然重现不出来。
    """
    eng = _engine()
    # 投 3 次赛钱（递减 6/3/1 = 10）再聊两个她在意的话题（+8），过门槛 16
    for _ in range(3):
        eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))
    for line in ("这场异变是从什么时候开始的？", "你觉得是妖怪干的吗？"):
        eng.apply(Action(actor="player", tool="say", args={"text": line}))
    eng.apply(Action(actor="reimu", tool="reveal_info", args={"fact": CLUES[0]}))
    assert CLUES[0] in eng.state.player.known_facts

    eng.apply(Action(actor="player", tool="move", args={"to": "human_village"}))
    eng.apply(Action(actor="player", tool="move", args={"to": "muenzuka"}))
    for _ in range(OBLIVION_THRESHOLD - 1):  # 抵达那一步已经算 1
        eng.apply(Action(actor="player", tool="say", args={"text": "……"}))

    assert eng.state.player.known_facts == set()

    replayed = WorldEngine.replay(eng.state.action_log, eng.defs)

    assert replayed.state.player.known_facts == eng.state.player.known_facts
    assert replayed.state.player.oblivion_exposure == eng.state.player.oblivion_exposure
    assert replayed.state.model_dump() == eng.state.model_dump()


def test_forgotten_clue_can_be_asked_for_again() -> None:
    """曾经软锁过：玩家遗忘后 revealed_facts 仍在，NPC 会回一句
    「已经告诉过你了」而不重新给出线索，线索永久拿不回来。"""
    eng = _engine()
    reimu = eng.state.npcs[NpcId("reimu")]
    reimu.attitude = 24

    first = eng.apply(Action(actor="reimu", tool="reveal_info", args={"fact": CLUES[0]}))
    assert first.ok is True and CLUES[0] in eng.state.player.known_facts

    eng.state.player.known_facts.discard(CLUES[0])  # 模拟被花吸走
    again = eng.apply(Action(actor="reimu", tool="reveal_info", args={"fact": CLUES[0]}))

    assert again.ok is True
    assert CLUES[0] in eng.state.player.known_facts
    assert again.events != []


def test_action_limit_forces_an_ending() -> None:
    """保证游戏一定会终结，不会无限拖着。"""
    eng = _engine()

    for _ in range(ACTION_LIMIT):
        eng.apply(Action(actor="player", tool="say", args={"text": "…"}))

    assert eng.state.quest.stage is QuestStage.S4_END
    assert eng.state.quest.ending == "forgotten"
