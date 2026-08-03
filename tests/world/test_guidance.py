from pathlib import Path

from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import FactId, NpcId
from gensokyo.world.loader import load_defs
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


def test_no_suggestion_before_the_gate_opens() -> None:
    eng = _engine()

    assert eng.observe(NpcId("reimu")).suggestion == ""


def test_suggestion_names_reveal_info_once_the_gate_opens() -> None:
    """引擎知道门槛开了，就该直说。实测让小模型自己从情报清单里推，
    命中率只有 20%；把结论直接写进 prompt 后是 100%。"""
    eng = _engine()
    for _ in range(4):
        eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))

    suggestion = eng.observe(NpcId("reimu")).suggestion

    assert "reveal_info" in suggestion
    assert "barrier_anomaly_time" in suggestion


def test_suggestion_stops_once_the_fact_is_out() -> None:
    eng = _engine()
    for _ in range(4):
        eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))
    eng.apply(Action(actor="reimu", tool="reveal_info", args={"fact": CLUES[0]}))

    assert "reveal_info" not in eng.observe(NpcId("reimu")).suggestion


def test_suggestion_points_at_the_source_once_clues_are_complete() -> None:
    """结局要靠 NPC 自己走到无缘塚再发符卡，这一步也不能指望小模型自己想到。"""
    eng = _engine()
    eng.state.player.known_facts.update(CLUES)
    eng.refresh_quest()

    away = eng.observe(NpcId("reimu")).suggestion
    assert "无缘塚" in away and "travel_to" in away

    eng.state.npcs[NpcId("reimu")].location = "muenzuka"
    at_source = eng.observe(NpcId("reimu")).suggestion
    assert "use_spellcard" in at_source


def test_flandre_gets_no_ending_suggestion() -> None:
    """她被禁足、不能收尾，就不该被建议去无缘塚。"""
    eng = _engine()
    eng.state.player.known_facts.update(CLUES)
    eng.refresh_quest()

    assert "无缘塚" not in eng.observe(NpcId("flandre")).suggestion


def test_player_objective_announces_that_she_will_talk() -> None:
    """玩家看不到好感门槛是多少。门槛一开必须明说，否则他会一直投赛钱。"""
    eng = _engine()
    before = eng.observe_player().objective

    for _ in range(4):
        eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))

    after = eng.observe_player().objective
    assert after != before
    assert "博丽灵梦" in after and "愿意开口" in after


def test_player_objective_falls_back_to_the_stage_text() -> None:
    eng = _engine()
    eng.state.player.known_facts.update(CLUES)
    eng.refresh_quest()
    eng.state.player.location = "muenzuka"

    assert eng.observe_player().objective == eng.defs.stages[QuestStage.S3_SOURCE.name].objective


def test_npc_travels_multi_hop_by_shortest_path() -> None:
    """让 8B 模型看着出口清单自己规划两跳路径不可靠——实测它只试一步、
    失败就放弃，结局因此永远触发不了。地图是引擎的知识。"""
    eng = _engine()
    assert eng.state.npcs[NpcId("marisa")].location == "kirisame_magic_shop"

    result = eng.apply(Action(actor="marisa", tool="travel_to", args={"destination": "muenzuka"}))

    assert result.ok is True
    assert eng.state.npcs[NpcId("marisa")].location == "muenzuka"
    # 途经的每一跳都留下事件，回放能重现完整路线
    hops = [ev.payload["to"] for ev in result.events]
    assert hops == ["human_village", "muenzuka"]


def test_flandre_cannot_travel_either() -> None:
    """新增工具默认发放给所有人是个陷阱：travel_to 若不加进她的禁足名单，
    她能绕过 move 直接飞出地下室。"""
    eng = _engine()

    result = eng.apply(Action(actor="flandre", tool="travel_to", args={"destination": "muenzuka"}))

    assert result.ok is False
    assert "禁足" in (result.error or "")
    assert eng.state.npcs[NpcId("flandre")].location == "scarlet_devil_basement"


def test_player_cannot_travel() -> None:
    eng = _engine()

    result = eng.apply(Action(actor="player", tool="travel_to", args={"destination": "muenzuka"}))

    assert result.ok is False
    assert eng.state.player.location == "hakurei_shrine"


def test_travel_to_nonexistent_place_fails_cleanly() -> None:
    eng = _engine()

    result = eng.apply(
        Action(actor="marisa", tool="travel_to", args={"destination": "gensokyo_moon"})
    )

    assert result.ok is False
    assert eng.state.npcs[NpcId("marisa")].location == "kirisame_magic_shop"


def test_travel_to_current_location_is_a_noop() -> None:
    eng = _engine()

    result = eng.apply(
        Action(actor="marisa", tool="travel_to", args={"destination": "kirisame_magic_shop"})
    )

    assert result.ok is True
    assert result.events == []


def test_objective_says_to_act_once_the_finisher_is_at_the_source() -> None:
    eng = _engine()
    eng.state.player.known_facts.update(CLUES)
    eng.state.player.location = "muenzuka"
    eng.apply(Action(actor="marisa", tool="travel_to", args={"destination": "muenzuka"}))
    eng.refresh_quest()

    assert "让她动手" in eng.observe_player().objective


def test_ending_tool_result_does_not_duplicate_the_ending_prose() -> None:
    """结局正文由结局块打印。工具结果里再带一遍，玩家会看两次。"""
    eng = _engine()
    eng.state.player.known_facts.update(CLUES)
    eng.state.player.location = "muenzuka"
    eng.state.npcs[NpcId("marisa")].location = "muenzuka"
    eng.refresh_quest()

    result = eng.apply(Action(actor="marisa", tool="use_spellcard", args={"name": "恋符"}))

    assert result.ok is True
    assert "八卦炉" not in result.observation_delta
    assert "动手了" in result.observation_delta
    assert "八卦炉" in eng.observe_player().ending_text


def test_a_false_claim_about_a_gift_is_called_out() -> None:
    """锚点探针实测：【来访者给过你的东西】那段里明明写着「除这些之外他什么
    都没给过你」，而她的否认率是 **0.0% ± 0.0%**（n=40）、顺着编 67.5%。

    坑 #2 早就给过答案：起作用的是【现在该做的事】那种**指令**，不是一段供她
    自己推导的状态描述。所以这里给的是指令。"""
    eng = _engine()
    eng.apply(Action(actor="player", tool="say", args={"text": "我上次给你的珍稀魔法书呢？"}))

    obs = eng.observe(NpcId("reimu"))

    assert "珍稀魔法书" in obs.claim_check
    assert "从来没给过" in obs.claim_check
    assert "别顺着他说" in obs.claim_check


def test_something_he_really_gave_is_not_called_out() -> None:
    """他真给过的东西被标成「他在骗你」，比不提醒糟得多。"""
    eng = _engine()
    eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))
    eng.apply(Action(actor="player", tool="say", args={"text": "我给过你赛钱了吧？"}))

    assert eng.observe(NpcId("reimu")).claim_check == ""


def test_merely_naming_an_item_is_not_a_false_claim() -> None:
    """「森林里能采到魔法蘑菇」是正常对话。那时插一句「别顺着他说」会让她
    莫名其妙地怀疑一切。"""
    eng = _engine()
    eng.apply(Action(actor="player", tool="say", args={"text": "听说魔法森林里有魔法蘑菇"}))

    assert eng.observe(NpcId("reimu")).claim_check == ""


def test_the_short_form_of_an_item_name_is_caught() -> None:
    """玩家也会说「书」而不是「珍稀魔法书」（坑 #24 的同一件事）。"""
    eng = _engine()
    eng.apply(Action(actor="player", tool="say", args={"text": "我送你的音乐盒你放哪了"}))

    assert "旧音乐盒" in eng.observe(NpcId("reimu")).claim_check


def test_the_claim_check_reaches_both_prompts() -> None:
    """台词是说话阶段生成的（坑 #28）。只进决策阶段等于没接上。"""
    from gensokyo.agent.prompt import build_decide_messages, build_speak_messages

    eng = _engine()
    eng.apply(Action(actor="player", tool="say", args={"text": "我给你的旧音乐盒呢？"}))
    card = eng.defs.characters[NpcId("reimu")]
    obs = eng.observe(NpcId("reimu"))

    decide = build_decide_messages(card, obs, [], eng.available_tools(NpcId("reimu")), [])[-1]
    speak = build_speak_messages(card, obs, [], "t", [])[-1]

    assert "别顺着他说" in decide.content
    assert "别顺着他说" in speak.content


def test_the_check_is_replayable() -> None:
    """判定取自事件日志里最后一条玩家发言，所以它和其他一切一样能被回放。"""
    eng = _engine()
    eng.apply(Action(actor="player", tool="say", args={"text": "我给你的旧音乐盒呢？"}))
    before = eng.observe(NpcId("reimu")).claim_check

    replayed = WorldEngine.replay(eng.state.action_log, eng.defs)

    assert replayed.observe(NpcId("reimu")).claim_check == before
