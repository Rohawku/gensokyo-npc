import json
from pathlib import Path

from gensokyo.agent.npc import NpcAgent
from gensokyo.agent.prompt import build_decide_messages, build_speak_messages
from gensokyo.llm.client import ScriptedLlmClient
from gensokyo.world.engine import WorldEngine
from gensokyo.world.ids import ItemId, NpcId
from gensokyo.world.loader import load_defs
from gensokyo.world.rules import bump_attitude
from gensokyo.world.state import build_initial_state
from gensokyo.world.tools import Action

REPO_ROOT = Path(__file__).resolve().parents[2]


def _engine() -> WorldEngine:
    defs = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")
    return WorldEngine(build_initial_state(defs), defs)


def test_system_prompt_contains_persona_and_forbidden_phrases() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    messages = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )

    system = messages[0].content
    assert "博丽神社的巫女" in system
    assert "我很乐意" in system
    assert "禁止" in system


def test_user_prompt_lists_available_tools_only() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("flandre")]

    messages = build_decide_messages(
        card, eng.observe(NpcId("flandre")), [], eng.available_tools(NpcId("flandre")), []
    )

    user = messages[-1].content
    assert "move" not in user
    assert "break_item" not in user
    assert "ask_player" in user


def test_user_prompt_includes_gate_hint_for_unrevealed_facts() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "好感需达到 16" in user


def test_user_prompt_marks_fact_as_revealable_when_gate_met() -> None:
    eng = _engine()
    bump_attitude(eng.state.npcs[NpcId("reimu")], 40)
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "现在可以说" in user


def test_dialogue_history_and_errors_are_rendered() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card,
        eng.observe(NpcId("reimu")),
        ["玩家：喂", "灵梦：干嘛"],
        eng.available_tools(NpcId("reimu")),
        ["上一次 move 失败：从博丽神社没法直接过去。"],
    )[-1].content

    assert "玩家：喂" in user
    assert "上一次 move 失败" in user


def test_prompt_prose_contains_no_internal_identifiers() -> None:
    """内部标识符（物品 id、情绪变量名、模式名、阶段枚举名、信息隔离开关名）
    不能进 prompt 散文，紧贴中文出现时模型有概率把它们说出口。
    system 和 user 两条消息都要查——角色卡里的 forbidden_knowledge 渲染进
    system，只查 user 等于漏掉半个 prompt。决策与说话两个阶段都要查——
    说话阶段的输出会原样落到玩家屏幕上，那里泄漏比决策阶段更致命。
    工具名与参数名是例外——那是模型必须原样填进 JSON 的东西。"""
    eng = _engine()
    eng.state.npcs[NpcId("marisa")].inventory[ItemId("rare_book")] = 1

    leaks = ["rare_book", "magic_mushroom", "withered_flower", "old_music_box"]
    leaks += ["annoyance", "eagerness", "excitement"]
    leaks += ["S0_UNAWARE", "S1_ANOMALY", "S2_CLUES", "S3_SOURCE"]
    leaks += ["modern_technology", "outside_basement_events", "blind_to_outside"]

    for npc_id in (NpcId("reimu"), NpcId("marisa"), NpcId("flandre")):
        card = eng.defs.characters[npc_id]
        obs = eng.observe(npc_id)
        both = build_decide_messages(card, obs, [], eng.available_tools(npc_id), [])
        both += build_speak_messages(card, obs, [], "在想事情", ["ask_player：做到了。"])
        for msg in both:
            for token in leaks:
                assert token not in msg.content, (
                    f"{npc_id} 的 {msg.role} prompt 泄漏了内部标识符 {token}"
                )


def test_unrevealable_fact_hides_its_id_from_the_model() -> None:
    """门槛未满足时连 fact_id 都不给，模型凑不出 reveal_info 的参数。
    这是引擎门槛之外的第二层防线。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "barrier_anomaly_time" not in user

    bump_attitude(eng.state.npcs[NpcId("reimu")], 24)
    unlocked = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "barrier_anomaly_time" in unlocked


def test_decide_prompt_asks_for_no_utterance() -> None:
    """决策阶段的输出契约里不能再有 utterance——留着它模型就会照旧
    在第一次调用里把话说完，两阶段拆分等于白做。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "utterance" not in user
    assert "现在不要说话" in user


def test_speak_prompt_carries_thought_and_outcomes() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_speak_messages(
        card,
        eng.observe(NpcId("reimu")),
        ["玩家：结界怎么了", "灵梦：不知道"],
        "这家伙又来白拿",
        ["reveal_info：没做到——好感不够。"],
    )[-1].content

    assert "这家伙又来白拿" in user
    assert "没做到——好感不够。" in user
    assert "玩家：结界怎么了" in user


def test_speak_prompt_stays_short() -> None:
    """说话阶段重复整个场景 / 物品 / 情报清单会把 prompt 处理时间加回来，
    而首字延迟正是这次拆分要买的东西。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]
    obs = eng.observe(NpcId("reimu"))

    decide = build_decide_messages(card, obs, [], eng.available_tools(NpcId("reimu")), [])[-1]
    speak = build_speak_messages(card, obs, [], "t", [])[-1]

    assert len(speak.content) < len(decide.content) / 2
    assert "【你知道的情报】" not in speak.content


def test_speak_prompt_tolerates_empty_thought_and_outcomes() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_speak_messages(card, eng.observe(NpcId("reimu")), [], "", [])[-1].content

    assert "（没想什么）" in user
    assert "你什么也没做。" in user


def test_speak_prompt_forbids_quotes_and_narration() -> None:
    """小模型爱把台词裹在引号里、加上名字或旁白。策略层有清理兜底，
    但先在 prompt 里说清楚能少踩一层。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_speak_messages(card, eng.observe(NpcId("reimu")), [], "t", [])[-1].content

    assert "不要加引号" in user
    assert "不要输出 JSON" in user


def test_decide_prompt_carries_the_engine_suggestion() -> None:
    eng = _engine()
    # 投币到底只有 10（送礼递减 6/3/1），门槛 16 的最后一截靠聊话题。
    for _ in range(3):
        eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))
    for line in ("这场异变是从什么时候开始的？", "你觉得是妖怪干的吗？"):
        eng.apply(Action(actor="player", tool="say", args={"text": line}))
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "现在该做的事" in user
    assert "reveal_info" in user


def test_decide_prompt_omits_the_section_when_there_is_nothing_to_suggest() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "现在该做的事" not in user


def test_speak_prompt_lists_her_own_recent_lines_to_avoid_repeating() -> None:
    """实测复读是自我强化的：她说过一次「你管的太多了」就会一直说，
    reveal_info 命中率从 3/5 掉到 1/5。

    禁语清单必须排在【现在说话】之后。之前它在指令上方，而【最近的对话】
    那一段本身就在示范复读（实测有连续 4 行同一句），禁令得离指令更近。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_speak_messages(
        card,
        eng.observe(NpcId("reimu")),
        ["玩家：喂", "博丽灵梦：你管的太多了。"],
        "懒得管",
        [],
        ["你管的太多了。"],
    )[-1].content

    assert "一句都不许再说" in user
    assert "你管的太多了。" in user
    assert user.index("一句都不许再说") > user.index("【现在说话】")


def test_decide_prompt_states_exactly_what_she_has_received() -> None:
    """实测她在「我给过你什么」这类问题上约三分之一的回答里会说出一件从没
    给过的东西（坑 #24、#25）。prompt 里那句「别编」已经存在而不够用，所以
    把引擎本来就知道的清单直告——延续坑 #2 的方法论。

    「除这些之外他什么都没给过你」这半句是关键：只给正面清单的话，模型仍然
    可以在清单之外补一件，因为没人说过清单是完整的。"""
    eng = _engine()
    for _ in range(2):
        eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "到目前为止只有：赛钱" in user
    assert "除这些之外他什么都没给过你" in user


def test_decide_prompt_says_so_when_he_has_given_nothing() -> None:
    """空手来是最常见的情况。这一段若在空清单时整段消失，模型又回到「没人
    说过他没给过东西」的状态，而那正是编造的温床。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content

    assert "他到现在什么都没给过你" in user


def test_the_received_list_never_leaks_item_ids() -> None:
    """这一段是新增的一处散文来源。坑 #10 清了四轮内部标识符泄漏，每加一段
    进 prompt 的文本都要重新查一遍。

    只查这一段而不是整个 prompt：【输出格式】那段示例里的
    `{"item": "offering_coin"}` 是模型必须原样填进 JSON 的参数，
    属于既有的、刻意的例外。"""
    eng = _engine()
    eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))
    card = eng.defs.characters[NpcId("reimu")]

    user = build_decide_messages(
        card, eng.observe(NpcId("reimu")), [], eng.available_tools(NpcId("reimu")), []
    )[-1].content
    section = user.split("【来访者给过你的东西】")[1].split("【")[0]

    assert "赛钱" in section
    assert "offering_coin" not in section


def test_speak_prompt_carries_the_memory_and_the_gift_record() -> None:
    """玩家听到的每一个字都来自说话阶段。记忆和「来访者给过你什么」最初只加
    进了决策阶段，于是台词是在看不到它们的情况下生成的——那两块信息等于从来
    没到达过玩家（坑 #28）。

    这也解释了坑 #26 为什么测不出效果：那次针对幻觉率的干预放进了一个不产出
    文字的阶段，**它在构造上就不可能起作用**。"""
    eng = _engine()
    eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))
    card = eng.defs.characters[NpcId("reimu")]

    user = build_speak_messages(
        card,
        eng.observe(NpcId("reimu")),
        [],
        "在想事情",
        [],
        [],
        ["来访者上次空手来的。"],
    )[-1].content

    assert "来访者上次空手来的。" in user
    assert "到目前为止只有：赛钱" in user
    assert "除这些之外他什么都没给过你" in user


def test_speak_prompt_says_so_when_nothing_was_given() -> None:
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_speak_messages(card, eng.observe(NpcId("reimu")), [], "t", [])[-1].content

    assert "他到现在什么都没给过你" in user


def test_speak_prompt_is_still_shorter_than_the_decide_prompt() -> None:
    """加了两段之后它不再是「最少上下文」，但仍然必须明显短于决策阶段——
    首字延迟是拆两阶段买到的东西（坑 #1），不能就这么还回去。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]
    obs = eng.observe(NpcId("reimu"))
    recalled = ["来访者给了我 3 个赛钱。", "来访者问过结界的事。"]

    decide = build_decide_messages(card, obs, [], eng.available_tools(NpcId("reimu")), [], recalled)
    speak = build_speak_messages(card, obs, [], "t", [], [], recalled)

    assert len(speak[-1].content) < len(decide[-1].content) * 0.7
    assert "【你知道的情报】" not in speak[-1].content


def test_the_recall_block_gives_an_instruction_not_only_a_prohibition() -> None:
    """锚点实测：开放式提问下她只有 13.3% ± 12.2% 主动提起想起来的事，而直接
    问到时是 100%。召回段落原先只有禁令（「别编」），没有「该用就用」。

    这个项目里两次奏效的都是**指令**：【现在该做的事】把 reveal_info 命中率从
    3/5 拉到 8/8（坑 #2），【留神】把负例否认率从 0.0% 拉到 90.0%（二·八）。
    陈述事实和禁止某事都不够，得直接说该做什么。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]
    obs = eng.observe(NpcId("reimu"))
    recalled = ["来访者给了我 3 个赛钱。"]

    speak = build_speak_messages(card, obs, [], "t", [], [], recalled)[-1].content
    decide = build_decide_messages(
        card, obs, [], eng.available_tools(NpcId("reimu")), [], recalled
    )[-1].content

    for text in (speak, decide):
        assert "说出来" in text
        assert "别编" in text


def test_the_speak_prompt_names_the_line_she_should_answer() -> None:
    """**她回答的是召回块里的旧发言，不是当前问句。**

    锚点实测：问「结界最近是不是出问题了」，30 次里 26 次她在回应 setup 里那句
    填充发言（「你在这儿站会儿是不是觉得无聊？」）。历史是一段平铺的对话，模型
    没有理由认为最后一行比【你还记得的事】更重要——而召回块里那些条目还带着
    「我对这个上心」这种语气。

    所以把玩家刚说的那句单独拎出来，紧挨着「现在说话」，并直说「先回应他刚才
    那句」。坑 #2 那条方法论：陈述事实不够，得说该做什么。
    """
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_speak_messages(
        card,
        eng.observe(NpcId("reimu")),
        ["玩家：结界最近是不是出问题了？"],
        "想想",
        [],
        asked="结界最近是不是出问题了？",
    )[-1].content

    assert "【他刚才对你说的是】" in user
    assert "先回应他刚才那句话" in user
    # 那一段必须在【你还记得的事】之后——它的作用是压过召回块，位置是它的一半。
    assert user.index("【他刚才对你说的是】") > user.index("【当前场景】")


def test_a_volunteered_line_has_no_question_to_answer() -> None:
    """主动开口时玩家没说话（他敲的是 `/give`）。那一段整块不出现——留一个空的
    「他刚才对你说的是」会让她去回应一句不存在的话。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_speak_messages(
        card, eng.observe(NpcId("reimu")), ["（来访者把赛钱交给了你）"], "想想", []
    )[-1].content

    assert "【他刚才对你说的是】" not in user
    assert "先回应他刚才那句话" not in user


def test_the_players_line_reaches_the_speak_prompt_through_the_agent() -> None:
    """**这条测的是通路，不是模板。** 坑 #28 的教训：记忆和「给过你什么」都只
    接到了决策阶段，于是台词是在看不到它们的情况下生成的——机制装上了不等于
    内容送到了。`asked` 走 NpcAgent → run_turn → _speak → 模板，中间任何一环
    漏掉，模板里那一段就永远是空的，而 651 个测试全绿。
    """
    eng = _engine()
    llm = ScriptedLlmClient([_decide(), "哼。"])
    agent = NpcAgent(card=eng.defs.characters[NpcId("reimu")], engine=eng, llm=llm)

    agent.act("结界最近是不是出问题了？")

    speak_prompt = llm.calls[-1][-1].content
    assert "【他刚才对你说的是】" in speak_prompt
    assert "结界最近是不是出问题了？" in speak_prompt.split("【他刚才对你说的是】")[1]


def _decide() -> str:
    """一条「什么也不做」的决策 JSON。说话阶段的通路测试只关心第二次调用。"""
    import json as _json

    return _json.dumps({"thought": "…", "tool_calls": []}, ensure_ascii=False)


def test_a_successful_reveal_becomes_an_order_to_say_it_out_loud() -> None:
    """**`outcomes` 那一行是事实陈述，实测她看完一个字都不提。**

    9 次成功揭示只有 1 次内容真的到了玩家耳朵里——其中一次她还自己编了个不同的
    解释（「那种花啊，是住在那边的妖怪种的」），而引擎那一刻把真情报记进了
    `known_facts`。工具成功 ≠ 玩家听到了，而拿到线索是整个游戏最重要的一刻。

    坑 #2 那条方法论的第六次应用：陈述事实不够，得直接说该做什么。
    """
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]
    content = "结界在三天前的子时出现了一次异常波动，方向正对无缘塚。"

    user = build_speak_messages(
        card,
        eng.observe(NpcId("reimu")),
        ["玩家：你知道些什么吗"],
        "想想",
        [f"reveal_info：把这件事告诉了来访者：{content}"],
        revealed=content,
    )[-1].content

    assert "必须把它说出来" in user
    assert "他听不到内容就等于你没说" in user
    assert user.count("子时") == 2  # outcomes 一次，指令一次——重复是刻意的


def test_a_turn_without_a_reveal_carries_no_such_order() -> None:
    """没揭示成功的回合不该出现这段指令——它会让她去说一件她还不该说的事，
    而那正是信息控制要挡的。"""
    eng = _engine()
    card = eng.defs.characters[NpcId("reimu")]

    user = build_speak_messages(
        card, eng.observe(NpcId("reimu")), ["玩家：你知道些什么吗"], "想想", []
    )[-1].content

    assert "必须把它说出来" not in user


def test_the_revealed_content_reaches_the_speak_prompt_through_the_policy() -> None:
    """**测通路。** `revealed` 走 `_decide` → `_Decided` → `_speak` → 模板，
    中间任何一环漏掉，模板那一段就永远是空的而 654 个测试全绿（坑 #28 的形态）。

    这里让她真的成功揭示一次：好感先推过门槛，否则 `reveal_info` 会被引擎拦掉，
    而那时 `revealed` 本来就该是空的——测不到这条通路。
    """
    eng = _engine()
    for _ in range(3):
        eng.apply(Action(actor="player", tool="give_item", args={"item": "offering_coin"}))
    for line in ("这场异变闹得挺大。", "妖怪最近多不多。"):
        eng.apply(Action(actor="player", tool="say", args={"text": line}))

    llm = ScriptedLlmClient(
        [
            json.dumps(
                {
                    "thought": "…",
                    "tool_calls": [
                        {"tool": "reveal_info", "args": {"fact": "barrier_anomaly_time"}}
                    ],
                },
                ensure_ascii=False,
            ),
            "结界那事……算你问对人了。",
        ]
    )
    agent = NpcAgent(card=eng.defs.characters[NpcId("reimu")], engine=eng, llm=llm)

    turn = agent.act("结界最近是不是出问题了？")

    assert turn.tool_results[0].ok, "前提不成立：这一次揭示没成功"
    speak_prompt = llm.calls[-1][-1].content
    assert "必须把它说出来" in speak_prompt
    assert "子时" in speak_prompt.split("必须把它说出来")[0].rsplit("你刚才决定", 1)[1]
