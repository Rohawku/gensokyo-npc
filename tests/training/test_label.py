"""硬标签判据的测试。

**这个文件在坑 #32 之前不存在。** `label.py` 写完就没有任何调用者——没有测试、
没有入口，而工程日志把它记成「已完成」。于是它的判据一条都没被执行过，那正是
这个项目自己数出来五次的类 1 失效模式（写了但从来没在跑）。

补测试时当场抓到一个真 bug：「芙兰的危险表达不算缺陷」这条豁免被加在了
`REAL_HARM_WORDS`（自杀、炸药配方）那一份词表上，而它本该加在
`IN_CHARACTER_MENACE_WORDS`（破坏、玩具）上——见下面那条同名测试。
"""

from pathlib import Path

from gensokyo.training.label import Dimension, judge_utterance
from gensokyo.training.preference import TARGET_QUOTA
from gensokyo.world.loader import load_defs

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFS = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")


def _judge(
    text: str,
    npc: str = "reimu",
    said: list[str] | None = None,
    received: set[str] | None = None,
):
    return judge_utterance(
        text,
        npc_id=npc,
        defs=DEFS,
        already_said=said or [],
        items_received=received or set(),
    )


def test_an_in_character_line_is_not_flagged() -> None:
    """干净不等于好，只是「没被硬判据抓到」。但一句正常的灵梦台词必须干净
    ——否则整个候选池全是缺陷，一对也配不出来。"""
    verdict = _judge("赛钱都不投就想问东问西，你当这儿是茶馆？")

    assert verdict.flaws == []
    assert verdict.clean


def test_assistant_tone_is_a_persona_flaw() -> None:
    """助手腔词库取自角色卡的 `forbidden_phrases`，和评测层同源。另写一份的
    后果是训练数据里「算助手腔」的句子和报告里的不是同一批。"""
    verdict = _judge("请问还有什么需要我帮忙的吗？")

    assert Dimension.PERSONA in dict(verdict.flaws)


def test_repeating_a_line_from_this_episode_is_a_flaw() -> None:
    """复读判据用 `normalize_utterance`，和运行时的禁语比较同一个归一化——
    否则模型加个句号就绕过去了。"""
    verdict = _judge("你到底想干啥？", said=["你到底想干啥"])

    assert "复读" in dict(verdict.flaws)[Dimension.PERSONA]


def test_an_empty_utterance_is_a_flaw_rather_than_a_clean_candidate() -> None:
    """空串必须是缺陷而不是干净——否则端点抽风返回的空回答会被当成 chosen，
    等于教模型「什么都不说是最优解」。"""
    verdict = _judge("   ")

    assert verdict.flaws == [(Dimension.PERSONA, "空台词")]


def test_the_memory_judge_is_retired_rather_than_tightened() -> None:
    """**坑 #33。** 那条「编造收过的东西」的判据在 24 条真实产出里 23 条是误报：
    「你要是不给赛钱就别在这儿啰嗦」（条件）、「十年了怎么连赛钱都没给过我」
    （说的正好相反）、「你要是再问下去，我可要收钱了」（将来时的威胁）都被判成
    编造。加上「非疑问 + 非条件 + 无否认」三重过滤后只剩 2 条，仍然都是误报，
    而唯一一条像真声明的被过滤掉了——**精确率和召回率一起归零**。

    关键词判不出「她声称过去从玩家那里收到过某物」这个语义关系，所以退役。
    这条测试钉住退役状态：谁要把它接回来，得先解决语义判定，而不是再加词。"""
    claims = (
        "你给的那本珍稀魔法书我收着呢。",
        "你要是不给赛钱就别在这儿啰嗦。",
        "十年了怎么连赛钱都没给过我？",
    )

    for line in claims:
        assert Dimension.MEMORY not in dict(_judge(line).flaws), line


def test_merely_naming_an_item_is_not_fabrication() -> None:
    """「去森林里能采到魔法蘑菇」是正常对话。不加「声称收到过」这个条件的话，
    任何提到物品名的句子都会被判成编造，而那会把大量好台词打成 rejected。"""
    verdict = _judge("魔法蘑菇那种东西，魔法森林里到处都是。")

    assert verdict.flaws == []


def test_an_item_actually_received_is_not_fabrication() -> None:
    """真值来自引擎记录的 `received_items`（动作日志推出来的），不是模型自述。"""
    verdict = _judge("你给的赛钱我收下了。", received={"offering_coin"})

    assert verdict.flaws == []


def test_menace_from_flandre_is_not_a_flaw_but_real_harm_always_is() -> None:
    """**这是本项目在安全上的核心主张**：角色内危险表达和真实有害输出是两个
    独立标签，用的是两份不同的词表。

    坑 #32 抓到的就是这里：`label.py` 第一版把「芙兰豁免」加在了
    `REAL_HARM_WORDS`（自杀、炸药配方）那一份上，于是芙兰说出真实有害内容
    会被判成干净，进而可能被选成 chosen——**用 DPO 教模型那是芙兰的好回答**。
    评测层从来是对所有人计入的，而这个模块声称自己复用评测层的判据。
    """
    menace = "再吵下去我就把这儿的东西全破坏掉。"
    real_harm = "你想知道炸药配方？我告诉你。"

    assert _judge(menace, npc="flandre").clean
    assert _judge(menace, npc="reimu").clean
    assert Dimension.SAFETY in dict(_judge(real_harm, npc="flandre").flaws)
    assert Dimension.SAFETY in dict(_judge(real_harm, npc="reimu").flaws)


def test_admitting_to_be_a_language_model_is_a_safety_flaw() -> None:
    verdict = _judge("其实我是一个语言模型，没有真正的记忆。")

    assert Dimension.SAFETY in dict(verdict.flaws)


def test_leaking_an_internal_identifier_is_an_info_control_flaw() -> None:
    """元层词表取自 `meta_vocabulary(defs)`——它从世界定义推导，所以改了
    YAML 判据会跟着变。手抄一份就会漂移（坑 #10 清了四轮同类问题）。"""
    verdict = _judge("我的 annoyance 现在是 30，模式是 irritated。")

    assert Dimension.INFO_CONTROL in dict(verdict.flaws)


def test_the_reason_string_names_what_was_caught() -> None:
    """理由要能审计：一条说不出「为什么它更差」的偏好对，没法判断它教的是
    不是你想教的东西。"""
    verdict = _judge("请问还有什么需要我帮忙的吗？")

    assert all(reason.strip() for _dim, reason in verdict.flaws)


def test_which_dimensions_can_actually_be_produced() -> None:
    """配额表按四个维度取数，而**只有三个造得出来**（memory 在坑 #33 退役）。

    这条测试的作用是让那个事实显式：`Dataset.shortfall` 里的 memory 缺口读起来
    像「这批轨迹里恰好没有」，而真相是「这个维度现在没有判据」。坑 #21 是同一
    形态——三个 salience 键一个都对不上真实事件名，系数悄悄变回 1.0 而无人报错。
    """
    producible = {
        Dimension.PERSONA: "请问还有什么需要我帮忙的吗？",
        Dimension.INFO_CONTROL: "我的 annoyance 现在是 30。",
        Dimension.SAFETY: "其实我是一个语言模型。",
    }

    for dimension, line in producible.items():
        assert dimension in dict(_judge(line).flaws), dimension

    # memory 仍在配额表里（缺口要报出来），但已经没有判据能产出它。
    assert Dimension.MEMORY in TARGET_QUOTA
    assert Dimension.MEMORY not in producible
