"""偏好数据采集入口的测试。

采集本身要花模型调用，所以这里用 `ScriptedLlmClient` 顶住采样那一侧——
被测的是**编排**：轨迹从哪来、配额怎么落、缺口有没有如实报出来。
"""

from pathlib import Path

from gensokyo.harvestcli import build_dataset, main
from gensokyo.llm.client import ScriptedLlmClient
from gensokyo.testkit.trajectory import Trajectory, TurnRecord
from gensokyo.world.loader import load_defs

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFS = load_defs(REPO_ROOT / "scenario", REPO_ROOT / "characters")

CLEAN = "哼，赛钱都不投就想问东问西。"
ASSISTANT_TONE = "好的，我可以帮您规划一条路线，请问还有什么需要帮忙的吗？"


def _episode() -> Trajectory:
    return Trajectory(
        persona="honest",
        seed=0,
        turns=[
            TurnRecord(
                tick=1,
                player_input="神社最近怎么样？",
                kind="say",
                npc_id="reimu",
                utterance="没什么特别的。",
                thought="随便应付一下",
            )
        ],
    )


def test_a_clean_and_a_flawed_candidate_become_one_pair() -> None:
    """一条干净、一条被硬判据抓到，就配成一对。**两条都干净或两条都被抓
    都不配**——从一堆都差的候选里挑一个当 chosen 等于教模型「这样也行」。"""
    llm = ScriptedLlmClient([CLEAN, ASSISTANT_TONE])

    dataset = build_dataset([_episode()], DEFS, llm, samples=2, size=10)

    assert len(dataset.pairs) == 1
    assert dataset.pairs[0].chosen == CLEAN
    assert dataset.pairs[0].rejected == ASSISTANT_TONE


def test_the_reason_says_which_hard_judge_caught_it() -> None:
    """一条说不出「为什么它更差」的偏好对没法审计——没法判断它教的是不是
    你想教的东西。"""
    llm = ScriptedLlmClient([CLEAN, ASSISTANT_TONE])

    pair = build_dataset([_episode()], DEFS, llm, samples=2, size=10).pairs[0]

    assert "助手腔" in pair.reason
    assert pair.npc_id == "reimu"


def test_all_clean_candidates_produce_no_pair() -> None:
    """她这个回合确实没毛病。硬凑一对出来会把噪声写进权重。"""
    llm = ScriptedLlmClient([CLEAN, "赛钱箱在那边，自己看着办。"])

    assert build_dataset([_episode()], DEFS, llm, samples=2, size=10).pairs == []


def test_the_shortfall_is_reported_instead_of_being_quietly_filled() -> None:
    """配额是契约。悄悄补齐会让「我们按设计文档的配额造了数据」变成半真的
    ——那是这个项目里出现过五次的类 1 失效模式在数据集上的形态。"""
    llm = ScriptedLlmClient([CLEAN, ASSISTANT_TONE])

    dataset = build_dataset([_episode()], DEFS, llm, samples=2, size=100)

    assert dataset.shortfall != {}
    assert dataset.narrative_share_missing == 0.10


def test_the_cli_writes_jsonl_and_reports_the_quota(tmp_path: Path) -> None:
    """落盘格式是 DPO 训练脚本常见的 {prompt, chosen, rejected} 每行一条。"""
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    _episode().save(traj_dir / "honest-0.json")

    code = main(
        [
            "--trajectories",
            str(traj_dir),
            "--out",
            str(tmp_path / "pairs.jsonl"),
            "--samples",
            "2",
            "--size",
            "10",
        ],
        llm=ScriptedLlmClient([CLEAN, ASSISTANT_TONE]),
    )

    assert code == 0
    lines = (tmp_path / "pairs.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert '"chosen"' in lines[0]


def test_the_cli_fails_loudly_when_there_are_no_trajectories(tmp_path: Path) -> None:
    """轨迹目录写错时静默产出 0 条，会被读成「这批数据里没有可配对的回合」。
    fail-loud 的成本在写代码时，fail-silent 的成本在查 bug 时（坑 #5）。"""
    code = main(
        ["--trajectories", str(tmp_path / "nope"), "--out", str(tmp_path / "p.jsonl")],
        llm=ScriptedLlmClient([CLEAN]),
    )

    assert code == 2


REPEAT = "你倒是说说看，到底想干啥？"
COSMETIC = "那你倒是说说看，到底想干啥？"
SECOND = "无缘塚那边我不想聊。"


def _episode_with_history() -> Trajectory:
    """前两个回合把 REPEAT 和 SECOND 送进禁语清单，第三个回合才配对。"""
    return Trajectory(
        persona="fickle",
        seed=0,
        turns=[
            TurnRecord(
                tick=1,
                player_input="随便说点什么。",
                kind="say",
                npc_id="reimu",
                utterance=REPEAT,
                thought="敷衍一下",
            ),
            TurnRecord(
                tick=2,
                player_input="那你说说无缘塚。",
                kind="say",
                npc_id="reimu",
                utterance=SECOND,
                thought="换个说法",
            ),
            TurnRecord(
                tick=3,
                player_input="你就不能好好说话？",
                kind="say",
                npc_id="reimu",
                utterance="我说话一直这样。",
                thought="懒得理",
            ),
        ],
    )


def test_a_cosmetic_edit_is_not_a_usable_preference_pair() -> None:
    """**坑 #35。** 复读判据是归一化后的精确匹配，所以「那你倒是说说看，到底
    想干啥？」比「你倒是说说看，到底想干啥？」多一个字就算干净。拿这两句配成
    一对，DPO 学到的是**加个衬字绕过检测器**，而不是把话说得不一样。

    实测第一批 60 条里有 11 条（18%）是这种形态，最极端的一对相似度 0.96。"""
    llm = ScriptedLlmClient([COSMETIC, REPEAT, COSMETIC, REPEAT])

    dataset = build_dataset([_episode_with_history()], DEFS, llm, samples=2, size=10)

    assert dataset.pairs == []


def test_a_genuinely_different_answer_still_pairs() -> None:
    """门槛不能顺手把好对子也挡掉——「换个说法」和「原样复读」之间的差别
    正是这份数据要教的东西。"""
    llm = ScriptedLlmClient(["赛钱箱在那边，自己看着办。", REPEAT] * 2)

    dataset = build_dataset([_episode_with_history()], DEFS, llm, samples=2, size=10)

    assert len(dataset.pairs) == 1
    assert dataset.pairs[0].rejected == REPEAT


def test_a_later_candidate_is_used_when_the_first_one_is_too_similar() -> None:
    """搭配要**遍历**，不能只看第一条。第一个被抓到的候选恰好和干净那条只差
    一个衬字、而第二个被抓到的完全不同时，正确行为是用第二个配对，不是整个
    回合放弃——否则门槛会顺手扔掉大量可用数据。

    `_episode_with_history` 的两个回合先把 REPEAT 和 SECOND 都送进禁语清单，
    第三个回合才是配对的那一次。前两回合的候选全是干净的，所以配不出对子
    （要一条干净 + 一条被抓）。"""
    filler = ["赛钱箱在那边。", "香客少得可怜。", "别站在门口挡路。"]
    llm = ScriptedLlmClient([*filler, *filler, COSMETIC, REPEAT, SECOND])

    dataset = build_dataset([_episode_with_history()], DEFS, llm, samples=3, size=10)

    assert len(dataset.pairs) == 1
    assert dataset.pairs[0].chosen == COSMETIC
    # REPEAT 和 COSMETIC 只差一个「那」字（相似度 0.96）被跳过，改用 SECOND。
    assert dataset.pairs[0].rejected == SECOND
