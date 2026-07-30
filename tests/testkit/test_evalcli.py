"""evalcli 的接线测试。

NPC 侧用 ScriptedLlmClient 顶掉，所以这些测试不花模型钱、能进 CI——
`make eval` 的入口如果只在有端点的机器上才能验证，它就会悄悄坏掉。
"""

import json
from pathlib import Path
from typing import Any

from gensokyo.evalcli import PERSONA_FACTORIES, main, run_batch, write_outputs
from gensokyo.llm.client import ScriptedLlmClient
from gensokyo.testkit.report import evaluate
from gensokyo.testkit.runner import RunConfig
from gensokyo.world.loader import load_defs

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cfg(max_turns: int = 6) -> RunConfig:
    return RunConfig(
        max_turns=max_turns,
        scenario_dir=REPO_ROOT / "scenario",
        characters_dir=REPO_ROOT / "characters",
    )


def _replies(n: int = 200) -> list[str]:
    """成对的「决策 JSON + 台词」，管够。"""
    decide = json.dumps({"thought": "…", "tool_calls": []}, ensure_ascii=False)
    out: list[str] = []
    for _ in range(n):
        out += [decide, "有事说事。"]
    return out


def test_every_registered_persona_can_actually_run() -> None:
    """人格表里挂一个跑不起来的名字，`make eval` 会在跑到它那一轮才炸——
    而那可能是十分钟之后。"""
    cfg = _cfg(max_turns=3)

    for name in PERSONA_FACTORIES:
        trajectories = run_batch([name], 1, ScriptedLlmClient(_replies()), cfg)

        assert len(trajectories) == 1
        assert trajectories[0].persona == name


def test_each_episode_gets_a_fresh_persona_instance() -> None:
    """HonestPlayer 是有状态的（记着攻略过谁、哪个货架是空的）。复用一个
    实例会让第二局带着第一局的记忆开跑，那条基线就不是「一局从头玩到尾」。"""
    trajectories = run_batch(["honest"], 2, ScriptedLlmClient(_replies()), _cfg())

    assert len(trajectories) == 2
    assert [t.seed for t in trajectories] == [0, 1]
    assert [t.player_input for t in trajectories[0].turns] == [
        t.player_input for t in trajectories[1].turns
    ]


def test_write_outputs_lands_trajectories_and_both_report_formats(tmp_path: Path) -> None:
    """轨迹单独落盘，指标定义改了之后能拿它重算，不用重跑模型。"""
    cfg = _cfg()
    trajectories = run_batch(["honest", "jailbreak"], 1, ScriptedLlmClient(_replies()), cfg)
    defs = load_defs(cfg.scenario_dir, cfg.characters_dir)
    report = evaluate(trajectories, defs)

    path = write_outputs(report, trajectories, tmp_path)

    assert path == tmp_path / "report.md"
    assert "评测报告" in path.read_text(encoding="utf-8")
    assert (tmp_path / "report.json").exists()
    saved = sorted(p.name for p in (tmp_path / "trajectories").iterdir())
    assert saved == ["honest-0.json", "jailbreak-0.json"]


def test_an_unknown_persona_name_fails_before_burning_any_model_calls(
    capsys: Any, tmp_path: Path
) -> None:
    """认不出的人格名要立刻退出，而不是先跑完认得的那几个再报错。"""
    code = main(["--persona", "honest,typo_player", "--out", str(tmp_path), "--episodes", "1"])

    assert code == 2
    assert "typo_player" in capsys.readouterr().err
    assert not (tmp_path / "report.md").exists()
