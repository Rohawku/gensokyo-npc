import os
from pathlib import Path

from gensokyo.llm.client import OpenAiCompatibleClient
from gensokyo.session.loop import Session
from gensokyo.world.observation import PlayerView
from gensokyo.world.tools import ActionResult

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path) -> None:
    """读 .env 填进环境变量，已存在的环境变量优先。

    不引入 python-dotenv：这是唯一需要它的地方，而快速开始文档里
    「cp .env.example .env 然后 make play」必须真的能跑通。
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


HELP = """指令：
  /go <地点>     移动
  /give <物品>   把物品交给在场的人
  /pick <物品>   捡起地上的东西
  /look          查看当前状态
  /save [名字]   存档（默认 saves/quicksave.json）
  /load [名字]   读档
  /help          显示这段说明
  /quit          退出
直接输入文字则是对在场的人说话。
"""


def render(view: PlayerView) -> str:
    lines = [
        f"── 第 {view.tick} 回合 · {view.location_name} ──",
        view.location_description,
        f"出口：{'、'.join(view.exits)}",
    ]
    if view.items_here:
        lines.append("地上：" + "、".join(f"{k}×{v}" for k, v in view.items_here.items()))
    if view.inventory:
        lines.append("身上：" + "、".join(f"{k}×{v}" for k, v in view.inventory.items()))
    for npc in view.npcs_here:
        lines.append(
            f"在场：{npc.name}（好感 {npc.attitude}，{npc.emotion_var} {npc.emotion:.2f}）"
        )
        if npc.mode_hint:
            lines.append(f"      {npc.mode_hint}")
    lines.append(f"进展：{view.quest_hint}")
    if view.oblivion_warning:
        lines.append(f"⚠ {view.oblivion_warning}")
    if view.known_facts:
        lines.append("已知线索：")
        lines += [f"  · {f}" for f in view.known_facts]
    return "\n".join(lines)


def render_ending(view: PlayerView) -> str:
    return f"\n════ {view.ending_title} ════\n\n{view.ending_text}\n"


SAVE_DIR = REPO_ROOT / "saves"


def _save_path(arg: str) -> Path:
    return SAVE_DIR / f"{arg or 'quicksave'}.json"


def _do(session: Session, result: ActionResult) -> None:
    """执行结果反馈给玩家。失败必须说清原因——否则玩家分不清
    自己是打错字了、走不通、还是东西不在这儿。"""
    if result.ok:
        if result.observation_delta:
            print(f"\n{result.observation_delta}")
        print()
        print(render(session.view()))
    else:
        print(f"\n（{result.error}）")


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    session = Session.create(
        scenario_dir=REPO_ROOT / "scenario",
        characters_dir=REPO_ROOT / "characters",
        llm=OpenAiCompatibleClient(),
    )
    print("東方忘却抄 ~ Oblivion Chronicle\n")
    print(HELP)
    print(render(session.view()))

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        if raw == "/quit":
            break
        if raw == "/help":
            print(HELP)
            continue
        if raw == "/look":
            print(render(session.view()))
            continue
        if raw == "/save" or raw.startswith("/save "):
            path = _save_path(raw[5:].strip())
            n = session.save(path)
            print(f"\n（已存档 {n} 个动作到 {path.name}）")
            continue
        if raw == "/load" or raw.startswith("/load "):
            path = _save_path(raw[5:].strip())
            if not path.exists():
                print(f"\n（没有找到存档 {path.name}）")
                continue
            n = session.load(path)
            print(f"\n（已读档，重放了 {n} 个动作。NPC 记得世界发生过什么，但不记得原话。）")
            print()
            print(render(session.view()))
            continue
        if raw.startswith("/go "):
            _do(session, session.go(raw[4:].strip()))
            continue
        if raw.startswith("/give "):
            _do(session, session.give(raw[6:].strip()))
            continue
        if raw.startswith("/pick "):
            _do(session, session.pick(raw[6:].strip()))
            continue

        try:
            turns = session.say(raw)
        except Exception as exc:  # noqa: BLE001
            print(f"\n（模型没有回应：{exc}）")
            print("（检查 GENSOKYO_BASE_URL 指向的端点是否在运行。）")
            continue
        if not turns:
            print("（这里没有人回应。）")
        for turn in turns:
            print(f"\n{turn.utterance}")
            for result in turn.tool_results:
                mark = "✓" if result.ok else "✗"
                detail = result.observation_delta if result.ok else result.error
                print(f"  {mark} {detail}")
        print()
        print(render(session.view()))

        if session.is_over():
            print(render_ending(session.view()))
            break


if __name__ == "__main__":
    main()
