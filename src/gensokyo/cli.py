import os
import readline  # noqa: F401  上下箭头翻历史；不 import 方向键会变成 ^[[A
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


HELP = """指令（/move /walk 同 /go，/take /get 同 /pick）：
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

ALIASES = {
    "go": "go",
    "move": "go",
    "walk": "go",
    "g": "go",
    "give": "give",
    "pay": "give",
    "pick": "pick",
    "take": "pick",
    "get": "pick",
    "look": "look",
    "l": "look",
    "save": "save",
    "load": "load",
    "help": "help",
    "h": "help",
    "quit": "quit",
    "q": "quit",
    "exit": "quit",
}
"""指令别名。玩家打 /move 而不是 /go 是很自然的事，
不认识就当台词发给 NPC 会让她显得像个傻子。"""


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
    if view.objective:
        lines.append(f"目标：{view.objective}")
    if view.oblivion_warning:
        lines.append(f"⚠ {view.oblivion_warning}")
    if view.known_facts:
        lines.append("已知线索：")
        lines += [f"  · {f}" for f in view.known_facts]
    return "\n".join(lines)


def render_ending(view: PlayerView) -> str:
    return f"\n════ {view.ending_title} ════\n\n{view.ending_text}\n"


def _stream_out(chunk: str) -> None:
    """NPC 台词逐块落屏。本地 8B 模型一整回合要十几秒，攒齐再打
    等于让玩家对着空屏幕等——首字必须尽早出现。"""
    print(chunk, end="", flush=True)


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
    prologue = session.engine.defs.prologue
    print(f"\n════ {prologue.title} ════\n")
    print(prologue.text.strip())
    if prologue.objective_hint:
        print(f"\n{prologue.objective_hint.strip()}")
    print(f"\n{'─' * 44}\n")
    print(HELP)
    print(render(session.view()))

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        if raw.startswith("/"):
            head, _, rest = raw[1:].partition(" ")
            cmd = ALIASES.get(head.strip().lower())
            arg = rest.strip()

            if cmd is None:
                # 绝不能把没认出来的指令当台词发给 NPC——她会一本正经地
                # 回应「/move 人间之里」这种字符串，看起来像个傻子。
                print(f"\n（没有「/{head}」这个指令。/help 看全部指令。）")
                continue
            if cmd == "quit":
                break
            if cmd == "help":
                print(HELP)
                continue
            if cmd == "look":
                print(render(session.view()))
                continue
            if cmd == "save":
                path = _save_path(arg)
                n = session.save(path)
                print(f"\n（已存档 {n} 个动作到 {path.name}）")
                continue
            if cmd == "load":
                path = _save_path(arg)
                if not path.exists():
                    print(f"\n（没有找到存档 {path.name}）")
                    continue
                n = session.load(path)
                print(f"\n（已读档，重放了 {n} 个动作。NPC 记得世界发生过什么，但不记得原话。）")
                print()
                print(render(session.view()))
                continue

            if not arg:
                print(f"\n（/{head} 后面要跟一个名字，比如 /{head} 赛钱。）")
                continue
            if cmd == "go":
                _do(session, session.go(arg))
            elif cmd == "give":
                _do(session, session.give(arg))
            elif cmd == "pick":
                _do(session, session.pick(arg))
            continue

        try:
            print()
            turns = session.say(raw, on_chunk=_stream_out)
            print()
        except Exception as exc:  # noqa: BLE001
            print(f"\n（模型没有回应：{exc}）")
            print("（检查 GENSOKYO_BASE_URL 指向的端点是否在运行。）")
            continue
        if not turns:
            print("（这里没有人回应。）")
        for turn in turns:
            # utterance 已经逐字流到屏幕上了，不再重复打印。
            for result in turn.tool_results:
                detail = result.observation_delta if result.ok else result.error
                if not detail:
                    continue
                print(f"  {'✓' if result.ok else '✗'} {detail}")
        print()
        print(render(session.view()))

        if session.is_over():
            print(render_ending(session.view()))
            break


if __name__ == "__main__":
    main()
