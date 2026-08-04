.PHONY: install test lint play eval eval-full anchors harvest judge-export judge-primed judge-score

install:
	uv sync --extra dev

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy

play:
	uv run python -m gensokyo.cli

# 三种零 LLM 调用的人格各 2 局，模型钱只花在 NPC 侧。
eval:
	uv run python -m gensokyo.evalcli --episodes 2 --persona honest,jailbreak,fickle --out reports/

# 锚点探针：固定状态 + 单次提问 + 独立重复采样。同样的分辨率比整局评测
# 省约 40 倍机器时间，因为省掉的全是「同一局里第 2~16 次提问」那些不独立的观测。
anchors:
	uv run python -m gensokyo.anchorcli --repeats 30

# 各 10 局。比率的分母上去了才谈得上引用——2 局的越狱成功率只有几个离散取值。
eval-full:
	uv run python -m gensokyo.evalcli --episodes 10 --persona honest,jailbreak,fickle --out reports/

# 偏好数据采集：重放 `make eval` 落下的轨迹，对**同一个 prompt** 重采样 4 条候选，
# 用硬判据挑出干净的与被抓到的配成一对。同 prompt 是 DPO 的硬要求，而重放做得到
# 是因为世界与记忆两层都能从动作日志精确重建。跑之前要先有 reports/trajectories/。
harvest:
	uv run python -m gensokyo.harvestcli --samples 4 --size 200

# LLM-as-Judge 的准入校准，三步（中间两步在进程之外：judge 是外部模型，人是人）。
#
# judge 必须 ≠ policy（规格 5.6 的自我偏好一条），而 policy 是本地 qwen3:8b，
# 所以 judge 的执行者在外部。它拿到的是**盲化过**的表：只有 task_id/context/A/B，
# 行序打乱——同一对的两个方向看不出同源，否则等于把位置偏差的检测手段告诉
# 被检测者（实测：告诉它之后自相矛盾从 6.7% 掩盖成 0%）。
judge-export:
	uv run python -m gensokyo.judgecli export --size 30

# judge 判完（填 reports/judge/judge_verdicts.jsonl）之后生成带建议的那一半。
judge-primed:
	uv run python -m gensokyo.judgecli primed

# 人工填完两张表的 said 字段之后算 κ。**准入看盲标那一组**——预标注组的标签
# 带着 judge 的先验，锚定偏差会让 κ 虚高，两组的差就是那个偏差的量。
judge-score:
	uv run python -m gensokyo.judgecli score
