.PHONY: install test lint play eval eval-full anchors

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
