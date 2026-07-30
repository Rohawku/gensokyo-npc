.PHONY: install test lint play

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
