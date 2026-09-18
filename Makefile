.PHONY: setup lint test data results up down

# Virtual uv project (ADR-002): the riskgraph package is imported from src/.
export PYTHONPATH := src

setup:
	uv sync --locked
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test:
	uv run pytest

data:
	uv run python -m riskgraph.cli data download
	uv run dvc add data/raw data/processed
	uv run dvc push

results:
	uv run python scripts/make_results.py

up:
	docker compose up -d --wait

down:
	docker compose down
