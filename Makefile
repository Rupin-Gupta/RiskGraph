.PHONY: setup lint test data results up down book db-migrate risk-run backtest eval-dq \
	incidents rag-index eval-agents

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

book:
	uv run python -m riskgraph.cli book generate

db-migrate:
	uv run python -m riskgraph.db.migrate

risk-run: db-migrate
	$(if $(DATE),,$(error usage: make risk-run DATE=YYYY-MM-DD [MARKET=override.parquet]))
	uv run python -m riskgraph.cli run-daily --date $(DATE) $(if $(MARKET),--market-override $(MARKET))

backtest:
	uv run python -m riskgraph.cli backtest --start 2022-01-01

eval-dq:
	uv run python -m riskgraph.eval.dq --seed 42

incidents: db-migrate
	uv run python -m riskgraph.cli incidents generate --n 100 --seed 42

rag-index:
	uv run python -m riskgraph.cli rag index

# make eval-agents SPLIT=dev|test VARIANT=baseline|multi RUNS=n [DRY_RUN=1]
eval-agents: db-migrate
	$(if $(SPLIT),,$(error usage: make eval-agents SPLIT=dev|test VARIANT=baseline|multi RUNS=n [DRY_RUN=1]))
	uv run python -m riskgraph.eval.agents --split $(SPLIT) --variant $(or $(VARIANT),multi) \
		--runs $(or $(RUNS),1) $(if $(DRY_RUN),--dry-run)

up:
	docker compose up -d --wait

down:
	docker compose down
