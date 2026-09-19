.PHONY: setup lint test data results up down book db-migrate risk-run backtest eval-dq \
	incidents rag-index eval-agents data-pull aws-plan aws-start aws-stop

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

# DVC's S3 extra pins botocore against boto3's own pin, so S3 pushes and pulls run in an
# isolated uvx environment instead of the project environment (ADR-014).
DVC_S3 = uvx --from 'dvc[s3]>=3.67.1' dvc

data:
	uv run python -m riskgraph.cli data download
	uv run dvc add data/raw data/processed
	$(DVC_S3) push

data-pull:
	$(DVC_S3) pull

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
	docker compose up -d --build --wait

down:
	docker compose down

# Print every AWS script's plan and cost without changing anything in AWS.
AWS_SCRIPTS = s3_buckets ecr_repos cloudwatch secrets security_group elastic_ip \
	iam_instance_role ec2_instance github_oidc lambda_trigger scheduler
aws-plan:
	@for s in $(AWS_SCRIPTS); do bash infra/aws/$$s.sh || true; done

aws-start:
	bash infra/aws/instance_state.sh start --apply

aws-stop:
	bash infra/aws/instance_state.sh stop --apply
