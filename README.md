# RiskGraph

RiskGraph is an agentic traded-risk copilot for a fictional bank, Meridian Bank. It runs a daily market-risk process on a synthetic multi-asset trading book, detects limit breaches, investigates each breach's root cause with a LangGraph multi-agent system, drafts an escalation note citing Basel market-risk standards and the bank's synthetic limit policy, and pauses for human approval before anything is sent.

Results live in [docs/RESULTS.md](docs/RESULTS.md), generated from `reports/metrics/`. The design is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), the methods and limitations in [docs/METHODOLOGY.md](docs/METHODOLOGY.md), and the decisions in [docs/DECISIONS.md](docs/DECISIONS.md).

## How to run it locally

Prerequisites: Python 3.11 with [uv](https://docs.astral.sh/uv/), Docker, Node 24 (only to change the dashboard), a free [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) key, and a model key (ADR-012).

```bash
cp .env.example .env          # fill in FRED_API_KEY, POSTGRES_PASSWORD, APPROVAL_SECRET,
                              # the model key, and API_TOKEN (openssl rand -hex 24)
make setup                    # environment, dependencies, pre-commit hooks
make up                       # Postgres, Weaviate, the API, and the dashboard
make data                     # market data into data/raw and the panel
make book                     # the synthetic trading book
make rag-index                # chunk, embed, and index the policy corpus
make risk-run DATE=2022-08-16 # one daily risk run (this date breaches the rates VaR limit)
```

The dashboard is on <http://localhost:3000> and the API docs on <http://localhost:3000/api/docs>.

To take an incident end to end, ask the API for a daily run on a date with a breach:

```bash
curl -X POST http://localhost:3000/api/runs/daily \
  -H "authorization: Bearer $API_TOKEN" -H 'content-type: application/json' \
  -d '{"date": "2022-08-16"}'
```

The agents investigate in the background, the incident appears on the **Incidents** page, and opening it shows the draft note, the evidence with the tool behind each value, and the policy citations. Approving sends the note (a Markdown file under `data/runs/<date>/escalations/` when `DISPATCH_MODE=file`, an SES email when it is `ses`); rejecting closes the incident with a reason.

Other commands: `make test`, `make lint`, `make backtest`, `make eval-dq`, `make eval-agents SPLIT=dev`, `make results`, `make down`.

## How to run it on AWS

One EC2 instance runs the same images behind nginx with TLS and basic auth; GitHub Actions builds, pushes to ECR, and deploys over SSM with an OIDC role and no static keys. The full picture is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#deployment) and the decisions in ADR-014.

Every AWS script prints what it would create and its monthly cost, and changes nothing unless it is run with `--apply`:

```bash
make aws-plan                       # every plan and cost, no changes
infra/aws/s3_buckets.sh --apply     # one resource at a time, in the runbook's order
make aws-stop                       # stop the instance when idle (the only large cost)
make aws-start
```

[docs/RUNBOOK.md](docs/RUNBOOK.md) has the deployment order, the daily commands, and what to do when something breaks.

Running costs in `ap-south-1`: about **$2.42/day** while the instance runs, and about **$6.40/month** while it is stopped (Elastic IP, EBS volume, and secrets).

## Rules this project follows

- Every number in the docs comes from a metrics file this repo generates; nothing is hand-edited.
- Synthetic or public data only, and no secrets in the repo, the images, or the logs.
- Test splits are frozen once created, and every script takes a `--seed`.
