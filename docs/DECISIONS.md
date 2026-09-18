# Architecture Decision Records

Each ADR records context, decision, alternatives, and consequences. Changes to the SPEC go through this file.

## ADR-001: Locked technology stack

**Status:** Accepted (phase 00)

**Context.** RiskGraph is a portfolio project for data science and GenAI roles in banking. It must run on a laptop and on a single small EC2 instance, keep cloud cost near zero, and produce reproducible, honestly measured results. Changing tools mid-project costs more than any marginal gain from a different tool.

**Decision.** The stack is fixed for all phases:

- **Language and tooling:** Python 3.11; `uv` for environments and the lockfile; `ruff` (lint and format); `mypy` (strict on `pricing/`, `risk/`, `exposure/`); `pytest`.
- **Data:** pandas, NumPy, SciPy, Pandera, parquet; DVC (local remote in phase 00, S3 remote from phase 03).
- **Storage:** Postgres 16 (results, incidents, LangGraph checkpoints); Weaviate with bring-your-own vectors.
- **Agents:** LangGraph plus LangChain core interfaces; LLM via Amazon Bedrock (`ChatBedrockConverse`, model ID from env).
- **Observability:** Langfuse; MLflow for experiments (local tracking, S3 artifacts).
- **Serving:** FastAPI; llama.cpp server for the fine-tuned extractor (GGUF); Next.js static dashboard behind nginx.
- **Deploy:** Docker Compose on a single EC2 instance; GitHub Actions (OIDC) → ECR → SSM Run Command.

Dependencies are approved per phase in SPEC §12; anything else needs explicit approval.

**Alternatives.** Poetry or pip-tools instead of `uv`; a managed vector store (OpenSearch) or pgvector instead of Weaviate; CrewAI instead of LangGraph (compared optionally in phase 07); Kubernetes or ECS instead of Compose on EC2; SageMaker endpoints instead of llama.cpp.

**Consequences.** One lockfile and one toolchain across phases. Single-instance Compose keeps cost low but has no high availability, which is acceptable for a demo. Managed services that bill while idle (NAT gateway, load balancer, OpenSearch, SageMaker endpoints, RDS, GPU instances) are excluded.

## ADR-002: Virtual uv project, no build backend

**Status:** Accepted (phase 00)

**Context.** Making `riskgraph` an installable package needs a build backend (for example `uv_build` or `hatchling`), which is not on the SPEC §12 phase 00–01 dependency list.

**Decision.** `pyproject.toml` sets `[tool.uv] package = false`. Code is imported from `src/` via `PYTHONPATH=src` (exported in the `Makefile`) and `pythonpath` in the pytest config. The CLI runs as `python -m riskgraph.cli`.

**Alternatives.** Add `uv_build` as the build backend and expose a `riskgraph` console script.

**Consequences.** No new dependency. Commands must run through `make` (or set `PYTHONPATH=src`). Revisit when a wheel or console script is needed, such as the API image in phase 03.

## ADR-003: Market panel uses unadjusted Close and no fill

**Status:** Accepted (phase 00)

**Context.** SPEC §3.2 asks for an aligned business-day panel but does not say which yfinance price column it holds, or how gaps are handled.

**Decision.** `data/processed/market_panel.parquet` has one column per series: yfinance `Close` (split-adjusted, not dividend-adjusted) and the FRED observation value. The index is Monday to Friday from 2015-01-01 (`pandas.bdate_range`, no holiday calendar). Holidays and missing observations stay `NaN`; nothing is forward-filled. Yahoo's in-progress bar for the current session is dropped, so only completed sessions are stored. Raw files keep every yfinance column (OHLC, Adj Close, Volume).

**Alternatives.** Use `Adj Close` (total return, but its level is not a tradable spot for option pricing); use an exchange holiday calendar (needs a new dependency).

**Consequences.** Pricing uses the quoted spot level. Dividend drops appear as small negative returns on equity factors; phase 01a can switch equity returns to `Adj Close` from the raw files if needed. Phase 01b market data controls see every gap.
