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

## ADR-004: Constant-maturity book held at constant moneyness and USD notional

**Status:** Accepted (phase 01a)

**Context.** SPEC §3.1 makes the book constant-maturity so its composition is stable across the backtest window. The SPEC does not say how strikes and quantities behave. Equity prices move a long way between 2022 and 2026, so absolute strikes struck in 2024 would sit deep in or out of the money for much of the window, and fixed share counts would make equity VaR track price levels rather than risk.

**Decision.** On each valuation date the book is re-held: tenors keep their trade-date length, option strikes and FX contract rates scale by level(valuation date) / level(trade date), and equity and option positions keep their USD notional (units = notional / spot). A trade valued on its trade date is priced exactly as booked. Swap fixed rates and bond coupons stay absolute, because rate levels barely change rate risk.

**Alternatives.** Absolute terms (the book drifts in character across the window); regenerating the book every day from the seed (strikes then differ day to day, which creates a spurious position effect in VaR explain).

**Consequences.** VaR moves with volatility and correlation, which is what limits and backtests should measure. VaR explain has a zero position effect for an unchanged book. Backtest P&L excludes time decay. Trade notionals for equities and options are USD amounts, not share counts.

## ADR-005: Pricing conventions

**Status:** Accepted (phase 01a)

**Context.** SPEC §4.1 fixes the models but not the compounding, day count, or foreign-rate inputs.

**Decision.** DGS2/5/10 yields are used as zero rates with semi-annual compounding (so a par bond prices at par, and a swap's par rate equals a flat curve's rate), interpolated linearly with flat extrapolation. Day count is ACT/365.25. Coupons and fixed legs are semi-annual, counted back from maturity; a first stub shorter than about 4 days merges into the next period. Swaps are valued on a reset date (float leg = N · (1 − DF(T))). FX forwards use covered interest parity with constant proxy foreign rates (EUR 2%, INR 6.5%, `configs/risk.yaml`). Black-Scholes uses the curve rate converted to continuous compounding and no dividends. DV01 is PV(curve + 1bp) − PV(curve), so a long bond has negative DV01.

**Alternatives.** Bootstrapping zero rates from par yields; an OIS/SOFR curve (no data); foreign rates from a second data source (not in the approved data).

**Consequences.** Errors of a few bp against a bootstrapped curve. Foreign-rate risk on FX forwards is not a risk factor. All of this is listed in METHODOLOGY.md, Assumptions and limitations.

## ADR-006: Run identity, overrides, and storage

**Status:** Accepted (phase 01a)

**Context.** SPEC §4.8 lists the `risk_results` and `limit_status` columns. Incident injection (§11) will run many overridden runs on dates that also have a base run.

**Decision.** Both tables add a `run_id` column: the date for a base run, `<date>+<8 hex chars of the override files' SHA-256>` otherwise. A rerun replaces its own rows in one transaction, and run.json goes to `data/runs/<run_id>/`. `--book-override` replaces the whole book (parquet, or JSON as written by the generator). `--market-override` is a parquet on the panel's date index whose non-missing values replace the panel's. VaR explain compares the run's book with the base book (`data/synthetic/book/book.parquet`) as the prior day's positions. Tables are created by `python -m riskgraph.db.migrate` (SQLAlchemy `create_all`, idempotent).

**Alternatives.** Key rows by date only (incident runs would overwrite base runs); Alembic migrations (not an approved dependency, and unnecessary while tables are only added).

**Consequences.** Base and incident runs coexist for the same date. Changing a column on a table that already holds data will need a versioned migration.

## ADR-007: Backtest P&L and limit calibration

**Status:** Accepted (phase 01a)

**Context.** SPEC §4.5 asks for hypothetical P&L against the prior day's VaR, and §4.6 for limits calibrated so breaches are rare but occur. Neither fixes the P&L's treatment of model parameters or the calibration rule.

**Decision.** Hypothetical P&L for date t applies t's realized moves of the ten risk factors to the prior date's positions and market state, holding single-stock EWMA vols (a model parameter) fixed, so P&L and VaR share one factor set. Each desk and firm VaR limit, and the firm stress-loss limit, is the 98th percentile of that metric's daily history over 2022–2025, rounded to three significant figures. The backtest re-derives these values and reports breach and warning days under the configured limits.

**Alternatives.** Repricing with the next day's EWMA vols (adds risk the VaR does not model); two-significant-figure limits (the firm limit then exceeds its whole history, so no breach occurs); a fixed multiple of mean VaR (breach counts not controlled).

**Consequences.** By construction roughly 1 − quantile of calibration-period days breach each limit, clustered in a few episodes (actual counts in RESULTS.md). Slowly moving VaR series (rates) sit in the warning band on many days, so later near-miss control incidents are easy to find.

## ADR-008: Market data controls: exclusion, alignment, and evaluation protocol

**Status:** Accepted (phase 01b)

**Context.** SPEC §5 lists the checks and the evaluation. It leaves open how a critical finding changes the daily run, how to align the yfinance and FRED FX series, how to tell a holiday from a missed print, and how to define corruptions and per-type precision.

**Decision.**
- **Exclusion.** A risk factor with a critical finding on the run date is held flat: `RiskContext.excluded` zeroes its shocks in the historical-simulation window, the Monte Carlo draws (Cholesky of the remaining factors' covariance block), historical stress replays, and VaR explain. Its level, EWMA vol, and sensitivities are unchanged, and hypothetical stress still applies. This touches `risk/`, which is outside the phase's listed scope; the change was approved in the phase 01b session. Runs with no exclusion are bit-identical to before.
- **Severity.** Rule checks (Pandera, staleness, cross-source) are critical. Isolation Forest findings are warnings: they go to review and never hold a factor flat.
- **Scope of a daily run.** `run-daily` records findings for its own date only. Earlier dates were checked by their own runs.
- **FX alignment.** The yfinance close dated D is compared with FRED's previous print, which it tracks more closely than the same-day print. Tolerance 200bp.
- **Holidays.** A null is a missed print only when most of the series' calendar peers print that day.
- **Staleness.** Liquid series only (equities, yfinance FX, VIX). Treasury yields are quoted to 1bp and sit unchanged for days in normal markets.
- **Evaluation.** Five corruption definitions as in METHODOLOGY.md. A corruption's footprint is every factor-date whose level or 1-day move differs from the clean data. Per-type rows inject one type at a time at the same positions; the overall row injects all 200.

**Alternatives.**
- Zero the factor's shocks with no Monte Carlo change: Cholesky rejects a zero-variance factor.
- Mask the bad print and reuse the gap policy: the run date itself becomes incomplete and cannot be valued.
- Record exclusions without applying them.
- Same-day FX alignment: larger routine gaps.
- Score only the corrupted print: this penalizes the unavoidable flag on the next day's move, whose shock is equally corrupted.
- Inject all types into one panel for per-type precision: false alarms could not be attributed to a type.

**Consequences.** A held factor contributes no VaR that day, so VaR is understated until the data is fixed. This is visible in run.json and `dq_findings` for the phase 02 data-quality agent. Stale Treasury prints and ×10 spikes on quiet days are largely invisible to the rules; RESULTS.md reports this rather than tuning thresholds on the evaluation window.
