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

## ADR-009: Phase 02 runtime: CPU-only torch and the Bedrock model

**Status:** Accepted (phase 02)

**Context.** `sentence-transformers` (approved for phase 02) depends on `torch`. On Linux, the default PyPI torch wheel pulls about 3 GB of CUDA packages into every CI run. The SPEC fixes Bedrock as the LLM provider but not the model. In `ap-south-1`, this account's on-demand quotas are set per model.

**Decision.**
- `torch` is declared directly and sourced from the PyTorch CPU index (`[tool.uv.sources]`, explicit index), so the lock file has no CUDA packages. torch is already a transitive dependency and is approved for phase 05.
- The agents use `openai.gpt-oss-120b-1:0` in `ap-south-1` (`configs/agents.yaml`; `BEDROCK_MODEL_ID` overrides it). The owner chose it as the lowest-cost capable option: on-demand $0.18 input and $0.71 output per 1M tokens, from the AWS Price List API on 2026-09-19. The price is stored in the config and used for every cost figure.

**Alternatives.** The default torch wheel (slow, large CI). A small proprietary Bedrock model (about 5× the cost). Amazon Nova Lite (cheaper, weaker tool use). A local model through Ollama or llama.cpp (free, but it breaks ADR-001 and needs a new dependency).

**Consequences.** CI stays small. Results are specific to gpt-oss-120b. Cost figures move if AWS changes prices, so each metrics file records its model.

## ADR-010: Incident set design

**Status:** Accepted (phase 02)

**Context.** SPEC §11.1 fixes four types of 25 cases and says what each one injects. It leaves open how to pick dates, how to size injections, what the agents are shown, and how bad data can cause a breach. Under ADR-008, a factor with a critical finding is held flat, which *lowers* VaR, and a stale print is a zero shock.

**Decision.**
- **Generate and validate.** A seeded search proposes cases and keeps one only if the full `run-daily` with its overrides reproduces the expected status (details in EVALUATION.md). Every case has its own date. Rejected candidates leave no files behind.
- **Position jump:** one trade minted by the book generator on the incident date and numbered after the book's own IDs. It is scaled by bisection until its desk's HS VaR reaches a random 108–160% of the limit, then rounded to 0.5M.
- **Market shock:** real breach days, untouched.
- **Bad data:** a single run-date spike. It stays below the Pandera maximum move and inside the 200bp cross-source tolerance, is flagged only by the Isolation Forest (warning, factor stays live), and pushes a limit that was not in breach into breach. Stale prints are not used: a zero shock cannot raise VaR. Sign flips were tried and never qualified. Only EURUSD=X yields such cases, so all 25 are EURUSD spikes. This is documented as a limitation rather than retuning the frozen controls.
- **Control:** 10 near misses and 15 quiet days, no injection.
- **Leakage control.** Incident IDs are shuffled. Agents see only incident ID, date, and alerted limit. Tools resolve the hidden `run_id`. Labels live in a separate `ground_truth.json`.
- **Label safety.** Non-bad-data cases with any data-quality finding on a top-3 contributing factor are rejected, because MDCP-5.1 would route them to Data Operations.
- **Freezing.** The test checksum covers the IDs and their ground-truth bytes. It is pinned in the git-tracked `configs/agents.yaml` as well as in `split.json` (which DVC tracks), and verified before every evaluation.

**Alternatives.**
- Scaling historical factor returns to make market shocks: synthetic regimes with no historical counterpart.
- Corruptions caught by the rules: the factor is held flat, so VaR falls and no breach occurs.
- Corruptions no control flags: the ground truth could not be diagnosed from any tool.
- Distractor findings planted on non-bad-data cases: a stronger test, deferred as extra scope.

**Consequences.**
- The bad-data class is narrow: one factor and one corruption kind.
- Findings appear only in bad-data cases, so the set cannot measure resistance to unrelated findings.
- Market-shock utilizations sit just above 100% (median 100.9%).
- Regenerating incidents or changing the split after the first test run requires the owner's approval.

## ADR-011: Agent runtime, critic, and evaluation choices

**Status:** Accepted (phase 02)

**Context.** SPEC §10–11 fix the graph, tools, critic checks, and metrics. They leave open: how tools identify the run, how evidence is re-fetched, which rule checks the critic applies, how budgets work across parallel branches, how approval tokens are issued, and where cost and latency come from.

**Decision.**
- **Tools** are bound to one incident, and the LLM never sees the `run_id`. `result_id` is a hash of `(run_id, tool, args)`, logged to a new `tool_results` table. Identical calls share one row, and every result can be re-fetched.
- **Intake** fetches the alerted limit deterministically. The report's `incident_id`, `as_of_date`, and `breach` come from intake. The LLM generates the rest (`ReportDraft`). The baseline uses the same intake and assembly.
- **Critic rules** are the policy's own evidence requirements (MRLP-5, MRLP-6.7, ARCHITECTURE.md). Each issue names the agent that must fix it. An evidence mismatch goes to the specialist whose call produced the `result_id`. After 2 correction loops the report goes to human approval with the open issues attached.
- **Budget.** One thread-safe budget object per incident, shared by the parallel branches. Exceeding it stops the run with `needs_human`. So does a second structured-output failure, recorded as a `schema:` reason.
- **Approval token:** HMAC-SHA256 of `thread_id:incident_id` keyed by `APPROVAL_SECRET`. It is issued by the approving command and verified by the dispatch tool.
- **Evaluation** uses an in-memory checkpointer and auto-approves. Tokens come from each call's `usage_metadata`, cost from the price table, and latency from wall time up to the approval pause. Langfuse receives the same traces for inspection, but the metrics are computed in process rather than read back from Langfuse, because of ingestion delay.
- **Retrieval** is the dense baseline only. It embeds chunk text without a contextual prefix and returns the top 5, with an optional `doc_id` filter. Hybrid search, re-ranking, and contextual chunks are phase 05a ablations.

**Alternatives.**
- Reading metrics back from the Langfuse API.
- Separate budgets per specialist.
- LLM-judged evidence checks.
- Letting the critic change the report itself.

**Consequences.**
- The critic's rule checks use engine outputs directly, so the multi-agent variant gets deterministic corrections the baseline does not. That is the design being compared (SPEC §10.9), and it is stated next to every result.
- Numeric faithfulness is scored by a separate checker that re-executes tools, so a critic bug cannot inflate it.

## ADR-012: Gemini API free tier instead of Bedrock

**Status:** Accepted (phase 02). Supersedes the model choice in ADR-009 and changes the LLM line of ADR-001.

**Context.** The AWS account is on the AWS Free plan. Its applied Bedrock quotas are 0 and marked not adjustable, while the AWS defaults are 100M tokens/min for gpt-oss-120b. Lifting that requires upgrading to a paid plan, which the owner declined. The owner wants a hosted model at no cost. Free tiers checked on 2026-09-19:
- Groq: 8k tokens/min, below one request's size.
- Mistral: free API tier withdrawn.
- Cerebras: $5 trial that needs a card.
- Gemini API free tier: no card; per-project request limits.

**Decision.**
- The agents call `gemini-3.5-flash-lite` through Google's OpenAI-compatible endpoint with `langchain-openai` (approved for this change), at temperature 0. It is the only free model on this key with a usable daily quota: 15 RPM and 500 requests/day. The Flash models allow 20 requests/day, and Gemma 4 31B failed with server errors. A client-side rate limiter holds calls at 14 per minute.
- Gemini 3 models require each tool call's `thought_signature` to be sent back on the next turn. `langchain-openai` drops it, so a small subclass (`agents/llm.py`, `GeminiChat`) carries it through.
- Tuned on dev only, and applied equally to both variants to fit the 60k-token budget: batch tool calls into few turns, never re-fetch the intake limit status, state absences in the note rather than as evidence values, and at most 3 policy searches. The ReAct loop is capped at 6 turns, and retrieved sections are trimmed to 700 characters. After the first dev pass (26 multi-agent dev runs): the writer now receives every critic issue, not only its own; the shared definitions state that an anomaly warning on a contributing factor counts as bad data even when the source gap is inside tolerance; the citation guidance names the category, path, and limit-type sections; and the policy agent gets 2 turns.
- Structured output uses function calling on every provider.
- `llm.provider: bedrock` in `configs/agents.yaml` restores ChatBedrockConverse unchanged.
- Free-tier calls are priced at $0; token counts are still reported.
- The evaluation is resumable. Finished runs are appended to the details file as they complete. A spent daily quota stops the run cleanly, and rerunning the same command continues. Metrics are written only once every (incident, run) is done.

**Alternatives.** Upgrade AWS to a paid plan (declined). Local models via Ollama (declined: the owner wants hosted calls). Cerebras (needs a card). Test RUNS=1 to finish sooner (drops mean ± std).

**Consequences.**
- The full evaluation takes several days of daily quota.
- Latency includes rate-limiter waiting, so it measures throughput under the free tier, not model speed.
- Google may use free-tier prompts to improve its products. All data here is synthetic.
- Results are specific to the configured Gemini model, which every metrics file records.

## ADR-013: API incident registry and daily-run semantics

**Status:** Accepted (phase 03)

**Context.** Phase 02 incidents are injected cases in `data/synthetic/incidents/`, and the investigation's state lives only in its LangGraph thread. The API has to turn a real daily run into incidents, list them, and resume them from a browser. The SPEC diagram (§2) investigates a "breach or warning", while the phase 03 prompt says "for any breach". On the base book, firm VaR is in warning on 360 of 995 backtest days (`reports/metrics/backtest.json`), and each investigation costs about 12 LLM requests out of a 500-per-day free quota (ADR-012).

**Decision.**
- A new `incidents` table, owned by the API module, holds `incident_id`, `thread_id`, `as_of_date`, `run_id`, scope, metric, status, and reason. No existing table changes. The thread (PostgresSaver) still holds the report and the pending interrupt; the API resumes it with the existing `resume()` command.
- `POST /runs/daily` investigates **breaches only**. Warnings stay visible on the Risk page.
- The incident ID is `INC-<YYYYMMDD>-<scope>-<metric>`, so rerunning a date is idempotent. Only incidents whose investigation failed are retried, on a fresh thread.
- Without a `date`, the endpoint refreshes market data (`riskgraph data download --refresh`) and runs the panel's latest date. This is the scheduler's path.
- The risk run is synchronous (about 3 s). Investigations run in a FastAPI background task, one after another, behind the shared rate limiter. An API restart marks unfinished investigations `failed`, so the next run of the date retries them.
- A send that fails after approval leaves the thread at `dispatch` with status `dispatch_failed`. Approving again retries the send from the checkpoint, reusing the approved report and token.
- Dispatch always writes the Markdown audit copy. `DISPATCH_MODE=ses` also emails it through SES.

**Alternatives.** Listing incidents by scanning checkpoints (no status index, slow). Investigating warnings as well (quota). A task queue such as Celery (a new dependency and service for one sequential job).

**Consequences.** One API worker runs the investigations, so a burst of breaches is worked through serially. Incidents created with the CLI (`riskgraph investigate`) are not in the registry and do not appear in the dashboard.

## ADR-014: AWS deployment on one instance

**Status:** Accepted (phase 03). Amends the EC2, Bedrock, and TLS lines of SPEC §13.2.

**Context.** The account is on the AWS Free plan: $139.79 of credits, expiring 2026-11-14, and only free-tier-eligible EC2 instance types. The SPEC picks `t3.large`, which is not one of them. Bedrock is unusable on this plan (ADR-012), so the agents call the Gemini API. Basic auth over plain HTTP would put the demo password and the scheduler's token on the wire in the clear, and the project has no domain name.

**Decision.**
- **Instance:** one `m7i-flex.large` (2 vCPU, 8 GB, free-tier eligible, $0.1008/hour in `ap-south-1`) instead of `t3.large` (not eligible, $0.0896/hour), on Ubuntu 24.04 with a 30 GB encrypted gp3 root volume and IMDSv2 required (hop limit 2, so containers can use the role). 8 GB is the floor for Postgres, Weaviate, and the API's CPU torch together.
- **TLS:** a Let's Encrypt certificate for the **Elastic IP itself**, issued with the `shortlived` profile (about 6 days) that supports IP identifiers, renewed by certbot ≥ 5.4 through the webroot the proxy serves. `sslip.io` was rejected: it is not on the Public Suffix List, so every user of it shares one Let's Encrypt rate limit. Certbot renews at half the lifetime, roughly two certificates a week, inside the limit of five.
- **No inbound SSH.** Port 22 stays closed and shell access goes through SSM Session Manager, which the instance role already allows. Only 80 (redirect and ACME) and 443 are open.
- **No Bedrock permission** on the instance role. `GEMINI_API_KEY` comes from Secrets Manager instead.
- **Two secrets:** `riskgraph/app` (a JSON blob the host renders into `.env`) and `riskgraph/api-token`, so the trigger Lambda can read the token without seeing the model and Langfuse keys. $0.80/month.
- **Scheduling is trigger-only.** EventBridge Scheduler invokes the Lambda on weekdays at 18:00 IST; the Lambda checks the instance state first and returns a skip, not an error, when it is stopped, so `make aws-stop` never fires the alarm.
- **Every script is plan-by-default.** `infra/aws/*.sh` prints what it would create and its monthly cost and exits; `--apply` is what changes anything. `make aws-plan` prints them all.
- **DVC's S3 extra runs through `uvx`.** `dvc[s3]` pins `botocore` through `aiobotocore` in a range that contradicts the `boto3` this project already uses, so the lockfile keeps plain `dvc` and S3 transfers use an isolated environment (`make data`, `make data-pull`, and the host bootstrap).

**Alternatives.** `t3.large` (likely refused by the Free plan). `c7i-flex.large` (eligible but 4 GB, too small). HTTP-only with basic auth (credentials in the clear). A self-signed certificate (browser warnings on a portfolio link). A registered domain (a yearly cost, and not needed for an IP certificate). A NAT gateway, load balancer, or RDS (all forbidden by the cost rules).

**Consequences.**
- The site is `https://<elastic-ip>`, which is trusted but not a memorable name. Moving to a domain later only changes the certificate step.
- The Elastic IP costs $3.65/month even while the instance is stopped; the EBS volume adds $2.74. Idle cost is therefore about $6.40/month, and running costs about $2.42/day.
- If the instance stays stopped for more than about six days the certificate expires; the renewal cron runs at boot as well, so the first start refreshes it.
- Results and behavior are unchanged from the local stack: the same images run in both places.
