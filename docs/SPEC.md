# RiskGraph — Technical Specification

Version 1.0. Owner: Rupin Gupta. This document is the source of truth for design. Changes go through `docs/DECISIONS.md`.

---

## 1. Goal

Build a system that does what a bank's traded-risk team does every day, with an AI layer that investigates problems:

1. Price a multi-asset trading book and compute risk measures (VaR, ES, Greeks, stress, counterparty exposure).
2. Control the quality of the market data feeding those numbers.
3. Monitor risk against limits and detect breaches.
4. For each breach, have a multi-agent system determine the root cause, gather evidence, cite the relevant regulation and policy, and draft an escalation note.
5. Require human approval before anything is sent.
6. **Measure everything** against baselines on held-out data.

Target audiences: Celebal Technologies (Data Scientist Intern: ML, GenAI, agentic AI) and HSBC (Traded Risk Analyst Intern: market/counterparty risk plus LLM and agentic workflows). Coverage map in Appendix A.

---

## 2. System overview — the daily run

```
market data (yfinance, FRED) ──► data controls (Pandera + Isolation Forest + cross-source)
                                          │
trade book (synthetic; later: extracted   ▼
from confirmation PDFs) ──────────► pricing + risk engine ──► risk_results, limit_status (Postgres)
                                          │
                                  breach or warning?
                                          │ yes
                                          ▼
              LangGraph: supervisor → [attribution | data-quality | policy] → writer → critic
                                          │ (critic fails → back to specialist, max 2 loops)
                                          ▼
                              human approval (interrupt, persisted)
                                  │ approve            │ reject
                                  ▼                    ▼
                    SES email + memory write        close with reason
```

Triggered on weekdays by EventBridge Scheduler (in production) or `make risk-run DATE=...` (locally).

---

## 3. Trading book and market data

### 3.1 Book (`configs/book.yaml`, generator in `book/`)
Reporting currency USD. About 50 trades across three desks, eight fictional counterparties (CP01–CP08, generated names), one netting set per counterparty.

| Desk | Instruments | Count (approx.) | Counterparty risk? |
|---|---|---|---|
| FX | FX forwards EUR/USD and USD/INR (1M–2Y); FX spot positions | 12 forwards, 4 spot | Forwards: yes (OTC) |
| Rates | USD fixed-float interest rate swaps (2Y/5Y/10Y, payer and receiver); US Treasury positions (2Y/5Y/10Y) | 6 swaps, 3 bonds | Swaps: yes (OTC) |
| Equity derivatives | European options on SPY, AAPL, MSFT, JPM (calls/puts, 1M–1Y); cash equity hedges | 16 options, 5 equity | No (treated as exchange-traded) |

Trade schema (Pydantic `Trade`): `trade_id, desk, instrument_type, counterparty_id, currency, notional, trade_date, maturity_date`, plus instrument fields (`strike, option_type, underlying, fixed_rate, pay_receive, coupon, currency_pair, forward_rate`).

**Simplification:** for backtesting, the book is a constant-maturity book: tenors are measured relative to the valuation date, so its composition is stable across the backtest window. Documented in Appendix B.

### 3.2 Market data (`marketdata/`)
Daily history from 2015-01-01 to the latest available date.

| Series | Source | Use |
|---|---|---|
| SPY, AAPL, MSFT, JPM (OHLC + close) | yfinance | Equity factors; OHLC for volatility targets |
| EURUSD=X, INR=X | yfinance | FX factors |
| DGS2, DGS5, DGS10 | FRED | USD curve points |
| VIXCLS | FRED | Implied-volatility proxy for SPY options |
| DEXUSEU, DEXINUS | FRED | Second source for FX cross-checks |

Raw pulls go to `data/raw/<source>/<series>.parquet` and are DVC-tracked. Processed, aligned business-day panel: `data/processed/market_panel.parquet`.

---

## 4. Pricing and risk engine

### 4.1 Pricing (`pricing/`)
- **Options:** Black-Scholes (European). Vol for SPY options = VIX level / 100; for single-stock options = forecast vol from the selected vol model (phase 05b), EWMA vol before that. Greeks: delta, gamma, vega, theta (analytic).
- **FX forward:** discounted difference between contract rate and forward (covered interest parity; foreign rate proxy documented).
- **IRS:** single-curve valuation off the Treasury curve (linear interpolation of 2Y/5Y/10Y, flat extrapolation). DV01 by 1bp parallel bump.
- **Bond:** fixed-coupon bond priced off the same curve. DV01 by bump.
- Known-value tests are mandatory: Black-Scholes reference values, put-call parity, swap PV ≈ 0 at the par rate, bond price = par when coupon = yield, FX forward PV = 0 at the fair forward.

### 4.2 Risk factors
Equity and FX: log returns. Curve points: absolute changes in bp. VIX: log changes. Shocks are applied to today's market state; every trade is fully revalued per scenario.

### 4.3 VaR and ES (`risk/var.py`), parameters in `configs/risk.yaml`
- **Historical simulation:** 1-day, 99%, 500-day window, full revaluation. ES at 97.5% over the same scenarios.
- **Parametric (delta-normal):** sensitivities × covariance (EWMA, λ = 0.94). Comparison only.
- **Monte Carlo:** 10,000 correlated scenarios via Cholesky of the covariance matrix; vols from EWMA or a vol model (phase 05b); full revaluation.
- Outputs per desk and firm-wide.

### 4.4 Stress testing (`configs/scenarios.yaml`)
- Historical: worst day in the 2020-03-09 to 2020-03-20 window; the largest 1-day curve move in 2022.
- Hypothetical: +200bp parallel; USD/INR +10%; equities −20% with VIX +50%.

### 4.5 Backtesting (`risk/backtest.py`)
Hypothetical P&L (static positions, next-day market move) vs the prior day's 99% VaR over the last 250 days. Report exception count, **Kupiec POF** test, **Christoffersen independence** test, and the Basel traffic light (green 0–4, yellow 5–9, red ≥10 exceptions).

### 4.6 Limits (`configs/limits.yaml`, `risk/limits.py`)
Desk VaR limits, firm VaR limit, stress-loss limits, and counterparty PFE limits (phase 06). Utilization = metric / limit. Status: `ok` < 90% ≤ `warning` < 100% ≤ `breach`. Limits are calibrated so that breaches are rare but occur in the history (document the calibration).

### 4.7 VaR explain and attribution (`risk/explain.py`)
Day-over-day VaR change decomposed as:
- **Position effect:** VaR(positions_t, market_{t−1}) − VaR(positions_{t−1}, market_{t−1})
- **Market effect:** VaR(positions_{t−1}, market_t) − VaR(positions_{t−1}, market_{t−1})
- **Interaction:** the residual

Top contributors by position and risk factor via Euler-style allocation: the average position P&L across the tail scenarios nearest the VaR quantile (5 scenarios either side, configurable).

### 4.8 Daily run
`riskgraph run-daily --date D [--book-override PATH] [--market-override PATH]` writes `risk_results` (date, desk, metric, value) and `limit_status` (date, scope, metric, value, limit, utilization, status) to Postgres, plus a JSON artifact per run. The override flags are used by incident injection (§11).

---

## 5. Market data controls (`marketdata/controls.py`)

- **Pandera rules:** positive prices; no nulls on business days; monotonic date index; yields within [−1%, 20%]; maximum absolute 1-day move per factor (configurable).
- **Staleness:** unchanged value for 3 or more consecutive business days on a liquid series.
- **Cross-source check:** yfinance vs FRED for EUR/USD and USD/INR, tolerance in bp.
- **Isolation Forest** on features: return z-score, next-day reversal, rolling vol ratio, cross-source difference.
- Findings written to `dq_findings` (date, factor, check, severity, detail).

**Evaluation (`eval/dq.py`):** inject 200 corruptions into a held-out copy of the panel (stale run, ×10 spike, sign flip, missing value, decimal shift). Report precision, recall, and F1 per corruption type for rules-only vs rules + Isolation Forest.

---

## 6. Counterparty exposure (`exposure/`, `cpp/`)

- Scope: OTC trades only (FX forwards, IRS), aggregated per netting set, no collateral (Appendix B).
- **Simulation:** FX under GBM (drift = rate differential, vol from history); short rate under Vasicek, with κ, θ, σ estimated by OLS on the discretized AR(1) form using 2Y yield history. Time grid: weekly for 1 year, then monthly to the longest maturity.
- At each node and path, revalue trades; netting-set exposure = max(Σ MTM, 0).
- Outputs: EE(t); PFE(t) at 95%; Effective EE (non-decreasing running max of EE); **EEPE** = time-weighted average of Effective EE over the first year (or to maturity if shorter).
- **C++ engine:** path simulation and aggregation kernel in C++ exposed via pybind11 (built with scikit-build-core + CMake).
  - Correctness test: given identical pre-generated normals, C++ output equals NumPy output within 1e-10.
  - Benchmark: wall time at 10k and 50k paths vs vectorized NumPy.
- Counterparty PFE limits feed `limit_status`, so counterparty breaches can also trigger incidents.

---

## 7. Volatility models (`volmodels/`)

- **Target:** realized volatility over the next 5 business days (annualized), per equity and FX series. Use Garman-Klass on OHLC where available, squared log returns otherwise.
- **Features:** HAR lags (1-day, 5-day, 22-day realized vol), returns, absolute returns, VIX for equities.
- **Models:** GARCH(1,1) (`arch`); HAR-RV linear regression; LightGBM tuned with Optuna; PyTorch LSTM; 1D-CNN (dilated causal convolutions, TCN-style); small Transformer encoder.
- **Protocol:** walk-forward, expanding window, monthly retrain. Metrics: QLIKE and RMSE. Everything tracked in MLflow.
- **Downstream test:** plug each model's forecasts into Monte Carlo VaR and compare backtest exception counts and Kupiec results.
- **Report honestly.** If deep models don't beat HAR, say so. That is a legitimate finding and a good interview answer.

---

## 8. Trade confirmation extractor (`extraction/`, `notebooks/kaggle/`)

### 8.1 Data
- Generator (reportlab) produces confirmations for FX forwards, IRS, options, and bonds across **8 layout templates**. Templates vary field order, label synonyms ("Notional Amount" / "Principal"), date formats, tables vs key-value blocks, and single vs multi-page.
- About 3,000 documents; 15% have deliberately missing optional fields (gold value `null`).
- Scanned variant: render to image, rotate ±2°, blur, add JPEG artifacts → OCR path.
- **Split by template:** train on templates 1–6, test on templates 7–8 (unseen layouts). Also report a within-template test for contrast.
- Text: `pdfplumber` for digital PDFs, Tesseract for scanned.

### 8.2 Models
- Schema: Pydantic `TradeConfirmation` mirroring §3.1 fields; every field nullable.
- Baselines: regex/rules extractor; base model zero-shot; base model few-shot; (optional) a large Bedrock model zero-shot as an upper reference.
- **SFT:** `meta-llama/Llama-3.2-3B-Instruct` (fallback: `Qwen/Qwen2.5-3B-Instruct`) with TRL `SFTTrainer` + PEFT QLoRA: 4-bit NF4, fp16 (T4 has no bf16), r=16, alpha=32, dropout 0.05, target `q,k,v,o,gate,up,down` projections, max sequence length 2048, 2–3 epochs. Log to W&B.
- **DPO:** TRL `DPOTrainer`, β=0.1. Chosen = gold JSON (nulls where the field is absent). Rejected = the same JSON with 1–3 fields replaced by plausible fabricated values, prioritizing fields that are null in gold. If time allows, add on-policy rejected samples from the SFT model's own dev-set mistakes.
- **Quantization and serving:** merge LoRA → FP16 → GGUF `Q4_K_M` (llama.cpp) for CPU serving on EC2; AWQ 4-bit (llm-compressor, or AutoAWQ if it installs cleanly) for vLLM. Serve GGUF with llama.cpp server using JSON-schema-constrained decoding.

### 8.3 Metrics (`eval/extractor.py`)
Field-level precision, recall, and F1 after normalization (ISO dates, parsed numbers, currency codes); document exact match; JSON validity rate; **hallucination rate** (fields predicted non-null where gold is null, or values not found in the source text). Benchmarks: vLLM FP16 vs AWQ throughput on a Kaggle T4; GGUF latency on CPU.

---

## 9. Policy RAG (`rag/`, `corpus/`)

### 9.1 Corpus
- Basel Framework chapters **MAR** (market risk) and **CRE50–CRE55** (counterparty credit risk), fetched by `corpus/download.py` from the BIS Basel Framework site. Stored locally; **not committed**.
- Synthetic **"Meridian Bank Market Risk Limit Policy"** (about 10–15 pages, numbered sections, clearly labeled synthetic) and **"Market Data Controls Procedure"**, written in `corpus/synthetic/` as Markdown.

### 9.2 Pipeline
- **Structure-aware chunking** by section and paragraph, keeping IDs (e.g. `MAR33.4`, `MRLP-4.2`) as metadata so citations can be checked deterministically.
- **Contextual chunks:** prepend document title and section path; optionally a 1–2 sentence LLM-generated context (Bedrock, cached to disk).
- **Embeddings:** `BAAI/bge-small-en-v1.5` on CPU. Weaviate collection with bring-your-own vectors plus BM25; hybrid query with `alpha` tuned on dev.
- **Re-ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` by default (top 20 → top 5); `BAAI/bge-reranker-base` as an ablation.
- **Agentic RAG:** the policy agent writes its own queries, rewrites them when the top re-ranker score is below a threshold, and stops after at most 2 retrieval rounds.

### 9.3 Gold set and metrics (`eval/rag.py`)
- 100 questions (60 regulatory, 40 policy), each with gold section IDs and a reference answer. LLM drafting is allowed, but each item needs `verified_by_human: true` from Rupin. **Unverified items are excluded from all metrics.** Split: 30 dev / 70 test.
- Deterministic retrieval metrics: hit@5 and MRR@10 on gold section IDs.
- RAGAS: faithfulness, answer relevancy, context precision, context recall (Bedrock as judge; estimate cost first).
- **Ablation:** dense → BM25 → hybrid → + re-ranker → + contextual chunks → + agentic query rewriting.

---

## 10. Agent system (`agents/`)

### 10.1 Graph (LangGraph `StateGraph`)
`intake → supervisor → (fan-out via Send) attribution | data_quality | policy → writer → critic → [fail: back to the named specialist, max 2 loops] → human_approval (interrupt) → dispatch | close`

| Node | Pattern | Role |
|---|---|---|
| Supervisor | Plan-and-execute | Reads the breach, plans which specialists to call and what to ask each |
| Attribution | ReAct tool use; **Tree-of-Thought from phase 05c** | Explains why the metric moved: position vs market effect, top contributors, new trades |
| Data quality | ReAct tool use | Decides whether the move is driven by bad market data |
| Policy | Agentic RAG | Retrieves and cites the relevant limit-policy and Basel sections; determines the required escalation path |
| Writer | Structured generation | Produces the `IncidentReport` and a draft note for a risk manager |
| Critic | Reflection / self-correction | Verifies numbers and citations; returns specific fixes to the responsible specialist |

LLM: Bedrock through `langchain_aws.ChatBedrockConverse`, model ID from `configs/agents.yaml` / env, temperature 0. Budget per incident: 60k tokens and 25 tool calls. Exceeding either stops the run with status `needs_human`.

### 10.2 Tools (typed, deterministic, read-only except dispatch)
`get_limit_status(date, scope)`, `get_risk_results(date, desk)`, `run_var_explain(date, desk)`, `get_top_contributors(date, desk, k)`, `list_new_trades(date, desk)`, `get_trade(trade_id)`, `get_greeks(date, desk)`, `run_stress(date, scenario)`, `get_dq_findings(date, factor)`, `compare_sources(factor, date)`, `search_policy(query, filters)`, `get_similar_incidents(summary)` (phase 05c), `send_escalation_email(incident_id, approval_token)` (dispatch node only).

Each agent has a tool allowlist. Tool results carry a `result_id` so evidence can reference them.

### 10.3 Output schema (Pydantic `IncidentReport`)
```
incident_id, as_of_date,
breach: {scope, metric, value, limit, utilization},
root_cause: enum[position_change, market_move, bad_market_data, no_true_breach, unknown],
confidence: float,
evidence: [{claim, value, unit, result_id}],
recommended_action: enum[escalate_to_risk_manager, route_to_data_ops, no_action],
policy_citations: [{doc_id, section_id}],
draft_note: str
```

### 10.4 Critic checks
1. Deterministic: every evidence value is re-fetched via its `result_id` and compared (relative tolerance 1e-6 for engine outputs; rounding tolerance for displayed values).
2. Deterministic: every citation `(doc_id, section_id)` exists in the corpus and appeared in the policy agent's retrieved contexts.
3. Rule consistency: e.g. `bad_market_data` requires at least one DQ finding on a contributing factor; `position_change` requires a positive position effect.
4. LLM check: is the note clear, correctly prioritized, and actionable for a risk manager?

### 10.5 Human in the loop
`interrupt()` before dispatch. Persisted with LangGraph's `PostgresSaver`, so paused incidents survive restarts. Resumed via `Command(resume={decision, edits})` from the CLI (phase 02) or the API (phase 03). Dispatch requires a server-issued approval token.

### 10.6 Tree-of-Thought (phase 05c)
The attribution agent generates up to 3 competing hypotheses (for example "new trade", "market move", "bad price"). For each, it calls tools to gather confirming and disconfirming evidence, scores each hypothesis, prunes, and selects. Compared against the linear chain in the ablation.

### 10.7 Memory (phase 05c)
After approval, store `{summary, root_cause, resolution, as_of_date}` in a Weaviate `IncidentMemory` collection. At intake, retrieve the top 3 similar past incidents as **hints only**; the critic rejects any evidence sourced from memory rather than tools. Evaluated by running test incidents in chronological order with memory on vs off.

### 10.8 Guardrails
Pydantic validation on all outputs; per-agent tool allowlists; no dispatch without an approval token; retrieved text wrapped and labeled as untrusted data in prompts; token and tool-call budgets.

### 10.9 Baseline
A single ReAct agent with all tools, the same budget, and the same output schema.

### 10.10 Observability
Langfuse traces per incident, tagged `incident_id`, `variant`, `split`. Token counts, cost, and latency are exported to `reports/metrics/agents_*.json`.

---

## 11. Incident injection and evaluation (`incidents/`, `eval/agents.py`)

### 11.1 Cases
100 cases on dates in 2022–2025, generated with `riskgraph incidents generate --n 100 --seed 42`:

| Type | Count | Injection | Expected root cause | Expected action |
|---|---|---|---|---|
| Position jump | 25 | Add a large new trade (book override) | `position_change` | `escalate_to_risk_manager` |
| Market shock | 25 | Pick real high-volatility dates, or scale factor returns | `market_move` | `escalate_to_risk_manager` |
| Bad market data | 25 | Corrupt one factor (stale, spike, sign flip) via market override | `bad_market_data` | `route_to_data_ops` |
| Control | 25 | No injection; 10 of these are near-misses at 90–99% utilization | `no_true_breach` | `no_action` |

Each case stores `book_override`, `market_override`, and `ground_truth.json` (root cause, action, expected policy section IDs). **Split: 30 dev / 70 test, stratified by type; test is frozen.**

### 11.2 Metrics
- Root-cause accuracy (overall and per type)
- Action accuracy; false-escalation rate on control and bad-data cases
- **Numeric faithfulness:** share of evidence values confirmed by an **independent checker script** (not the critic, to avoid self-grading)
- Citation validity (exists and was retrieved) and citation relevance (cited sections ∩ expected sections)
- Mean tokens, cost, and latency per incident (from Langfuse)

### 11.3 Variants and runs
Single-agent ReAct baseline; multi-agent; + Tree-of-Thought; + memory; optional CrewAI re-implementation (phase 07).
- Headline comparison (baseline vs best variant): 3 runs each on test, report mean ± std.
- Ablations: 1 run each on test.
- **Estimate the token cost before any test run and ask for approval if it exceeds $2.**

### 11.4 Scoring
All scoring is deterministic from `IncidentReport` fields vs ground truth. The only LLM judgment is the critic's note-quality check, which is not a reported metric.

---

## 12. Approved dependencies by phase

Pin to the latest stable version at install time and record in `uv.lock`. Anything not listed requires approval.

| Phase | Packages |
|---|---|
| 00–01 | numpy, pandas, scipy, pyarrow, pydantic, pandera, scikit-learn, yfinance, fredapi, typer, pyyaml, python-dotenv, psycopg[binary], sqlalchemy, matplotlib, pytest, ruff, mypy, pre-commit, dvc |
| 02 | langgraph, langchain-core, langchain-aws, langgraph-checkpoint-postgres, weaviate-client, sentence-transformers, pdfplumber, langfuse, boto3, tiktoken (token estimates) |
| 03 | fastapi, uvicorn, httpx; frontend: next, react, typescript, tailwindcss, recharts; dvc[s3] |
| 04 | reportlab, pytesseract, pypdfium2, Pillow; Kaggle notebooks only: transformers, trl, peft, bitsandbytes, accelerate, datasets, wandb, vllm, llm-compressor or autoawq |
| 05 | ragas, arch, lightgbm, optuna, torch (CPU locally), mlflow, statsmodels |
| 06 | pybind11, scikit-build-core, cmake |
| 07 | crewai; Terraform (CLI); a vision-language model via transformers (Kaggle only) |

---

## 13. Infrastructure and deployment

### 13.1 Local
`docker-compose.yml`: postgres, weaviate, api, frontend (nginx serving the static export), llamacpp (from phase 04). MLflow runs locally, outside Compose, with S3 artifacts from phase 03.

### 13.2 AWS (phase 03). One resource at a time, each with approval.
- **EC2:** 1× `t3.large` (2 vCPU, 8 GB), Ubuntu LTS, 30 GB gp3. Security group allows 22 from Rupin's IP only, and 80/443. The same Compose stack runs behind nginx; basic auth on the demo.
- **S3:** `riskgraph-data-<suffix>` (DVC remote) and `riskgraph-artifacts-<suffix>` (MLflow artifacts, GGUF model). Block all public access.
- **ECR:** repositories for `api` and `frontend`. The llama.cpp server uses its public image and pulls the GGUF model from S3 at startup.
- **Bedrock:** model access enabled in-region; the instance role gets `bedrock:InvokeModel` on the chosen model only.
- **SES** (sandbox): verified sender and recipient; `ses:SendEmail` from the instance role.
- **Secrets Manager:** Langfuse keys, FRED key, API token.
- **EventBridge Scheduler → Lambda (Python):** weekdays 18:00 IST; POSTs `/runs/daily` with a token from Secrets Manager.
- **CloudWatch:** Docker `awslogs` driver; an alarm on Lambda errors and a failed-run metric.
- **CI/CD:** GitHub Actions.
  - On PR: ruff, mypy, pytest.
  - On main: build and push to ECR via an OIDC-assumed role (no static keys), then deploy with SSM Run Command (`docker compose pull && docker compose up -d`). The instance role includes `AmazonSSMManagedInstanceCore`.
- **Cost rules:**
  - AWS Budgets alerts at $5 and $15.
  - `make aws-stop` whenever idle.
  - No NAT gateway, load balancer, OpenSearch, SageMaker endpoint, RDS, or GPU instance.
  - Note that public IPv4 addresses are billed hourly.
- All AWS commands live as reviewed scripts in `infra/aws/`. Rupin approves each execution.

### 13.3 Optional Terraform (phase 07)
`infra/terraform/` mirrors §13.2 so the whole environment can be rebuilt with one command.

---

## 14. Documentation deliverables

- `README.md`: problem, demo video link, Mermaid architecture diagram, results table (embedded from `RESULTS.md`), and how to run it.
- `docs/ARCHITECTURE.md`: components, data flow, deployment.
- `docs/METHODOLOGY.md`: risk math, model choices, evaluation protocol, and **assumptions and limitations**.
- `docs/DECISIONS.md`: ADRs (ADR-001 is the stack).
- `docs/EVALUATION.md`: datasets, splits, metrics definitions.
- `docs/RESULTS.md`: generated only.

---

## 15. Phase plan

| Phase | Prompt | Core deliverable | Done when |
|---|---|---|---|
| 0 | 00_setup | Scaffold, CI, data ingestion | Fresh clone: `make setup test lint data` pass; CI green |
| 1a | 01a_risk_engine | Pricing, VaR/ES, stress, backtest, limits, VaR explain, daily run | Known-value tests pass; backtest metrics JSON written |
| 1b | 01b_market_data_controls | Controls + anomaly detection + eval | DQ metrics JSON written |
| 2 | 02_agents_eval | Incidents, dense RAG baseline, agent graph, HITL, baseline, eval harness | Test-split metrics for baseline and multi-agent |
| 3 | 03_api_dashboard_aws | API, dashboard, AWS deployment, CI/CD, email, scheduler | Live URL; approve-and-send works end to end |
| 4 | 04_extractor | Confirmation generator, Kaggle notebooks, eval, llama.cpp serving | Extractor metrics JSON (base / SFT / SFT+DPO) |
| 5a | 05a_rag | Hybrid, re-ranking, contextual chunks, agentic RAG, gold set, ablation | RAG ablation JSON |
| 5b | 05b_vol_models | Six vol models, walk-forward, VaR impact | Vol comparison JSON |
| 5c | 05c_tot_memory | Tree-of-Thought, memory, agent re-evaluation | Agent ablation JSON |
| 6 | 06_exposure_cpp_docs | PFE/EEPE, C++ engine, docs, demo | Exposure + benchmark JSON; docs complete |
| 7 | 07_stretch | Terraform, CrewAI comparison, vision-language extractor | Optional |

---

## Appendix A — Job-description coverage

| Requirement | Where it's proven |
|---|---|
| Supervised, unsupervised, ensemble ML and trade-offs | LightGBM, Isolation Forest, HAR/GARCH baselines; DECISIONS.md |
| CNN, RNN/LSTM, Transformers | Vol models (§7) |
| Instruction tuning, LoRA/QLoRA, DPO (TRL) | Extractor (§8) |
| Quantization (AWQ, GGUF), vLLM | Extractor serving and benchmarks (§8.2–8.3) |
| Agentic patterns: tool use, function calling, ReAct, planning, ToT, memory, reflection, multi-agent | Agent system (§10) |
| RAG, vector DB, hybrid search, re-ranking, agentic RAG | Policy RAG (§9) |
| LangChain, LangGraph (CrewAI optional) | §10, phase 07 |
| Deployment: FastAPI, Docker, CI/CD, AWS; observability and token cost | §13, §10.10 |
| MLOps: MLflow, DVC, W&B | §7, §3.2, §8.2 |
| Statistics, probability, linear algebra, optimization | VaR/ES, Kupiec/Christoffersen, Cholesky, Optuna |
| HSBC: VaR, ES, PFE, EEPE, stress, sensitivities | §4, §6 |
| HSBC: limit monitoring, breaches, escalation | §4.6, §10 |
| HSBC: market data validation, data lineage, controls | §5, DVC |
| HSBC: Python + C++, full-stack | §6 (pybind11), phase 03 dashboard |
| Communication in business terms | Escalation notes, METHODOLOGY.md, demo |

## Appendix B — Known simplifications (state these openly in METHODOLOGY.md)

- The trading book is synthetic and constant-maturity for backtesting.
- Single-curve USD valuation off Treasury CMT yields; no OIS/SOFR curve.
- No implied-vol surface: VIX proxies SPY implied vol; single stocks use forecast vol.
- Counterparty exposure: GBM for FX and Vasicek for rates, no collateral or margin period of risk, options treated as exchange-traded.
- Incidents are injected, so real-world breach causes may be messier than these four categories.
- Regulatory text is used for a non-commercial portfolio project and is not redistributed.
