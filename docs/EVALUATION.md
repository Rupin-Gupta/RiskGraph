# Evaluation

How the agent system is measured. Every reported number comes from `reports/metrics/*.json` and is rendered into `docs/RESULTS.md` by `make results`. The market data controls evaluation is described in METHODOLOGY.md.

## Datasets

### Injected incidents (`data/synthetic/incidents/`, SPEC §11.1)

`riskgraph incidents generate --n 100 --seed 42` (`make incidents`) creates 100 cases on 2022–2025 business days, 25 of each type. Every case is on its own date. Before a case is accepted, the full daily risk process (`run-daily` with the case's overrides) must reproduce its expected limit status. A case is also rejected if a critical data-quality finding holds a factor flat on its date, or if a non-bad-data case has any data-quality finding on one of the three factors contributing most to its alerted limit. Such a case would be routed to Data Operations under MDCP-5.1, which would make its label ambiguous. Design decisions are in ADR-010.

| Type | Injection | Expected root cause | Expected action | Validated status |
|---|---|---|---|---|
| Position jump | One new trade (book override), minted by the book generator on the incident date, sized so its desk's VaR reaches 108–160% of the limit | `position_change` | `escalate_to_risk_manager` | breach, positive VaR explain position effect, limit not breached before the trade |
| Market shock | None: a real day on which a limit is breached | `market_move` | `escalate_to_risk_manager` | breach, no data-quality finding on a top-3 contributing factor |
| Bad market data | One run-date print corrupted below the hard-rule thresholds (a spike), so only the Isolation Forest flags it: a warning, and the factor stays live | `bad_market_data` | `route_to_data_ops` | breach caused by the corruption, warning on the corrupted factor, which is a top-3 contributor |
| Control | None. 10 near misses (highest utilization 91–98.5%), 15 quiet days (below 88%) | `no_true_breach` | `no_action` | warning (near miss) or ok |

Each case directory holds:
- `incident.json`: what the agents are given, namely the incident ID, the as-of date, the alerted limit (scope and metric), and the override file names. The alert is the limit with the highest utilization in the validated run.
- `book_override.json` and `market_override.parquet`: present only when the case uses them.
- `ground_truth.json`: labels. It holds the type, root cause, action, expected policy sections, the validated breach row, the injection details, the top contributing factors, and the findings.

Incident IDs are assigned in a random order, so an ID says nothing about the type. The tools read the case's risk run through a hidden `run_id`, and the agents see only dates. So the run ID format (base run vs override run) cannot leak the type.

**Known limitations of the case set:**
- All 25 bad-data cases are EURUSD=X spikes. With this book, these limits, and the frozen phase 01b controls, a single bad print moves 99% historical VaR by only one order statistic. It can breach a limit only on a day when the limit is already near 100%. Among the factors, only EURUSD=X both draws an Isolation Forest warning and stays inside the 200bp cross-source tolerance. The other nine factors produced no qualifying case in 150 tries each. Stale prints (zero shocks) and sign flips never produced one.
- Data-quality findings appear only in bad-data cases: the real dates chosen for the other types carry no natural findings. The set therefore does not test whether an agent ignores an unrelated finding.
- Controls on quiet days all alert on rates VaR, which sits at 85–87% of its limit on most quiet days.

### Policy corpus (`corpus/`, SPEC §9.1)

- Basel Framework MAR (all chapters) and CRE50–CRE55, downloaded by `corpus/download.py` into `corpus/raw/` (not committed): 782 numbered paragraphs (535 MAR, 247 CRE) in 981 chunks; paragraphs over 350 words are split into parts that keep their section ID.
- The synthetic Meridian Bank Market Risk Limit Policy (MRLP, 49 sections) and the Market Data Controls Procedure (MDCP, 20 sections) in `corpus/synthetic/`.
- `corpus/sections.json` lists every indexed `(doc_id, section_id)`. The critic and the checker use it to test whether a citation exists.

## Splits

`data/synthetic/incidents/split.json`: 30 dev and 70 test, stratified by type, with near-miss controls as their own stratum (dev: 7 position, 8 market, 8 bad data, 3 near-miss and 4 quiet controls).

The test split is **frozen**. `test_sha256` is the SHA-256 of the sorted test IDs together with their `ground_truth.json` bytes. It is stored in `split.json` and pinned in `configs/agents.yaml` (`eval.test_split_sha256`, git-tracked). `eval/agents.py` recomputes it before every run and refuses to run if any of the three values differ. Prompts and graph design are iterated on dev only. The test split is run only to produce reported numbers.

## Running

```
make eval-agents SPLIT=dev  VARIANT=multi    RUNS=1
make eval-agents SPLIT=test VARIANT=baseline RUNS=3 DRY_RUN=1   # token and cost estimate only
```

Each incident runs in a fresh thread. The run is auto-approved at the human-approval interrupt, so the evaluation measures the report, not the dispatch. An in-memory checkpointer is used; the CLI uses Postgres. Langfuse traces are tagged `incident_id`, `variant`, and `split`. `DRY_RUN` multiplies the mean tokens per incident from `agents_<variant>_dev.json` by the price in `configs/agents.yaml`. Before any dev run it uses the 60k-token budget ceiling, an upper bound. Outputs:
- `reports/metrics/agents_<variant>_<split>.json`: metrics as mean, std, and per-run values; per-type rows; the confusion counts; the total cost.
- `.config.json`: the resolved configuration.
- `.details.jsonl`: one record per incident and run (report, status, usage, latency, score).

## Metric definitions

All scoring is deterministic, from `IncidentReport` fields against ground truth (SPEC §11.4). A run that ends without a report (`needs_human`, or a crash) scores as wrong on every accuracy metric. Each metric is computed per run over the split's incidents, then reported as the mean and sample standard deviation over runs.

| Metric | Definition |
|---|---|
| Root-cause accuracy | Share of incidents whose `root_cause` equals the expected one; also reported per type |
| Action accuracy | Share whose `recommended_action` equals the expected one |
| False-escalation rate | Share of control and bad-data incidents whose action is `escalate_to_risk_manager` |
| Numeric faithfulness | Evidence items confirmed by the independent checker, divided by all evidence items (pooled over incidents) |
| Citation validity | Citations whose `(doc_id, section_id)` is in the corpus manifest **and** was returned by a `search_policy` call logged for this run, divided by all citations |
| Citation recall ("relevance") | Per incident, the share of expected policy sections that were cited, averaged |
| Citation precision | Per incident, the share of cited sections that are expected, averaged (0 if nothing is cited) |
| Needs-human rate | Share of runs stopped by the token or tool-call budget, or by two structured-output failures |
| Tokens, cost, latency | Mean per incident. Tokens are the model's reported usage, summed over every LLM call in the run. Cost = tokens × the on-demand price in `configs/agents.yaml`. Latency = wall time from invoke to the approval pause |

**Independent numeric checker (`eval/checker.py`).** It is separate from the critic, so the system does not grade itself. For each evidence item, it looks up the logged call by `result_id` and requires that call to belong to this run's incident and thread. It then **re-executes** the tool from the engine's stored outputs and accepts the value only if it equals a number in the fresh result: either within a relative 1e-6, or exactly that number rounded to 3 or more significant figures, after applying the stated unit (`%`, `k`, `mn`, `bn`).

**Expected policy sections.** The ground truth lists 2–3 sections per case:
- position jump: MRLP-5.1 and MRLP-6.4
- market shock: MRLP-5.2 and MRLP-6.5
- both breach types also get the escalation rule for the alerted limit: MRLP-6.1 (desk VaR), MRLP-6.2 (firm VaR), or MRLP-6.3 (stress)
- bad data: MRLP-5.3, MRLP-6.6, and MDCP-5.1
- controls: MRLP-5.4 plus MRLP-4.3 (near miss) or MRLP-4.4 (false alert)

Basel paragraphs may be cited and are checked for validity, but they are not part of the expected set.

**Variants (SPEC §11.3).** `baseline` is a single ReAct agent with every tool. `multi` is the supervisor, specialist, writer, and critic graph. Both use the same model, temperature 0, the same budgets (60k tokens and 25 tool calls per incident), the same intake, and the same report schema. The critic's note-quality judgment (check 4) is the only LLM judgment in the loop, and it is not a reported metric.
