# Methodology

Numbers quoted as results live in `docs/RESULTS.md`, generated from `reports/metrics/*.json`. This document describes methods and parameters (from `configs/`), not outcomes.

## Risk math

### Trading book

The synthetic book (`riskgraph book generate --seed 42`, config `configs/book.yaml`) holds 46 trades on three desks: FX forwards and spot (FX), USD fixed-float swaps and Treasury bonds (rates), and European options and cash equity hedges on SPY, AAPL, MSFT, and JPM (equity derivatives). Eight fictional counterparties each have one netting set. Every trade is struck on the book date (2024-01-02): FX contract rates within 1% of the fair forward, swap rates within 25bp of par, bond coupons at the curve yield rounded to 1/8%, and strikes within 10% of spot.

**Constant-maturity book (ADR-004).** To keep the book comparable across a multi-year backtest, it is re-held on every valuation date:

- Tenors keep their trade-date length (a 5Y swap is always a 5Y swap).
- Option strikes and FX contract rates keep their trade-date moneyness: they scale by level(valuation date) / level(trade date).
- Equity and option positions keep their USD notional: units held = notional / spot(valuation date).

A trade valued on its own trade date is priced exactly as booked. Risk therefore changes with volatility and correlation, not with price levels or the passage of time.

### Pricing

| Instrument | Model | Units |
|---|---|---|
| European option | Black-Scholes, no dividends; rate from the curve at expiry, converted to continuous compounding | Vol: SPY = VIX / 100; AAPL, MSFT, JPM = EWMA vol (annualized, λ = 0.94, 500-day window) |
| FX forward and spot | Covered interest parity: F = S · DF_base / DF_quote; PV = N · (F − K) · DF_quote, converted to USD at spot | Foreign rates are constant proxies: EUR 2%, INR 6.5% (ADR-005) |
| Interest rate swap | Single curve, valued on a reset date: float leg = N · (1 − DF(T)); fixed leg semi-annual | DV01 = PV(curve + 1bp) − PV(curve), USD |
| Treasury bond | Semi-annual coupons discounted off the curve (dirty PV) | Same DV01 convention |

**Curve.** The DGS2, DGS5, and DGS10 Treasury constant-maturity yields are treated as zero rates with semi-annual compounding, DF(t) = (1 + z(t)/2)^(−2t). z(t) is linear in t between pillars and flat outside them. Year fractions are ACT/365.25.

**Known-value tests** (`tests/pricing/`): three published Black-Scholes cases (Hull Example 15.6; Hull's Greek-letters example for price, delta, gamma, vega, and theta; the S = K = 100 textbook case), put-call parity, analytic Greeks against central finite differences (relative tolerance 1e-4), swap PV = 0 at the par rate (which equals the flat curve rate), bond at par when coupon = yield, bond DV01 against modified duration, and FX forward PV = 0 at the fair forward.

### Risk factors and gap policy

Ten factors: SPY, AAPL, MSFT, JPM, EURUSD, USDINR (log returns), DGS2, DGS5, DGS10 (absolute changes in bp), and VIX (log changes). Shocks are applied to the valuation date's market state, and every trade is fully revalued in every scenario (vectorized with NumPy; no loop over scenarios or trades).

**Gap policy.** A date is used only if all ten factors are observed. Incomplete dates (exchange or FRED holidays, missing prints) are dropped, so the next shock spans the gap. Each run lists the dropped dates inside its scenario window in `run.json`. Missed prints (as opposed to holidays) are also reported by the market data controls below.

### VaR and ES

All figures are 1-day, in USD, reported as positive losses, per desk and firm-wide.

- **Historical simulation (limit metric):** the last 500 daily shocks, full revaluation. VaR 99% is the empirical quantile: the ⌈n · 1%⌉-th worst P&L (the 5th worst of 500). ES 97.5% averages the worst n · 2.5% scenarios, giving the boundary scenario its fractional weight (12.5 scenarios of 500).
- **Delta-normal (comparison only):** VaR = z₀.₉₉ · √(sᵀ Σ s). The sensitivities s are central finite differences of full revaluation, and Σ is the zero-mean EWMA covariance (λ = 0.94) of the same 500 shocks.
- **Monte Carlo:** 10,000 draws from N(0, Σ) via the Cholesky factor of the EWMA covariance, with full revaluation. The generator is seeded with (seed, date), so each date's MC VaR is reproducible.

Single-stock option vols (EWMA) are model parameters and stay fixed within a scenario. SPY option vol moves with the VIX factor.

### Stress testing

Scenarios in `configs/scenarios.yaml` are replayed on today's book with full revaluation:

- **Historical:** the day in 2020-03-09 to 2020-03-20 with the worst firm P&L on today's book, and the 2022 day with the largest absolute 1-day move of any curve point. Both replay every factor's move on that day.
- **Hypothetical:** +200bp parallel; USD/INR +10%; equities −20% with VIX +50%.

The stress-loss metric is the largest loss across scenarios.

### Backtesting

`riskgraph backtest --start 2022-01-01` computes, for each date t, the VaR set on the prior date d and the **hypothetical P&L** from d to t. That P&L uses d's positions and market state with t's realized factor moves applied (static positions, no time decay, single-stock vols held at d's value), which is the same factor set the VaR models. An exception is a loss strictly larger than VaR.

- **Kupiec POF:** LR = −2 ln[(1−p)^(T−x) p^x] + 2 ln[(1−x/T)^(T−x) (x/T)^x], χ²(1), p = 1%.
- **Christoffersen independence:** a likelihood ratio of a first-order Markov chain of exceptions against independence, χ²(1).
- **Basel traffic light** over 250 days: green 0–4, yellow 5–9, red 10 or more exceptions.

Statistics are reported for the Basel window (the last 250 days) and the full period, for historical and Monte Carlo VaR. Both tests are checked against hand-computed values in `tests/risk/`.

### Limits and calibration

Utilization = metric / limit. Status is ok below 90%, warning from 90%, and breach from 100%. Limit metrics: historical-simulation VaR 99% per desk and firm, and the firm stress loss.

Limits are calibrated so breaches are rare but present in 2022–2025. Each limit is the 98th percentile of that metric's daily history over 2022-01-01 to 2025-12-31 (from the backtest run), rounded to three significant figures. Coarser rounding (two figures) put the firm limit above the entire history, which would leave no breaches. The backtest re-derives the calibrated values and counts breach and warning days under the configured limits (RESULTS.md, Limits table).

Because the book is constant and the window is 500 days, historical VaR moves slowly. A limit that binds at all therefore leaves a smooth series like the rates desk's in the warning band on many days.

### VaR explain

The day-over-day change in historical VaR is split into:

- position effect = VaR(positions_t, market_{t−1}) − VaR(positions_{t−1}, market_{t−1})
- market effect = VaR(positions_{t−1}, market_t) − VaR(positions_{t−1}, market_{t−1})
- interaction = total − position − market (the residual, so the three sum to the total exactly)

positions_{t−1} is the base book, and positions_t is the run's book (a book override, if any). Top contributors use Euler-style allocation: each trade's mean loss over the 11 scenarios ranked around the VaR scenario (5 either side). Factor contributions replay each factor's move alone over the same scenarios; the part the one-at-a-time losses leave unexplained is reported as `cross_effects`.

## Market data controls

`marketdata/controls.py` checks the processed panel (all twelve series) before each daily run. Parameters are in `configs/risk.yaml` under `controls:`. They were set from 2015–2021 data only; the 2022–2025 evaluation window was not used to choose them.

### Checks

| Check | Rule | Severity |
|---|---|---|
| Pandera `positive_price` | Every series except the yields is strictly positive | critical |
| Pandera `yield_range` | DGS2, DGS5, DGS10 within [−1%, 20%] | critical |
| Pandera `max_move` | \|1-day move\| within a per-series limit: 0.35 (log) for equities, 0.06 for FX, 60bp for yields, 1.6 (log) for VIX, about twice the largest 2015–2021 move of each asset class | critical |
| Pandera `missing_print` | A null on a day when most of the series' calendar peers print. Calendars: equities with VIX, yfinance FX, Treasuries, FRED FX. A holiday nulls the whole group, so it is not flagged | critical |
| Pandera index | Dates sorted and unique. A failure is structural and stops the run | — |
| Staleness | 3 or more consecutive identical prints, on liquid series only (equities, yfinance FX, VIX). Treasury yields are quoted to 1bp and often sit unchanged for days, so they are not checked | critical |
| Cross-source | yfinance EUR/USD and USD/INR vs FRED DEXUSEU and DEXINUS, beyond 200bp (just above the largest 2015–2021 gap). The pairs share a quote convention: DEXUSEU is USD per EUR like EURUSD=X, and DEXINUS is INR per USD like INR=X. yfinance's close dated D matches FRED's noon New York print of D−1 more closely than the same-day print, so D is compared with FRED's previous print | critical |
| Isolation Forest | Anomaly score above the 99.9th percentile of the training scores | warning |

Moves are measured between observed prints, as the engine measures shocks: log returns, and bp for yields. A move after a gap spans the gap.

**Isolation Forest.** One scikit-learn model (200 trees, seeded) pooled across the ten risk factors. Features per factor and date, all computable on the run date:

- return z-score: the move divided by the volatility of the previous 60 moves (today excluded)
- reversal: −(move × previous move) / trailing variance, positive when the day undoes the previous day's move. This is the causal form of SPEC's "next-day reversal": the day after a bad print scores high, and that day belongs to the corruption's footprint (below).
- vol ratio: volatility of the last 5 moves over the trailing volatility
- cross-source gap: |gap| / tolerance for EUR/USD and USD/INR, 0 for other factors

The model is fit on data up to 2021-12-31. The panel is cut at that date before features are computed, so no later print can reach the fit; a test checks that corrupting every later print leaves the fitted model unchanged. The score cutoff was fixed before the evaluation and not tuned on 2022+. Isolation Forest findings are warnings: a statistical flag goes to review and does not change the run.

### Effect on the daily run

`run-daily` runs the controls for its date first and writes the findings to `dq_findings` and to `controls.findings` in run.json before the engine starts, so a run that then fails on bad data still leaves its evidence. A risk factor with a critical finding on the run date is **held flat** in that day's revaluation (ADR-008). Its shocks are zero in the historical-simulation window, in the Monte Carlo draws (the Cholesky factor is taken over the remaining factors), in the historical stress replays, and in the VaR-explain legs. Its level, EWMA vol, and sensitivities are not altered, and hypothetical stress scenarios (specified shocks, not data) still apply. The engine never fills or replaces a print. Held factors are listed in `controls.excluded_factors`.

### Evaluation protocol (`eval/dq.py`, `make eval-dq`)

- **Held-out copy:** the panel through 2025-12-31. Corruptions and scoring are confined to 2022-01-01 to 2025-12-31; earlier data only serves as look-back history for the checks.
- **Corruptions:** 200, 40 per type, at seeded random dates on the ten risk factors. A factor's corruptions are at least 10 business days apart, and each must change the data.
  - stale run: the print and the next two repeat the previous print
  - ×10 spike: the day's move is multiplied by 10 (log return; bp for yields), and the next day returns to the true path
  - sign flip: the level is negated
  - missing value: the print is removed
  - decimal shift: the level is multiplied by 10 or 0.1
- **Footprint:** the factor-dates whose level or 1-day move differs from the clean data. That is the corrupted prints plus the next print, whose move starts from a corrupted value; the engine would take a bad shock on that day too.
- **Metrics:** a flag is any finding on a risk-factor date. FRED second-source series are not scored. Recall is the share of corruptions with at least one flagged footprint date. Precision is the share of flagged factor-dates inside a footprint, so false alarms on genuine 2022–2025 data count against it. F1 is their harmonic mean.
- **Per type vs overall:** per-type rows inject that type's 40 corruptions alone, at the same positions, so every false alarm belongs to one type. The overall row injects all 200 together.
- **Variants:** rules only (Pandera, staleness, cross-source) and rules + Isolation Forest, scored from the same run of the controls.

Results are in RESULTS.md, "Market data controls".

## Model choices

TBD (volatility models in phase 05b).

## Evaluation protocol

TBD (phase 02).

## Assumptions and limitations

From SPEC Appendix B, as they apply to the risk engine:

- The trading book is synthetic and constant-maturity (extended to constant moneyness and USD notional, ADR-004). Backtest P&L therefore excludes time decay and position changes.
- Valuation is single-curve USD off Treasury CMT yields treated as zero rates: no OIS/SOFR curve, no bootstrap, no basis. Swaps are valued on a reset date.
- There is no implied-vol surface: VIX proxies SPY implied vol (flat across strikes and expiries), and single stocks use EWMA forecast vol. Single-stock vol risk is not a VaR factor.
- The EUR and INR legs of FX forwards discount at constant proxy rates, so foreign-rate risk is not captured.
- Equity prices are unadjusted closes and options assume no dividends (ADR-003).
- Monte Carlo and delta-normal VaR assume normal shocks, so they understate fat tails compared with historical simulation.
- Market data gaps are handled by dropping incomplete dates.
- A risk factor held flat after a critical data finding contributes no VaR that day, so VaR is understated until the data is corrected. The run records this rather than repairing the data.
- Stale Treasury prints are not checked for staleness (1bp quoting), and a ×10 spike on a quiet day stays within normal ranges. The rules largely miss both by design.
- The cross-source check compares with FRED's previous print. FRED's H.10 FX series are published with a delay, so in live operation the check would use the latest available print.
