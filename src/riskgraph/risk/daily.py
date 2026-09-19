"""Daily risk run for one date (SPEC §4.8): VaR/ES, sensitivities, stress, limits, VaR explain.

Returns plain dicts; the CLI writes them to Postgres and data/runs/<run_id>/run.json.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from riskgraph.book.schema import Trade
from riskgraph.pricing.market import CURVE_IDX, FACTORS, LOG_FACTOR, MarketState
from riskgraph.pricing.portfolio import Positions, option_greeks, revalue, roll
from riskgraph.risk.explain import (
    decompose,
    factor_contributions,
    tail_scenarios,
    trade_contributions,
)
from riskgraph.risk.factors import RiskContext, ewma_cov
from riskgraph.risk.limits import evaluate
from riskgraph.risk.stress import run_stress, worst_loss
from riskgraph.risk.var import (
    SCOPES,
    mc_shocks,
    parametric_var,
    scenario_pnl,
    scope_matrix,
    sensitivities,
    var_es,
)

FloatArray = npt.NDArray[np.float64]
TOP_TRADES = 10
GAP_POLICY = "dates with any missing risk factor are dropped; the next shock spans the gap"


@dataclass(frozen=True)
class Configs:
    """Parsed configs/risk.yaml, configs/limits.yaml, and configs/scenarios.yaml."""

    risk: Mapping[str, Any]
    limits: Mapping[str, Any]
    scenarios: Mapping[str, Any]


@dataclass(frozen=True)
class Held:
    """A book held on one date: positions, market state, HS window, and HS scenario P&L (USD)."""

    pos: Positions
    state: MarketState
    shocks: FloatArray
    pnl: FloatArray
    scope_m: FloatArray


def hold(
    ctx: RiskContext, trades: Sequence[Trade], ref: Mapping[str, float], day: pd.Timestamp
) -> Held:
    """Roll `trades` to `day` and revalue them under the historical scenarios ending on `day`."""
    state = ctx.state(day)
    pos = roll(trades, state, ref)
    shocks = ctx.window_shocks(day)
    return Held(pos, state, shocks, scenario_pnl(pos, state, shocks), scope_matrix(pos.desks))


def hs_var(h: Held, var_conf: float, es_conf: float) -> list[tuple[float, float]]:
    """Historical-simulation (VaR, ES) in USD per scope, in SCOPES order."""
    scope_pnl = h.pnl @ h.scope_m
    return [var_es(scope_pnl[:, j], var_conf, es_conf) for j in range(len(SCOPES))]


def mc_var(
    h: Held, cfg: Mapping[str, Any], seed: int, var_conf: float, es_conf: float
) -> list[tuple[float, float]]:
    """Monte Carlo (VaR, ES) in USD per scope. EWMA covariance, Cholesky, full revaluation.

    The generator is seeded with (seed, date), so a date's MC VaR is reproducible anywhere.
    """
    cov = ewma_cov(h.shocks, float(cfg["ewma_lambda"]))
    rng = np.random.default_rng([seed, h.state.date.toordinal()])
    shocks = mc_shocks(cov, int(cfg["monte_carlo"]["n_scenarios"]), rng)
    scope_pnl = scenario_pnl(h.pos, h.state, shocks) @ h.scope_m
    return [var_es(scope_pnl[:, j], var_conf, es_conf) for j in range(len(SCOPES))]


def run(
    ctx: RiskContext,
    book: Sequence[Trade],
    base_book: Sequence[Trade],
    day: pd.Timestamp,
    cfg: Configs,
    seed: int,
) -> dict[str, Any]:
    """Risk run for `day`. `base_book` is the prior day's positions for VaR explain.

    All money amounts are USD. VaR/ES are 1-day, positive losses. Metric units: sens_<factor>
    per +1% move (equities, FX, VIX) or per +1bp (curve points); dv01 per +1bp parallel.
    """
    day = pd.Timestamp(day)
    prev = ctx.prev(day)
    rc = cfg.risk
    conf, es_conf = float(rc["var"]["confidence"]), float(rc["var"]["es_confidence"])
    ref, base_ref = ctx.ref_spots(book), ctx.ref_spots(base_book)

    now = hold(ctx, book, ref, day)
    pos, state, s_m = now.pos, now.state, now.scope_m
    hs = hs_var(now, conf, es_conf)
    mc = mc_var(now, rc, seed, conf, es_conf)
    cov = ewma_cov(now.shocks, ctx.ewma_lambda)
    sens = sensitivities(pos, state) @ s_m  # (factors, scopes)
    param = parametric_var(sens, cov, conf)
    stress = run_stress(pos, state, s_m, ctx, cfg.scenarios)
    pv = revalue(pos, state)[0]
    greeks = option_greeks(pos, state)
    greek_scope = {}
    for name, g in greeks.items():
        full = np.zeros(len(pv))
        full[pos.opt_col] = g
        greek_scope[name] = full @ s_m

    # VaR explain: positions t vs t-1 crossed with markets t vs t-1 (historical simulation).
    var_pp = hs_var(hold(ctx, base_book, base_ref, prev), conf, es_conf)
    var_cp = hs_var(hold(ctx, book, ref, prev), conf, es_conf)
    var_pc = hs_var(hold(ctx, base_book, base_ref, day), conf, es_conf)

    metrics: dict[str, dict[str, float]] = {}
    explain: dict[str, dict[str, Any]] = {}
    n_tail = int(rc["explain"]["tail_neighbors"])
    for j, scope in enumerate(SCOPES):
        effects = decompose(var_pp[j][0], var_cp[j][0], var_pc[j][0], hs[j][0])
        w = s_m[:, j]
        idx = tail_scenarios(now.pnl, w, conf, n_tail)
        contrib = trade_contributions(now.pnl, w, idx)
        rows = zip(pos.trade_ids, contrib, w, strict=True)
        in_scope = [(t, float(c)) for t, c, wi in rows if wi]
        factors = factor_contributions(pos, state, now.shocks, w, idx)
        explain[scope] = effects | {
            "top_trades": [
                {"trade_id": t, "contribution": c}
                for t, c in sorted(in_scope, key=lambda x: -x[1])[:TOP_TRADES]
            ],
            "top_factors": [
                {"factor": f, "contribution": c}
                for f, c in sorted(factors.items(), key=lambda x: -x[1])
            ],
        }
        metrics[scope] = {
            "pv": float(pv @ w),
            "var_99_1d_hs": hs[j][0],
            "es_975_1d_hs": hs[j][1],
            "var_99_1d_param": float(param[j]),
            "var_99_1d_mc": mc[j][0],
            "es_975_1d_mc": mc[j][1],
            "dv01": float(sens[CURVE_IDX, j].sum()),
            **{k: float(v[j]) for k, v in greek_scope.items() if k in ("vega_1pt", "theta_1d")},
            **{
                f"sens_{f}": float(sens[i, j] * (0.01 if LOG_FACTOR[i] else 1.0))
                for i, f in enumerate(FACTORS)
            },
            **{f"stress_{name}": float(r["pnl"][scope]) for name, r in stress.items()},
            "stress_worst_loss": worst_loss(stress, scope),
            **{f"var_explain_{k}": v for k, v in effects.items()},
        }

    limit_values = {(s, "var_99_1d"): metrics[s]["var_99_1d_hs"] for s in SCOPES}
    limit_values |= {(s, "stress_loss"): metrics[s]["stress_worst_loss"] for s in SCOPES}
    opt_pos = {int(c): i for i, c in enumerate(pos.opt_col)}
    trades = []
    for col, tr in enumerate(book):
        row: dict[str, Any] = {"trade_id": tr.trade_id, "desk": tr.desk}
        row |= {"instrument_type": tr.instrument_type, "pv": float(pv[col])}
        if col in opt_pos:
            row |= {k: float(v[opt_pos[col]]) for k, v in greeks.items()}
        trades.append(row)
    return {
        "date": f"{day:%Y-%m-%d}",
        "prev_date": f"{prev:%Y-%m-%d}",
        "market": {
            "levels": dict(zip(FACTORS, state.levels[0].tolist(), strict=True)),
            "equity_vol": state.equity_vol,
            "scenario_window": {
                "start": f"{ctx.window_start(day):%Y-%m-%d}",
                "end": f"{day:%Y-%m-%d}",
                "days": ctx.window,
            },
            "gap_policy": GAP_POLICY,
            "dropped_dates": ctx.dropped_in_window(day),
        },
        "metrics": metrics,
        "limits": evaluate(limit_values, cfg.limits),
        "stress": stress,
        "explain": explain,
        "trades": trades,
    }
