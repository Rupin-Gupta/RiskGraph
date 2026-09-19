"""VaR backtesting: exceptions, Kupiec POF, Christoffersen independence, Basel zones (SPEC §4.5)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

from riskgraph.book.schema import Trade
from riskgraph.risk.daily import Configs, hold, hs_var, mc_var
from riskgraph.risk.factors import RiskContext
from riskgraph.risk.stress import run_stress, worst_loss
from riskgraph.risk.var import SCOPES, scenario_pnl

FloatArray = npt.NDArray[np.float64]


def _lr_test(ll_null: float, ll_alt: float) -> tuple[float, float]:
    lr = max(0.0, -2.0 * (ll_null - ll_alt))  # clip float noise below zero
    return lr, float(chi2.sf(lr, 1))


def _ll(*pairs: tuple[float, float]) -> float:
    """Sum of count * ln(prob), with 0 * ln(0) = 0."""
    return float(sum(xlogy(n, p) for n, p in pairs))


def kupiec_pof(exceptions: int, days: int, p: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures LR statistic and chi2(1) p-value.

    H0: the exception rate equals p (0.01 for 99% VaR).
    """
    x, n = exceptions, days
    phat = x / n
    return _lr_test(_ll((n - x, 1 - p), (x, p)), _ll((n - x, 1 - phat), (x, phat)))


def christoffersen_independence(hits: Sequence[int] | npt.ArrayLike) -> tuple[float, float]:
    """Christoffersen independence LR statistic and chi2(1) p-value.

    H0: an exception today is as likely after an exception yesterday as after none.
    """
    h = np.asarray(hits, dtype=bool)
    a, b = h[:-1], h[1:]
    n00, n01 = int(np.sum(~a & ~b)), int(np.sum(~a & b))
    n10, n11 = int(np.sum(a & ~b)), int(np.sum(a & b))
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / len(a)
    null = _ll((n00 + n10, 1 - pi), (n01 + n11, pi))
    alt = _ll((n00, 1 - pi0), (n01, pi0), (n10, 1 - pi1), (n11, pi1))
    return _lr_test(null, alt)


def traffic_light(exceptions: int, zones: Mapping[str, Any]) -> str:
    """Basel zone for an exception count in 250 days: green, yellow, or red."""
    if exceptions >= zones["red_from"]:
        return "red"
    return "yellow" if exceptions >= zones["yellow"][0] else "green"


def summarize(
    pnl: FloatArray, var: FloatArray, p: float, zones: Mapping[str, Any]
) -> dict[str, Any]:
    """Backtest P&L (USD, day t) against VaR (USD loss, set on day t-1). Exception: loss > VaR."""
    hits = -pnl > var
    x, n = int(hits.sum()), len(hits)
    k_lr, k_p = kupiec_pof(x, n, p)
    c_lr, c_p = christoffersen_independence(hits)
    return {
        "days": n,
        "exceptions": x,
        "expected": round(n * p, 2),
        "kupiec_lr": round(k_lr, 4),
        "kupiec_p_value": round(k_p, 6),
        "christoffersen_lr": round(c_lr, 4),
        "christoffersen_p_value": round(c_p, 6),
        "traffic_light": traffic_light(x, zones),
    }


def history(
    ctx: RiskContext,
    book: Sequence[Trade],
    start: str,
    end: str,
    cfg: Configs,
    seed: int,
) -> pd.DataFrame:
    """Daily VaR, hypothetical P&L, and stress loss per scope for each complete date t.

    Row t (index "date"): VaR (HS and MC) and worst stress loss as of the prior complete date
    d ("asof"), and the hypothetical P&L from d to t: d's positions and market state with
    t's realized factor moves applied (static positions, no time decay). USD.
    """
    rc = cfg.risk
    conf, es_conf = float(rc["var"]["confidence"]), float(rc["var"]["es_confidence"])
    ref = ctx.ref_spots(book)
    idx = ctx.levels.index
    rows = []
    for t in idx[(idx >= start) & (idx <= end)]:
        d = ctx.prev(t)
        h = hold(ctx, book, ref, d)
        hs, mc = hs_var(h, conf, es_conf), mc_var(h, rc, seed, conf, es_conf)
        pnl = (scenario_pnl(h.pos, h.state, ctx.shocks.loc[[t]].to_numpy()) @ h.scope_m)[0]
        stress = run_stress(h.pos, h.state, h.scope_m, ctx, cfg.scenarios)
        row: dict[str, Any] = {"date": t, "asof": d}
        for j, s in enumerate(SCOPES):
            row |= {f"pnl_{s}": pnl[j], f"var_hs_{s}": hs[j][0], f"var_mc_{s}": mc[j][0]}
            row[f"stress_{s}"] = worst_loss(stress, s)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def report(hist: pd.DataFrame, cfg: Configs) -> dict[str, Any]:
    """Backtest statistics per method and scope: the last `window_days` days and the full period."""
    bt = cfg.risk["backtest"]
    p = 1 - float(cfg.risk["var"]["confidence"])

    def stats(h: pd.DataFrame, zones: bool) -> dict[str, Any]:
        out: dict[str, Any] = {
            "start": f"{h.index[0]:%Y-%m-%d}",
            "end": f"{h.index[-1]:%Y-%m-%d}",
            "days": len(h),
        }
        for method in ("hs", "mc"):
            name = "historical" if method == "hs" else "monte_carlo"
            out[name] = {}
            for s in SCOPES:
                r = summarize(
                    h[f"pnl_{s}"].to_numpy(),
                    h[f"var_{method}_{s}"].to_numpy(),
                    p,
                    bt["traffic_light"],
                )
                del r["days"]
                if not zones:
                    del r["traffic_light"]
                out[name][s] = r
        return out

    return {
        "window": stats(hist.iloc[-int(bt["window_days"]) :], zones=True),
        "full_period": stats(hist, zones=False),
    }
