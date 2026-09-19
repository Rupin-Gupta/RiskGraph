"""Limit utilization and status (SPEC §4.6). Limits and thresholds come from configs/limits.yaml."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

METRICS = ("var_99_1d", "stress_loss")  # counterparty PFE limits arrive in phase 06


def status(utilization: float, thresholds: Mapping[str, float]) -> str:
    """ok < warning threshold <= warning < breach threshold <= breach."""
    if utilization >= thresholds["breach"]:
        return "breach"
    return "warning" if utilization >= thresholds["warning"] else "ok"


def evaluate(
    values: Mapping[tuple[str, str], float], cfg: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """One row per configured (non-null) limit. values: (scope, metric) -> USD value."""
    rows = []
    for metric in METRICS:
        limits = {"firm": cfg[metric].get("firm")} | (cfg[metric].get("desks") or {})
        for scope, limit in limits.items():
            if limit is None:
                continue
            value = float(values[(scope, metric)])
            u = value / float(limit)
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "value": value,
                    "limit": float(limit),
                    "utilization": u,
                    "status": status(u, cfg["status_thresholds"]),
                }
            )
    return rows


def calibrate(history: npt.ArrayLike, quantile: float) -> float:
    """Limit = `quantile` of the metric's history (USD), rounded to 3 significant figures.

    Coarser rounding can move a limit past the whole history of a slow-moving VaR series,
    which would leave no breaches to calibrate against.
    """
    return float(f"{float(np.quantile(np.asarray(history, dtype=np.float64), quantile)):.3g}")


def history_report(hist: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Calibrated limits and configured-limit status counts over the calibration period.

    hist: backtest history (risk.backtest.history); metric values are taken as of each
    run date ("asof"). The calibrated limit is the `quantile` of each history (USD).
    """
    cal = cfg["calibration"]
    h = hist.set_index("asof").loc[str(cal["start"]) : str(cal["end"])]
    out: dict[str, Any] = {
        "period": [f"{h.index[0]:%Y-%m-%d}", f"{h.index[-1]:%Y-%m-%d}"],
        "days": len(h),
        "quantile": cal["quantile"],
    }
    for metric, column in (("var_99_1d", "var_hs_"), ("stress_loss", "stress_")):
        limits = {"firm": cfg[metric].get("firm")} | (cfg[metric].get("desks") or {})
        out[metric] = {}
        for scope, limit in limits.items():
            series = h[column + scope]
            entry: dict[str, Any] = {"calibrated": calibrate(series, float(cal["quantile"]))}
            if limit is not None:
                u = series / float(limit)
                breach = u >= cfg["status_thresholds"]["breach"]
                days = [f"{d:%Y-%m-%d}" for d in series.index[breach]]
                entry |= {
                    "limit": float(limit),
                    "breach_days": int(breach.sum()),
                    "warning_days": int(
                        ((u >= cfg["status_thresholds"]["warning"]) & ~breach).sum()
                    ),
                    "first_breach": days[0] if days else None,
                    "last_breach": days[-1] if days else None,
                }
            out[metric][scope] = entry
    return out
