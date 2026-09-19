"""Stress scenarios from configs/scenarios.yaml (SPEC §4.4), fully revalued."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from riskgraph.pricing.market import CURVE, CURVE_IDX, EQUITIES, FACTORS, IDX, MarketState
from riskgraph.pricing.portfolio import Positions
from riskgraph.risk.factors import RiskContext
from riskgraph.risk.var import SCOPES, scenario_pnl

FloatArray = npt.NDArray[np.float64]


def hypothetical_shock(shocks: Mapping[str, float]) -> FloatArray:
    """Shock vector (len(FACTORS),) from config: relative moves as fractions, curve in bp."""
    out = np.zeros(len(FACTORS))
    for key, value in shocks.items():
        if key == "curve_parallel_bp":
            out[CURVE_IDX] += value
        else:
            cols = [IDX[f] for f in EQUITIES] if key == "equities" else [IDX[key]]
            out[cols] += np.log1p(value)
    return out


def run_stress(
    pos: Positions,
    state: MarketState,
    scope_m: FloatArray,
    ctx: RiskContext,
    cfg: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Scenario P&L (USD) per scope; historical scenarios also report the chosen date.

    worst_firm_pnl_day: the day in the window with the worst firm P&L on today's book.
    largest_curve_move_day: the day with the largest absolute 1-day move of any curve point.
    Historical days replay every factor's move on that day.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, sc in cfg["historical"].items():
        days = ctx.shocks.loc[str(sc["window"][0]) : str(sc["window"][1])]
        days = days.assign(**dict.fromkeys(ctx.excluded, 0.0))  # excluded factors held flat
        if sc["select"] == "worst_firm_pnl_day":
            firm = scenario_pnl(pos, state, days.to_numpy()) @ scope_m[:, -1]
            i = int(np.argmin(firm))
        elif sc["select"] == "largest_curve_move_day":
            i = int(np.argmax(days[list(CURVE)].abs().max(axis=1).to_numpy()))
        else:
            raise ValueError(f"{name}: unknown select rule {sc['select']}")
        pnl = scenario_pnl(pos, state, days.to_numpy()[[i]]) @ scope_m
        out[name] = {
            "date": f"{days.index[i]:%Y-%m-%d}",
            "pnl": dict(zip(SCOPES, pnl[0].tolist(), strict=True)),
        }
    for name, sc in cfg["hypothetical"].items():
        pnl = scenario_pnl(pos, state, hypothetical_shock(sc["shocks"])[None, :]) @ scope_m
        out[name] = {"pnl": dict(zip(SCOPES, pnl[0].tolist(), strict=True))}
    return out


def worst_loss(results: Mapping[str, Mapping[str, Any]], scope: str) -> float:
    """Largest loss (positive USD, floored at zero) across scenarios for one scope."""
    return max(0.0, -min(float(r["pnl"][scope]) for r in results.values()))
