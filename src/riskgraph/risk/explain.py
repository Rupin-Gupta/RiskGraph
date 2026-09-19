"""VaR explain and attribution (SPEC §4.7)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskgraph.pricing.market import FACTORS, MarketState
from riskgraph.pricing.portfolio import Positions
from riskgraph.risk.var import scenario_pnl, tail_indices

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]


def decompose(prev: float, pos_only: float, mkt_only: float, curr: float) -> dict[str, float]:
    """Split the VaR change (USD) into position, market, and interaction effects.

    prev = VaR(positions t-1, market t-1); pos_only = VaR(positions t, market t-1);
    mkt_only = VaR(positions t-1, market t); curr = VaR(positions t, market t).
    The interaction is the residual, so the three effects sum to the total by construction.
    """
    total, position, market = curr - prev, pos_only - prev, mkt_only - prev
    return {
        "total": total,
        "position": position,
        "market": market,
        "interaction": total - position - market,
    }


def tail_scenarios(
    pnl_trades: FloatArray, weights: FloatArray, var_conf: float, n: int
) -> IntArray:
    """Scenarios ranked within n of the VaR scenario of the P&L weighted by 0/1 per trade."""
    return tail_indices(pnl_trades @ weights, var_conf, n)


def trade_contributions(pnl_trades: FloatArray, weights: FloatArray, idx: IntArray) -> FloatArray:
    """Euler-style contribution (USD loss) of each trade: its mean loss over tail scenarios idx.

    Contributions sum to the mean loss of the scope over those scenarios, which approximates VaR.
    """
    out: FloatArray = -(pnl_trades[idx] * weights).mean(axis=0)
    return out


def factor_contributions(
    pos: Positions, state: MarketState, shocks: FloatArray, weights: FloatArray, idx: IntArray
) -> dict[str, float]:
    """Mean loss (USD) over tail scenarios idx from each factor's move on its own.

    Each factor's shock is replayed alone with full revaluation; "cross_effects" is what the
    one-at-a-time losses leave unexplained (non-linearity across factors).
    """
    tail = shocks[idx]
    m, k = tail.shape
    alone = np.zeros((m, k, k))
    alone[:, np.arange(k), np.arange(k)] = tail
    loss_alone = -(scenario_pnl(pos, state, alone.reshape(m * k, k)) @ weights).reshape(m, k)
    loss_total = -(scenario_pnl(pos, state, tail) @ weights)
    out = dict(zip(FACTORS, loss_alone.mean(axis=0).tolist(), strict=True))
    out["cross_effects"] = float(loss_total.mean() - loss_alone.sum(axis=1).mean())
    return out
