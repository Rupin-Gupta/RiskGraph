"""VaR and ES: historical simulation, delta-normal, and Monte Carlo (SPEC §4.3).

All money amounts are USD; VaR and ES are reported as positive losses over one day.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.special import ndtri

from riskgraph.pricing.market import FACTORS, LOG_FACTOR, MarketState, apply_shocks
from riskgraph.pricing.portfolio import Positions, revalue

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

DESKS = ("fx", "rates", "equity_derivatives")
SCOPES = (*DESKS, "firm")
FD_STEP = np.where(LOG_FACTOR, 1e-4, 0.01)  # sensitivity bumps: log return, bp


def scope_matrix(desks: Sequence[str]) -> FloatArray:
    """Map trades to scopes, shape (n_trades, len(SCOPES)); pnl @ matrix sums by desk and firm."""
    m = np.array([[d == s for s in DESKS] for d in desks], dtype=np.float64).reshape(-1, 3)
    return np.hstack([m, np.ones((len(desks), 1))])


def scenario_pnl(pos: Positions, state: MarketState, shocks: FloatArray) -> FloatArray:
    """Full-revaluation P&L in USD, shape (n_scenarios, n_trades), of shocks applied to state."""
    return revalue(pos, apply_shocks(state, shocks)) - revalue(pos, state)


def _tail_count(n: int, conf: float) -> float:
    return round(n * (1 - conf), 9)  # round away float noise: 500 * (1 - 0.99) = 5.000000000000004


def var_es(pnl: FloatArray, var_conf: float, es_conf: float) -> tuple[float, float]:
    """VaR at var_conf and ES at es_conf from scenario P&L (1-D, USD), as positive losses.

    VaR is the empirical quantile: the ceil(n * (1 - var_conf))-th worst P&L. ES averages the
    worst n * (1 - es_conf) scenarios, weighting the boundary scenario by the fractional part.
    """
    srt = np.sort(pnl)
    k = max(1, math.ceil(_tail_count(len(pnl), var_conf)))
    m = _tail_count(len(pnl), es_conf)
    whole = math.floor(m)
    tail = srt[:whole].sum() + (m - whole) * (srt[whole] if m > whole else 0.0)
    return float(-srt[k - 1]), float(-tail / m)


def tail_indices(pnl: FloatArray, var_conf: float, neighbors: int) -> IntArray:
    """Scenario indices ranked within `neighbors` of the VaR scenario, worst first."""
    order = np.argsort(pnl, kind="stable")
    k = max(1, math.ceil(_tail_count(len(pnl), var_conf))) - 1
    return order[max(0, k - neighbors) : k + neighbors + 1]


def mc_shocks(cov: FloatArray, n: int, rng: np.random.Generator) -> FloatArray:
    """n correlated normal shocks, shape (n, k), with covariance `cov` (Cholesky)."""
    return rng.standard_normal((n, len(cov))) @ np.linalg.cholesky(cov).T


def sensitivities(pos: Positions, state: MarketState) -> FloatArray:
    """USD P&L per unit shock of each factor, shape (len(FACTORS), n_trades).

    Central differences with full revaluation. Units: per 1.00 log return for equities, FX,
    and VIX (multiply by 0.01 for a 1% move); per 1bp for curve points.
    """
    bumps = np.diag(FD_STEP)
    pv = revalue(pos, apply_shocks(state, np.vstack([bumps, -bumps])))
    k = len(FACTORS)
    out: FloatArray = (pv[:k] - pv[k:]) / (2 * FD_STEP[:, None])
    return out


def parametric_var(sens: FloatArray, cov: FloatArray, conf: float) -> FloatArray:
    """Delta-normal VaR per column of `sens` (k, n_scopes): z_conf * sqrt(s' cov s), USD."""
    z = float(ndtri(conf))
    variance: FloatArray = np.einsum("ij,ik,kj->j", sens, cov, sens)
    return z * np.sqrt(variance)
