"""Market state for one valuation date, one row per scenario (SPEC §4.1, §4.2)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

EQUITIES = ("SPY", "AAPL", "MSFT", "JPM")
FX_FACTORS = ("EURUSD=X", "INR=X")
CURVE = ("DGS2", "DGS5", "DGS10")
VIX = "VIXCLS"
FACTORS = (*EQUITIES, *FX_FACTORS, *CURVE, VIX)
IDX = {f: i for i, f in enumerate(FACTORS)}
CURVE_IDX = [IDX[f] for f in CURVE]
# Curve points are shocked additively in bp; every other factor by a log return.
LOG_FACTOR = np.array([f not in CURVE for f in FACTORS])


@dataclass(frozen=True)
class MarketState:
    """Market levels on `date`.

    levels: shape (n_scenarios, len(FACTORS)) in FACTORS order. Equities in USD, EURUSD=X in
    USD per EUR, INR=X in INR per USD, Treasury yields in decimal (0.045 = 4.5%), VIXCLS in
    index points (SPY option vol = VIX / 100).
    equity_vol: annualized decimal vol for single-stock options, a model parameter held fixed
    under scenario shocks.
    foreign_rates: constant decimal rates (semi-annual) for the non-USD FX legs, e.g. EUR, INR.
    """

    date: date
    levels: FloatArray
    equity_vol: dict[str, float]
    foreign_rates: dict[str, float]

    def __getitem__(self, factor: str) -> FloatArray:
        """Level of one factor per scenario, shape (n_scenarios,)."""
        return self.levels[:, IDX[factor]]

    @property
    def curve(self) -> FloatArray:
        """Decimal yields at the 2Y/5Y/10Y pillars, shape (n_scenarios, 3)."""
        return self.levels[:, CURVE_IDX]


def apply_shocks(state: MarketState, shocks: FloatArray) -> MarketState:
    """Shift a one-row state by shocks of shape (n_scenarios, len(FACTORS)).

    Shock units (SPEC §4.2): log returns for equities, FX, and VIX; bp for curve points.
    """
    if len(state.levels) != 1:
        raise ValueError("apply_shocks needs a single-scenario base state")
    base = state.levels
    levels = np.where(LOG_FACTOR, base * np.exp(shocks), base + shocks / 1e4)
    return replace(state, levels=levels)
