"""Risk-factor shocks, gap policy, EWMA covariance, and daily market states (SPEC §4.2-§4.3).

Gap policy: a date is used only if every risk factor is observed. Incomplete dates are
dropped, so the next shock spans the gap. Dropped dates are reported. Missed prints are also
flagged by the market data controls (marketdata/controls.py).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from riskgraph.book.schema import Trade
from riskgraph.pricing.market import CURVE, EQUITIES, FACTORS, IDX, MarketState
from riskgraph.pricing.portfolio import reference_factor

FloatArray = npt.NDArray[np.float64]
TRADING_DAYS = 252  # annualization of daily EWMA vols


def ewma_cov(shocks: FloatArray, lam: float) -> FloatArray:
    """Zero-mean EWMA covariance of daily shocks, shape (k, k), in the shocks' units squared.

    shocks: (n_days, k), oldest row first. Weights are proportional to lam**age and sum to one.
    """
    w = lam ** np.arange(len(shocks) - 1, -1, -1, dtype=np.float64)
    w /= w.sum()
    cov: FloatArray = (shocks * w[:, None]).T @ shocks
    return cov


@dataclass(frozen=True)
class RiskContext:
    """Cleaned market history plus risk parameters; builds states and scenario windows.

    levels: complete dates only, panel units (yields in percent). shocks: row d is the move
    from the previous complete date to d (log returns; bp for yields). dropped: panel dates
    removed by the gap policy. excluded: factors with a critical data-quality finding on the
    run date, held flat (zero shock) in every scenario; their levels are not altered.
    """

    levels: pd.DataFrame
    shocks: pd.DataFrame
    dropped: pd.DatetimeIndex
    window: int
    ewma_lambda: float
    foreign_rates: dict[str, float]
    excluded: tuple[str, ...] = ()

    @classmethod
    def from_panel(cls, panel: pd.DataFrame, risk_cfg: Mapping[str, Any]) -> RiskContext:
        """Apply the gap policy to a market panel and compute factor shocks."""
        p = panel[list(FACTORS)].sort_index()
        complete = p.notna().all(axis=1)
        levels = p[complete]
        shocks = pd.DataFrame(
            {f: levels[f].diff() * 100 if f in CURVE else np.log(levels[f]).diff() for f in FACTORS}
        ).iloc[1:]
        return cls(
            levels=levels,
            shocks=shocks,
            dropped=pd.DatetimeIndex(p.index[~complete]),
            window=int(risk_cfg["var"]["window_days"]),
            ewma_lambda=float(risk_cfg["ewma_lambda"]),
            foreign_rates=dict(risk_cfg["fx_forward"]["foreign_rate_proxy"]),
        )

    def observed_window(self, day: pd.Timestamp) -> FloatArray:
        """The `window` most recent daily shocks up to and including `day`, oldest first."""
        s = self.shocks.loc[:day]
        if len(s) < self.window:
            raise ValueError(f"{day:%Y-%m-%d}: {len(s)} shocks available, {self.window} needed")
        out: FloatArray = s.to_numpy(dtype=np.float64, copy=True)[-self.window :]
        return out

    def window_shocks(self, day: pd.Timestamp) -> FloatArray:
        """Scenario shocks: the observed window with excluded factors held flat."""
        out = self.observed_window(day)
        out[:, [IDX[f] for f in self.excluded]] = 0.0
        return out

    def window_start(self, day: pd.Timestamp) -> pd.Timestamp:
        """First date of the scenario window ending on `day`."""
        return pd.Timestamp(self.shocks.loc[:day].index[-self.window])

    def dropped_in_window(self, day: pd.Timestamp) -> list[str]:
        """Dates removed by the gap policy inside the scenario window ending on `day`."""
        d = self.dropped
        return [f"{x:%Y-%m-%d}" for x in d[(d >= self.window_start(day)) & (d <= day)]]

    def prev(self, day: pd.Timestamp) -> pd.Timestamp:
        """Previous complete date before `day` (a complete date)."""
        i = self.levels.index.get_loc(day)
        if not isinstance(i, int) or i == 0:
            raise ValueError(f"{day:%Y-%m-%d}: no previous complete date")
        return pd.Timestamp(self.levels.index[i - 1])

    def state(self, day: pd.Timestamp) -> MarketState:
        """Market state on a complete date, with EWMA vols for single-stock options.

        Vols come from observed shocks, so holding a factor flat does not change its vol.
        """
        if day not in self.levels.index:
            raise ValueError(f"{day:%Y-%m-%d}: no complete market data (gap policy)")
        row = self.levels.loc[day].to_numpy(dtype=np.float64, copy=True)
        row[[IDX[f] for f in CURVE]] /= 100
        cov = ewma_cov(self.observed_window(day), self.ewma_lambda)
        vols = {u: float(np.sqrt(TRADING_DAYS * cov[IDX[u], IDX[u]])) for u in EQUITIES[1:]}
        return MarketState(day.date(), row[None, :], vols, self.foreign_rates)

    def ref_spots(self, trades: Sequence[Trade]) -> dict[str, float]:
        """Trade-date level of each trade's reference factor (last complete date on or before)."""
        out = {}
        for tr in trades:
            f = reference_factor(tr)
            if f is not None:
                level = self.levels[f].asof(pd.Timestamp(tr.trade_date))
                if pd.isna(level):
                    raise ValueError(f"{tr.trade_id}: no market data on or before trade date")
                out[tr.trade_id] = float(level)
        return out
