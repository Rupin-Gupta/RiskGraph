"""FX forwards by covered interest parity (SPEC §4.1). A spot position is a forward due now.

Pair convention: rate = units of quote currency per unit of base (EURUSD: USD per EUR;
USDINR: INR per USD).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _f(x: npt.ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def forward_rate(
    spot: npt.ArrayLike, df_base: npt.ArrayLike, df_quote: npt.ArrayLike
) -> FloatArray:
    """Fair forward rate in quote units per base unit. df_*: discount factors to delivery."""
    return _f(spot) * _f(df_base) / _f(df_quote)


def forward_pv(
    notional: npt.ArrayLike,
    contract_rate: npt.ArrayLike,
    spot: npt.ArrayLike,
    df_base: npt.ArrayLike,
    df_quote: npt.ArrayLike,
) -> FloatArray:
    """PV in quote currency of buying `notional` base units at `contract_rate` (sell if < 0)."""
    fwd = forward_rate(spot, df_base, df_quote)
    return _f(notional) * (fwd - _f(contract_rate)) * _f(df_quote)


def foreign_df(rate: npt.ArrayLike, t: npt.ArrayLike) -> FloatArray:
    """Discount factor (1 + rate/2)^(-2t) for a constant foreign rate (decimal) and t in years."""
    return (1 + _f(rate) / 2) ** (-2 * _f(t))
