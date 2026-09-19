"""Black-Scholes European options on a non-dividend-paying underlying (SPEC §4.1).

All inputs broadcast, so one call prices every trade under every scenario.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

FloatArray = npt.NDArray[np.float64]


def _f(x: npt.ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def _cdf(x: FloatArray) -> FloatArray:
    return np.asarray(ndtr(x), dtype=np.float64)


def _pdf(x: FloatArray) -> FloatArray:
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _d1_d2(
    s: FloatArray, k: FloatArray, t: FloatArray, r: FloatArray, v: FloatArray
) -> tuple[FloatArray, FloatArray]:
    vt = v * np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * v * v) * t) / vt
    return d1, d1 - vt


def bs_price(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    t: npt.ArrayLike,
    rate: npt.ArrayLike,
    vol: npt.ArrayLike,
    is_call: npt.ArrayLike,
) -> FloatArray:
    """Option value per unit of underlying, in the underlying's price units.

    spot, strike: price units; t: years to expiry; rate: continuously compounded, decimal;
    vol: annualized, decimal (0.20 = 20%); is_call: True for calls, False for puts.
    """
    s, k, tt, r, v = _f(spot), _f(strike), _f(t), _f(rate), _f(vol)
    d1, d2 = _d1_d2(s, k, tt, r, v)
    df = np.exp(-r * tt)
    call = s * _cdf(d1) - k * df * _cdf(d2)
    put = k * df * _cdf(-d2) - s * _cdf(-d1)
    return np.where(np.asarray(is_call, dtype=bool), call, put)


def bs_greeks(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    t: npt.ArrayLike,
    rate: npt.ArrayLike,
    vol: npt.ArrayLike,
    is_call: npt.ArrayLike,
) -> dict[str, FloatArray]:
    """Analytic Greeks per unit of underlying. Inputs as in `bs_price`.

    delta: dV/dS (dimensionless); gamma: d2V/dS2 (per price unit); vega: dV/dvol per 1.00 of
    vol (divide by 100 for one vol point); theta: dV/dt in calendar time, per year (divide by
    365 for one day).
    """
    s, k, tt, r, v = _f(spot), _f(strike), _f(t), _f(rate), _f(vol)
    call = np.asarray(is_call, dtype=bool)
    d1, d2 = _d1_d2(s, k, tt, r, v)
    df = np.exp(-r * tt)
    decay = -s * _pdf(d1) * v / (2 * np.sqrt(tt))
    return {
        "delta": np.where(call, _cdf(d1), _cdf(d1) - 1),
        "gamma": _pdf(d1) / (s * v * np.sqrt(tt)),
        "vega": s * _pdf(d1) * np.sqrt(tt),
        "theta": np.where(call, decay - r * k * df * _cdf(d2), decay + r * k * df * _cdf(-d2)),
    }
