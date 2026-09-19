"""Fixed-float interest rate swaps and fixed-coupon bonds off the USD curve (SPEC §4.1).

Both are linear in discount factors, so each is a list of cash flows (times in years,
amounts in USD). The portfolio stacks them into one matrix for vectorized revaluation.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from riskgraph.pricing.curve import discount_factors

FloatArray = npt.NDArray[np.float64]
Cashflows = tuple[FloatArray, FloatArray]

FREQ = 2  # semi-annual fixed leg and coupons


def coupon_times(t: float) -> FloatArray:
    """Payment times in years, every 6 months counted back from maturity t (years).

    A first stub shorter than about 4 days merges into the next period.
    """
    n = max(1, math.ceil(FREQ * t - 0.02))
    return t - np.arange(n - 1, -1, -1, dtype=np.float64) / FREQ


def swap_cashflows(notional: float, fixed_rate: float, t: float, payer: bool) -> Cashflows:
    """Cash flows of a fixed-float swap. Payer pays fixed and receives float.

    notional: USD (> 0); fixed_rate: decimal; t: years to maturity. Single-curve float leg
    valued on a reset date: +notional now and -notional at maturity.
    """
    times = coupon_times(t)
    fixed = np.full(len(times), -notional * fixed_rate / FREQ)
    fixed[-1] -= notional
    sign = 1.0 if payer else -1.0
    return np.r_[0.0, times], sign * np.r_[notional, fixed]


def bond_cashflows(face: float, coupon: float, t: float) -> Cashflows:
    """Cash flows of a fixed-coupon bond. face: USD; coupon: decimal per year; t: years."""
    times = coupon_times(t)
    amounts = np.full(len(times), face * coupon / FREQ)
    amounts[-1] += face
    return times, amounts


def pv(times: FloatArray, amounts: FloatArray, pillar_rates: FloatArray) -> FloatArray:
    """PV in USD per scenario, shape (n_scenarios,). pillar_rates: (n_scenarios, 3), decimal."""
    return discount_factors(pillar_rates, times) @ amounts


def dv01(
    times: FloatArray, amounts: FloatArray, pillar_rates: FloatArray, bump_bp: float = 1.0
) -> FloatArray:
    """USD change in PV for a parallel +bump_bp shift of the curve (negative for a long bond)."""
    return pv(times, amounts, pillar_rates + bump_bp / 1e4) - pv(times, amounts, pillar_rates)


def par_swap_rate(t: float, pillar_rates: FloatArray) -> FloatArray:
    """Fixed rate (decimal) that gives a t-year swap zero PV, shape (n_scenarios,)."""
    df = discount_factors(pillar_rates, coupon_times(t))
    annuity: FloatArray = df.sum(axis=1) / FREQ
    return (1 - df[:, -1]) / annuity
