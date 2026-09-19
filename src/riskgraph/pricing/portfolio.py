"""Vectorized full revaluation of the trading book (SPEC §4.1, §4.2; ADR-004).

`roll` turns trades into per-instrument arrays once per valuation date; `revalue` then prices
every trade under every scenario with array operations (no loop over scenarios or trades).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from riskgraph.book.schema import Trade
from riskgraph.pricing.black_scholes import bs_greeks, bs_price
from riskgraph.pricing.curve import discount_factors
from riskgraph.pricing.fx import foreign_df, forward_pv
from riskgraph.pricing.market import IDX, VIX, MarketState
from riskgraph.pricing.rates import bond_cashflows, swap_cashflows

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
BoolArray = npt.NDArray[np.bool_]
T_ = TypeVar("T_")

YEAR_DAYS = 365.25
PAIR_FACTOR = {"EURUSD": "EURUSD=X", "USDINR": "INR=X"}
FOREIGN_CCY = {"EURUSD": "EUR", "USDINR": "INR"}  # the non-USD leg
QUOTE_IS_USD = {"EURUSD": True, "USDINR": False}


def _req(value: T_ | None) -> T_:
    if value is None:
        raise ValueError("trade is missing a field its instrument type requires")
    return value


def tenor_years(trade: Trade) -> float:
    """Years from trade date to maturity (ACT/365.25), 0 without a maturity.

    Under the constant-maturity book (ADR-004) this is the time to maturity on every date.
    """
    if trade.maturity_date is None:
        return 0.0
    return (trade.maturity_date - trade.trade_date).days / YEAR_DAYS


def reference_factor(trade: Trade) -> str | None:
    """Factor whose trade-date level anchors the trade's strike or contract rate, if any."""
    if trade.currency_pair is not None:
        return PAIR_FACTOR[trade.currency_pair]
    if trade.instrument_type == "european_option":
        return trade.underlying
    return None


@dataclass(frozen=True)
class Positions:
    """The book as held on one valuation date, as arrays for vectorized revaluation.

    Each block stores its trades' columns in the PV matrix (`*_col`) and their factor indices
    (`*_factor`). Units: `*_units` in units of the underlying; strikes and FX rates in price
    units; `*_t` in years; `opt_vol` annualized decimal (NaN = use VIX / 100); FX notionals in
    base currency; `cf` in USD at `cf_times` (years).
    """

    trade_ids: tuple[str, ...]
    desks: tuple[str, ...]
    eq_col: IntArray
    eq_factor: IntArray
    eq_units: FloatArray
    opt_col: IntArray
    opt_factor: IntArray
    opt_units: FloatArray
    opt_strike: FloatArray
    opt_t: FloatArray
    opt_call: BoolArray
    opt_vol: FloatArray
    fx_col: IntArray
    fx_factor: IntArray
    fx_notional: FloatArray
    fx_rate: FloatArray
    fx_t: FloatArray
    fx_quote_usd: BoolArray
    fx_foreign_rate: FloatArray
    lin_col: IntArray
    cf_times: FloatArray
    cf: FloatArray


def roll(trades: Sequence[Trade], state: MarketState, ref_spot: Mapping[str, float]) -> Positions:
    """Hold `trades` on `state.date` (a one-row state) as a constant-maturity book (ADR-004).

    Tenors keep their trade-date length. Option strikes and FX contract rates keep their
    trade-date moneyness: they scale by level(state.date) / ref_spot[trade_id], where ref_spot
    is the underlying or pair level on the trade date. Equity and option positions keep their
    USD notional: units = notional / spot(state.date).
    """
    x = state.levels[0]
    eq: list[tuple[int, int, float]] = []
    opt: list[tuple[int, int, float, float, float, bool, float]] = []
    fx: list[tuple[int, int, float, float, float, bool, float]] = []
    flows: list[tuple[int, FloatArray, FloatArray]] = []
    for col, tr in enumerate(trades):
        t, kind = tenor_years(tr), tr.instrument_type
        if kind == "cash_equity":
            f = IDX[_req(tr.underlying)]
            eq.append((col, f, tr.notional / x[f]))
        elif kind == "european_option":
            u = _req(tr.underlying)
            f = IDX[u]
            vol = np.nan if u == "SPY" else state.equity_vol[u]
            strike = _req(tr.strike) * x[f] / ref_spot[tr.trade_id]
            opt.append((col, f, tr.notional / x[f], strike, t, tr.option_type == "call", vol))
        elif kind in ("fx_forward", "fx_spot"):
            pair = _req(tr.currency_pair)
            f = IDX[PAIR_FACTOR[pair]]
            rate = _req(tr.forward_rate) * x[f] / ref_spot[tr.trade_id]
            r_for = state.foreign_rates[FOREIGN_CCY[pair]]
            fx.append((col, f, tr.notional, rate, t, QUOTE_IS_USD[pair], r_for))
        elif kind == "interest_rate_swap":
            payer = tr.pay_receive == "payer"
            flows.append((col, *swap_cashflows(tr.notional, _req(tr.fixed_rate), t, payer)))
        else:
            flows.append((col, *bond_cashflows(tr.notional, _req(tr.coupon), t)))

    times = np.unique(np.concatenate([f[1] for f in flows])) if flows else np.zeros(0)
    cf = np.zeros((len(flows), len(times)))
    for i, (_, t_i, a_i) in enumerate(flows):
        np.add.at(cf[i], np.searchsorted(times, t_i), a_i)

    def ints(rows: Sequence[tuple[object, ...]], j: int) -> IntArray:
        return np.array([r[j] for r in rows], dtype=np.intp)

    def floats(rows: Sequence[tuple[object, ...]], j: int) -> FloatArray:
        return np.array([r[j] for r in rows], dtype=np.float64)

    def bools(rows: Sequence[tuple[object, ...]], j: int) -> BoolArray:
        return np.array([r[j] for r in rows], dtype=np.bool_)

    return Positions(
        trade_ids=tuple(t.trade_id for t in trades),
        desks=tuple(t.desk for t in trades),
        eq_col=ints(eq, 0),
        eq_factor=ints(eq, 1),
        eq_units=floats(eq, 2),
        opt_col=ints(opt, 0),
        opt_factor=ints(opt, 1),
        opt_units=floats(opt, 2),
        opt_strike=floats(opt, 3),
        opt_t=floats(opt, 4),
        opt_call=bools(opt, 5),
        opt_vol=floats(opt, 6),
        fx_col=ints(fx, 0),
        fx_factor=ints(fx, 1),
        fx_notional=floats(fx, 2),
        fx_rate=floats(fx, 3),
        fx_t=floats(fx, 4),
        fx_quote_usd=bools(fx, 5),
        fx_foreign_rate=floats(fx, 6),
        lin_col=np.array([f[0] for f in flows], dtype=np.intp),
        cf_times=times,
        cf=cf,
    )


def _option_inputs(pos: Positions, state: MarketState) -> tuple[FloatArray, ...]:
    x = state.levels
    rate = -np.log(discount_factors(state.curve, pos.opt_t)) / pos.opt_t  # continuous
    vol = np.where(np.isnan(pos.opt_vol), x[:, [IDX[VIX]]] / 100, pos.opt_vol)
    return x[:, pos.opt_factor], rate, vol


def revalue(pos: Positions, state: MarketState) -> FloatArray:
    """PV in USD of every trade under every scenario, shape (n_scenarios, n_trades)."""
    x = state.levels
    out = np.zeros((len(x), len(pos.trade_ids)))
    out[:, pos.eq_col] = x[:, pos.eq_factor] * pos.eq_units

    s, rate, vol = _option_inputs(pos, state)
    out[:, pos.opt_col] = pos.opt_units * bs_price(
        s, pos.opt_strike, pos.opt_t, rate, vol, pos.opt_call
    )

    spot, usd = x[:, pos.fx_factor], pos.fx_quote_usd
    df_usd = discount_factors(state.curve, pos.fx_t)
    df_for = foreign_df(pos.fx_foreign_rate, pos.fx_t)
    df_base, df_quote = np.where(usd, df_for, df_usd), np.where(usd, df_usd, df_for)
    pv_quote = forward_pv(pos.fx_notional, pos.fx_rate, spot, df_base, df_quote)
    out[:, pos.fx_col] = np.where(usd, pv_quote, pv_quote / spot)

    out[:, pos.lin_col] = discount_factors(state.curve, pos.cf_times) @ pos.cf.T
    return out


def option_greeks(pos: Positions, state: MarketState) -> dict[str, FloatArray]:
    """Analytic position Greeks in USD for each option (order of `pos.opt_col`), first row.

    delta_1pct: P&L for a +1% underlying move (first order); gamma_1pct: second-order P&L for
    a 1% move; vega_1pt: P&L for +1 vol point; theta_1d: P&L for one calendar day.
    """
    s, rate, vol = (a[0] for a in _option_inputs(pos, state))
    g = bs_greeks(s, pos.opt_strike, pos.opt_t, rate, vol, pos.opt_call)
    u = pos.opt_units
    return {
        "delta_1pct": u * g["delta"] * s * 0.01,
        "gamma_1pct": 0.5 * u * g["gamma"] * (0.01 * s) ** 2,
        "vega_1pt": u * g["vega"] / 100,
        "theta_1d": u * g["theta"] / 365,
    }
