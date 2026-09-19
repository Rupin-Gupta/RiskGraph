from dataclasses import replace
from datetime import date
from typing import Any

import numpy as np
import pytest

from riskgraph.book.schema import Trade
from riskgraph.pricing.black_scholes import bs_price
from riskgraph.pricing.curve import discount_factors
from riskgraph.pricing.fx import foreign_df, forward_pv
from riskgraph.pricing.market import FACTORS, IDX, MarketState, apply_shocks
from riskgraph.pricing.portfolio import revalue, roll
from riskgraph.pricing.rates import bond_cashflows, pv, swap_cashflows

D0, D1 = date(2024, 1, 2), date(2025, 1, 2)  # D1 - D0 = 366 days
T = 366 / 365.25
LEVELS = {"SPY": 470.0, "AAPL": 185.0, "MSFT": 370.0, "JPM": 170.0, "EURUSD=X": 1.10}
LEVELS |= {"INR=X": 83.0, "DGS2": 0.043, "DGS5": 0.039, "DGS10": 0.040, "VIXCLS": 13.0}
STATE = MarketState(
    D0,
    np.array([[LEVELS[f] for f in FACTORS]]),
    {"AAPL": 0.22, "MSFT": 0.25, "JPM": 0.2},
    {"EUR": 0.02, "INR": 0.065},
)


def trade(kind: str, desk: str, **kw: Any) -> Trade:
    base = dict(trade_id=kind, desk=desk, instrument_type=kind, currency="USD", trade_date=D0)
    return Trade.model_validate(base | kw)


BOOK = [
    trade("cash_equity", "equity_derivatives", notional=-2e6, underlying="MSFT"),
    trade(
        "european_option",
        "equity_derivatives",
        notional=5e6,
        underlying="SPY",
        strike=480.0,
        option_type="call",
        maturity_date=D1,
    ),
    trade(
        "fx_forward",
        "fx",
        notional=1e7,
        currency_pair="USDINR",
        forward_rate=84.0,
        maturity_date=D1,
        counterparty_id="CP01",
    ),
    trade(
        "interest_rate_swap",
        "rates",
        notional=5e7,
        fixed_rate=0.041,
        pay_receive="payer",
        maturity_date=date(2029, 1, 2),
        counterparty_id="CP02",
    ),
    trade("treasury_bond", "rates", notional=2e7, coupon=0.04, maturity_date=date(2034, 1, 2)),
]
REF = {"european_option": 470.0, "fx_forward": 83.0}  # trade-date levels


def test_revalue_matches_single_trade_pricers_on_trade_date() -> None:
    pv_ = revalue(roll(BOOK, STATE, REF), STATE)[0]
    curve = STATE.curve
    r = -np.log(discount_factors(curve, T)[0, 0]) / T
    df_usd, df_inr = discount_factors(curve, T)[0, 0], float(foreign_df(0.065, T))
    t_swap, t_bond = BOOK[3].maturity_date, BOOK[4].maturity_date
    assert t_swap is not None and t_bond is not None
    expected = [
        -2e6,
        5e6 / 470 * float(bs_price(470.0, 480.0, T, r, 0.13, True)),
        float(forward_pv(1e7, 84.0, 83.0, df_usd, df_inr)) / 83.0,
        pv(*swap_cashflows(5e7, 0.041, (t_swap - D0).days / 365.25, True), curve)[0],
        pv(*bond_cashflows(2e7, 0.04, (t_bond - D0).days / 365.25), curve)[0],
    ]
    np.testing.assert_allclose(pv_, expected, rtol=1e-12)


def test_roll_keeps_moneyness_and_usd_notional() -> None:
    # Equity levels double since the trade date: strike and units rescale, USD value is unchanged.
    later = replace(STATE, levels=STATE.levels * np.where(np.arange(len(FACTORS)) < 4, 2.0, 1.0))
    base, rolled = (
        revalue(roll(BOOK, STATE, REF), STATE)[0],
        revalue(roll(BOOK, later, REF), later)[0],
    )
    assert rolled[:2] == pytest.approx(base[:2])


def test_shocks_revalue_all_scenarios_at_once() -> None:
    pos = roll(BOOK, STATE, REF)
    shocks = np.zeros((3, len(FACTORS)))
    shocks[1, IDX["MSFT"]] = np.log(1.1)  # MSFT +10%
    shocks[2, [IDX["DGS2"], IDX["DGS5"], IDX["DGS10"]]] = 100.0  # curve +100bp
    pnl = revalue(pos, apply_shocks(STATE, shocks)) - revalue(pos, STATE)
    assert pnl.shape == (3, len(BOOK))
    # Swap PV is a difference of legs worth millions: rounding there is ~1e-9 USD and varies
    # by platform, so zero P&L is checked to a micro-dollar.
    np.testing.assert_allclose(pnl[0], 0.0, atol=1e-6)
    assert pnl[1, 0] == pytest.approx(-2e5)  # short 2M of MSFT loses 10%
    assert pnl[2, 4] < 0 < pnl[2, 3]  # long bond loses, payer swap gains
