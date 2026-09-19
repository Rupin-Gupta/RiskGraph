import numpy as np
import pandas as pd
import pytest

from riskgraph.pricing.market import FACTORS, IDX
from riskgraph.risk.factors import RiskContext, ewma_cov

CFG = {"ewma_lambda": 0.94, "var": {"window_days": 500}}
CFG |= {"fx_forward": {"foreign_rate_proxy": {"EUR": 0.02, "INR": 0.065}}}


def test_ewma_cov_known_value() -> None:
    # lambda = 0.5, shocks 1, 2, 3 (oldest first): weights 1/7, 2/7, 4/7 -> (1 + 8 + 36) / 7.
    assert ewma_cov(np.array([[1.0], [2.0], [3.0]]), 0.5)[0, 0] == pytest.approx(45 / 7)


def test_gap_policy_drops_incomplete_dates_and_spans_the_gap(panel: pd.DataFrame) -> None:
    ctx = RiskContext.from_panel(panel, CFG)
    gaps = pd.to_datetime(["2019-07-04", "2019-10-14", "2019-12-25"])
    assert list(ctx.dropped) == list(gaps)
    # The shock on 2019-07-05 spans 07-03 -> 07-05; yields move in bp.
    s = ctx.shocks.loc["2019-07-05"]
    lv = panel.loc[["2019-07-03", "2019-07-05"]]
    assert s["SPY"] == pytest.approx(np.log(lv["SPY"].iloc[1] / lv["SPY"].iloc[0]))
    assert s["DGS5"] == pytest.approx((lv["DGS5"].iloc[1] - lv["DGS5"].iloc[0]) * 100)


def test_state_and_window(panel: pd.DataFrame) -> None:
    ctx = RiskContext.from_panel(panel, CFG)
    day = ctx.levels.index[-1]
    state = ctx.state(day)
    assert state.levels.shape == (1, len(FACTORS))
    assert state.levels[0, IDX["DGS5"]] == pytest.approx(panel.loc[day, "DGS5"] / 100)
    assert ctx.window_shocks(day).shape == (500, len(FACTORS))
    assert 0.1 < state.equity_vol["AAPL"] < 0.3  # ~1.2% daily -> ~19% annualized
    with pytest.raises(ValueError, match="500"):
        ctx.window_shocks(ctx.levels.index[100])
