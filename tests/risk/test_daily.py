from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from riskgraph.book.generate import generate
from riskgraph.book.schema import Trade
from riskgraph.pricing.market import IDX
from riskgraph.risk.backtest import history, report
from riskgraph.risk.daily import Configs, run
from riskgraph.risk.factors import RiskContext
from riskgraph.risk.var import SCOPES


def cfg(name: str) -> Any:
    return yaml.safe_load(Path(f"configs/{name}.yaml").read_text())


LIMITS = cfg("limits")
LIMITS["var_99_1d"] = {"firm": 5e6, "desks": {"fx": 1e6, "rates": 1e6, "equity_derivatives": 3e6}}
LIMITS["stress_loss"] = {"firm": 2e7}
CFGS = Configs(cfg("risk"), LIMITS, cfg("scenarios"))


@pytest.fixture(scope="module")
def setup(panel: pd.DataFrame) -> tuple[RiskContext, list[Trade], pd.Timestamp]:
    ctx = RiskContext.from_panel(panel, CFGS.risk)
    book, _ = generate(cfg("book"), ctx.state(ctx.levels.index[-30]), seed=42)
    return ctx, book, ctx.levels.index[-1]


def test_var_explain_sums_to_the_var_change(setup: tuple) -> None:
    ctx, book, day = setup
    big = Trade(
        trade_id="INJ001",
        desk="equity_derivatives",
        instrument_type="cash_equity",
        currency="USD",
        notional=2e8,
        trade_date=day.date(),
        underlying="SPY",
    )
    out = run(ctx, [*book, big], book, day, CFGS, seed=42)
    base_prev = run(ctx, book, book, ctx.prev(day), CFGS, seed=42)
    for s in SCOPES:
        e = out["explain"][s]
        assert e["position"] + e["market"] + e["interaction"] == pytest.approx(e["total"], abs=1e-8)
        prev_var = base_prev["metrics"][s]["var_99_1d_hs"]
        assert e["total"] == pytest.approx(out["metrics"][s]["var_99_1d_hs"] - prev_var)
    assert out["explain"]["equity_derivatives"]["position"] > 1e6
    assert out["explain"]["fx"]["position"] == 0.0
    assert out["explain"]["equity_derivatives"]["top_trades"][0]["trade_id"] == "INJ001"


def test_unchanged_book_has_only_a_market_effect(setup: tuple) -> None:
    ctx, book, day = setup
    out = run(ctx, book, book, day, CFGS, seed=42)
    for s in SCOPES:
        assert out["explain"][s]["position"] == 0.0
        assert out["explain"][s]["interaction"] == pytest.approx(0.0, abs=1e-8)
    firm = out["metrics"]["firm"]
    # Synthetic shocks are normal, so the three VaR methods should roughly agree.
    for other in ("var_99_1d_mc", "var_99_1d_param"):
        assert 0.7 < firm[other] / firm["var_99_1d_hs"] < 1.3
    assert firm["es_975_1d_hs"] > 0 and firm["stress_worst_loss"] > 0
    assert {(r["scope"], r["metric"]) for r in out["limits"]} == {
        *((s, "var_99_1d") for s in SCOPES),
        ("firm", "stress_loss"),
    }


def test_excluded_factors_are_held_flat(setup: tuple) -> None:
    ctx, book, day = setup
    flat = replace(ctx, excluded=("SPY", "AAPL"))
    cols = [IDX["SPY"], IDX["AAPL"]]
    assert not flat.window_shocks(day)[:, cols].any()
    assert ctx.window_shocks(day)[:, cols].all()  # the observed shocks are not modified
    assert flat.state(day).equity_vol == ctx.state(day).equity_vol  # vols from observed moves

    out, base = run(flat, book, book, day, CFGS, seed=42), run(ctx, book, book, day, CFGS, seed=42)
    m, b = out["metrics"], base["metrics"]
    assert m["fx"]["var_99_1d_hs"] == b["fx"]["var_99_1d_hs"]
    assert m["equity_derivatives"]["var_99_1d_hs"] != b["equity_derivatives"]["var_99_1d_hs"]
    assert np.isfinite(m["firm"]["var_99_1d_mc"])  # Cholesky of the live block only
    assert m["equity_derivatives"]["sens_SPY"] == b["equity_derivatives"]["sens_SPY"]
    top = out["explain"]["equity_derivatives"]["top_factors"]
    contrib = {r["factor"]: r["contribution"] for r in top}
    assert contrib["SPY"] == contrib["AAPL"] == 0.0


def test_backtest_history_and_report(setup: tuple) -> None:
    ctx, book, day = setup
    start = ctx.levels.index[-5]
    hist = history(ctx, book, f"{start:%Y-%m-%d}", f"{day:%Y-%m-%d}", CFGS, seed=42)
    assert list(hist.index) == list(ctx.levels.index[-5:])
    assert (hist["asof"] == ctx.levels.index[-6:-1]).all()
    # Row t's VaR is the prior day's run-daily VaR.
    prior = run(ctx, book, book, ctx.prev(day), CFGS, seed=42)["metrics"]["firm"]
    assert hist["var_hs_firm"].iloc[-1] == pytest.approx(prior["var_99_1d_hs"])
    assert hist["var_mc_firm"].iloc[-1] == pytest.approx(prior["var_99_1d_mc"])
    rep = report(hist, CFGS)
    assert rep["window"]["days"] == 5
    assert set(rep["window"]["historical"]) == set(SCOPES)
    assert "traffic_light" not in rep["full_period"]["monte_carlo"]["firm"]
