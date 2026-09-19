from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
import yaml

from riskgraph.marketdata.controls import Forest, critical_factors, fit_forest, run_controls
from riskgraph.marketdata.schemas import YF_PRICES, raw_schema

CFG = yaml.safe_load(Path("configs/risk.yaml").read_text())["controls"]
CFG["isolation_forest"] = CFG["isolation_forest"] | {"train_end": "2021-06-30"}  # synthetic dates
D = pd.Timestamp("2021-11-10")


def with_fred(panel: pd.DataFrame) -> pd.DataFrame:
    """Conftest panel with its gaps filled and FRED prints equal to the next yfinance close."""
    p = panel.ffill()
    return p.assign(DEXUSEU=p["EURUSD=X"].shift(-1), DEXINUS=p["INR=X"].shift(-1))


@pytest.fixture(scope="module")
def base(panel: pd.DataFrame) -> pd.DataFrame:
    return with_fred(panel)


@pytest.fixture(scope="module")
def forest(base: pd.DataFrame) -> Forest:
    return fit_forest(base, CFG, seed=42)


def rule_hits(p: pd.DataFrame, forest: Forest) -> set[tuple[pd.Timestamp, str, str]]:
    f = run_controls(p, CFG, forest)
    f = f[f["check"] != "isolation_forest"]
    return set(zip(f["date"], f["factor"], f["check"], strict=True))


def test_clean_panel_passes_every_rule(base: pd.DataFrame, forest: Forest) -> None:
    assert rule_hits(base, forest) == set()


def stale_msft(s: pd.Series) -> pd.Series:
    i = s.index.get_loc(D)
    return s.mask((s.index > s.index[i - 3]) & (s.index <= D), s.iloc[i - 3])  # 4 equal prints


def at_d(fn: Callable[[float], float]) -> Callable[[pd.Series], pd.Series]:
    return lambda s: s.mask(s.index == D, fn(s[D]))


CASES: dict[str, tuple[str, str, Callable[[pd.Series], pd.Series]]] = {
    "negative price": ("JPM", "pandera:positive_price", at_d(lambda x: -x)),
    "yield out of range": ("DGS10", "pandera:yield_range", at_d(lambda x: 25.0)),
    "jump": ("SPY", "pandera:max_move", at_d(lambda x: x * 1.6)),
    "missed print": ("AAPL", "pandera:missing_print", at_d(lambda x: np.nan)),
    "stale": ("MSFT", "staleness", stale_msft),
    "cross-source gap": ("EURUSD=X", "cross_source", at_d(lambda x: x * 1.03)),
}


@pytest.mark.parametrize("case", CASES)
def test_each_rule_flags_its_failure(case: str, base: pd.DataFrame, forest: Forest) -> None:
    factor, check, corrupt = CASES[case]
    p = base.assign(**{factor: corrupt(base[factor])})
    assert (D, factor, check) in rule_hits(p, forest)
    today = run_controls(p, CFG, forest, dates=[D])
    assert critical_factors(today) == [factor]


def test_isolation_forest_flags_a_move_the_rules_allow(base: pd.DataFrame, forest: Forest) -> None:
    i = base.index.get_loc(D)
    p = base.copy()
    p.loc[D, "MSFT"] = p["MSFT"].iloc[i - 1] * np.exp(10 * 0.012)  # a 10-sigma day, under 0.35

    def hit(x: pd.DataFrame) -> bool:
        f = run_controls(x, CFG, forest, dates=[D])
        return bool(((f["factor"] == "MSFT") & (f["check"] == "isolation_forest")).any())

    assert hit(p) and not hit(base)
    assert not any(d == D and f == "MSFT" for d, f, _ in rule_hits(p, forest))
    assert run_controls(p, CFG, forest, dates=[D])["severity"].eq("warning").all()


def test_forest_never_sees_data_after_train_end(base: pd.DataFrame, forest: Forest) -> None:
    junk = base.copy()
    junk.loc["2021-07-01":] *= 100.0
    assert fit_forest(junk, CFG, seed=42).cutoff == forest.cutoff


def test_structural_failure_raises(base: pd.DataFrame, forest: Forest) -> None:
    with pytest.raises(ValueError, match="structural"):
        run_controls(base.iloc[::-1], CFG, forest)


def test_raw_schemas() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3, name="date")
    bars = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in YF_PRICES} | {"Volume": [0, 5, 7]}, index=idx)
    raw_schema("yfinance", "SPY", CFG).validate(bars)
    with pytest.raises(pa.errors.SchemaError):
        raw_schema("yfinance", "SPY", CFG).validate(bars.assign(Close=[1.0, -2.0, 3.0]))
    fred = pd.DataFrame({"value": [4.1, np.nan, 4.2]}, index=idx)  # holiday null is fine
    raw_schema("fred", "DGS2", CFG).validate(fred)
    with pytest.raises(pa.errors.SchemaError):
        raw_schema("fred", "DGS2", CFG).validate(fred.assign(value=[4.1, 25.0, 4.2]))
