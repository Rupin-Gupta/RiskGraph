import numpy as np
import pandas as pd
import pytest

# Synthetic panel in market_panel.parquet units (CI has no DVC data): prices, FX, yields in
# percent, VIX in points. Random walks, a few holiday-style gaps.
START = {"SPY": 400.0, "AAPL": 150.0, "MSFT": 300.0, "JPM": 140.0, "EURUSD=X": 1.1}
START |= {"INR=X": 80.0, "DGS2": 4.0, "DGS5": 3.8, "DGS10": 3.9, "VIXCLS": 18.0}
DAILY_VOL = {"EURUSD=X": 0.004, "INR=X": 0.003, "VIXCLS": 0.05}
RATES = ("DGS2", "DGS5", "DGS10")
GAPS = {"SPY": ["2019-07-04", "2019-12-25"], "DGS10": ["2019-10-14"]}


def make_panel(periods: int = 800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=periods, name="date")
    cols = {}
    for name, x0 in START.items():
        z = rng.standard_normal(periods)
        if name in RATES:
            cols[name] = x0 + np.cumsum(0.06 * z)  # ~6bp a day, in percent
        else:
            cols[name] = x0 * np.exp(np.cumsum(DAILY_VOL.get(name, 0.012) * z))
    panel = pd.DataFrame(cols, index=idx)
    for name, days in GAPS.items():
        panel.loc[pd.to_datetime(days), name] = np.nan
    return panel


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    return make_panel()
