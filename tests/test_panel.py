from pathlib import Path

import pandas as pd

from riskgraph.cli import SERIES, build_panel


def test_panel_is_business_days_without_fill(tmp_path: Path) -> None:
    # Fri 2015-01-02, Sat 01-03 (weekend), Mon 01-05; Tue 01-06 missing on purpose.
    dates = pd.to_datetime(["2015-01-02", "2015-01-03", "2015-01-05", "2015-01-07"])
    for source, names in SERIES.items():
        col = "Close" if source == "yfinance" else "value"
        for name in names:
            (tmp_path / source).mkdir(exist_ok=True)
            pd.DataFrame({col: [1.0, 2.0, 3.0, 4.0]}, index=dates).to_parquet(
                tmp_path / source / f"{name}.parquet"
            )

    panel = build_panel(tmp_path)

    assert list(panel.index) == list(pd.bdate_range("2015-01-01", "2015-01-07"))
    assert list(panel.columns) == [n for names in SERIES.values() for n in names]
    assert panel["SPY"].isna().tolist() == [True, False, False, True, False]  # 01-06 not filled
    assert pd.Timestamp("2015-01-03") not in panel.index
