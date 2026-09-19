"""Pandera schemas for raw market series and the processed panel (SPEC §5).

Thresholds come from the `controls:` block of configs/risk.yaml.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

YF_PRICES = ("Open", "High", "Low", "Close", "Adj Close")
MONOTONIC = pa.Check(lambda i: i.is_monotonic_increasing, name="monotonic_index")


def moves(s: pd.Series, is_yield: bool) -> pd.Series:
    """1-day moves between observed prints, on s's index: bp for yields, log returns otherwise.

    Non-positive prices are skipped (the positivity rule flags them), so the next move spans them.
    """
    if is_yield:
        obs = s.dropna()
        return (obs.diff() * 100).reindex(s.index)
    obs = s[s > 0]
    return np.log(obs).diff().reindex(s.index)


def value_checks(name: str, cfg: Mapping[str, Any]) -> list[pa.Check]:
    """Yields within the configured range (percent); every other series strictly positive."""
    if name in cfg["yields"]:
        lo, hi = cfg["yield_range_pct"]
        return [pa.Check(lambda s: s.between(lo, hi), name="yield_range")]
    return [pa.Check(lambda s: s > 0, name="positive_price")]


def max_move(name: str, cfg: Mapping[str, Any]) -> pa.Check:
    """|1-day move| at most the configured limit (log return, or bp for yields)."""
    limit = float(cfg["max_move"][name])
    is_yield = name in cfg["yields"]
    return pa.Check(lambda s: ~(moves(s, is_yield).abs() > limit), name="max_move")


def panel_schema(cfg: Mapping[str, Any]) -> pa.DataFrameSchema:
    """Processed panel: monotonic unique dates, value rules, and max move. Gaps may be null."""
    cols = {
        n: pa.Column(float, [*value_checks(n, cfg), max_move(n, cfg)], nullable=True)
        for n in cfg["max_move"]
    }
    return pa.DataFrameSchema(cols, index=pa.Index(checks=[MONOTONIC], unique=True))


def open_but_null(panel: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """True where a series is null while most of its calendar peers print (a missed print).

    A holiday nulls the whole calendar group, so it is not flagged.
    """
    out = pd.DataFrame(False, index=panel.index, columns=panel.columns)
    for names in cfg["calendars"].values():
        seen = panel[names].notna()
        for n in names:
            peers = seen.drop(columns=n)
            out[n] = ~seen[n] & (peers.sum(axis=1) > peers.shape[1] / 2)
    return out


def missing_schema(columns: list[str]) -> pa.DataFrameSchema:
    """Validates open_but_null(panel). Pandera drops null failure values, so the rule checks
    the null mask rather than the panel itself."""
    return pa.DataFrameSchema(
        {n: pa.Column(bool, pa.Check(lambda s: ~s, name="missing_print")) for n in columns}
    )


def raw_schema(source: str, name: str, cfg: Mapping[str, Any]) -> pa.DataFrameSchema:
    """A raw pull as written by `riskgraph data download`: yfinance bars or one FRED series."""
    index = pa.Index(checks=[MONOTONIC], unique=True)
    if source == "yfinance":
        cols = {c: pa.Column(float, pa.Check.gt(0)) for c in YF_PRICES}
        cols["Volume"] = pa.Column(int, pa.Check.ge(0))
    else:  # FRED leaves holidays null
        cols = {"value": pa.Column(float, value_checks(name, cfg), nullable=True)}
    return pa.DataFrameSchema(cols, index=index)
