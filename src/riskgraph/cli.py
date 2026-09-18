"""RiskGraph command-line interface."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import typer
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True, help="Market data.")
app.add_typer(data_app, name="data")

START = "2015-01-01"
RAW = Path("data/raw")
PANEL = Path("data/processed/market_panel.parquet")
# SPEC §3.2. Panel column per series: yfinance Close, FRED value (ADR-003).
SERIES = {
    "yfinance": ["SPY", "AAPL", "MSFT", "JPM", "EURUSD=X", "INR=X"],
    "fred": ["DGS2", "DGS5", "DGS10", "VIXCLS", "DEXUSEU", "DEXINUS"],
}


def fetch_yfinance(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=START,
        auto_adjust=False,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    if df is None or df.empty:
        raise RuntimeError("no rows returned")
    # Yahoo appends an in-progress bar for the current session (always, for FX): drop it.
    return df[df.index.date < date.today()]


def fetch_fred(series_id: str) -> pd.DataFrame:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not set (see .env.example)")
    try:
        s = Fred(api_key=key).get_series(series_id, observation_start=START)
    except Exception as e:  # never let the key leak through an error message
        raise RuntimeError(str(e).replace(key, "***")) from None
    return s.to_frame("value")


FETCH = {"yfinance": fetch_yfinance, "fred": fetch_fred}


def build_panel(raw: Path = RAW) -> pd.DataFrame:
    """One column per series on a Mon-Fri index. No fill: holidays and gaps stay NaN."""
    cols = {}
    for source, names in SERIES.items():
        for name in names:
            df = pd.read_parquet(raw / source / f"{name}.parquet")
            s = df["Close" if source == "yfinance" else "value"]
            cols[name] = s[~s.index.duplicated(keep="last")]
    panel = pd.DataFrame(cols)
    return panel.reindex(pd.bdate_range(START, panel.index.max(), name="date"))


@data_app.command()
def download(
    refresh: bool = typer.Option(False, "--refresh", help="Re-download series already on disk."),
    seed: int = typer.Option(42, help="Recorded for lineage; downloading is not random."),
) -> None:
    """Download SPEC §3.2 series to data/raw and build the business-day panel."""
    load_dotenv()
    failed = []
    for source, names in SERIES.items():
        for name in names:
            path = RAW / source / f"{name}.parquet"
            status = "cached"
            if refresh or not path.exists():
                try:
                    df = FETCH[source](name)
                except Exception as e:
                    failed.append(f"{source}/{name}: {e}")
                    typer.echo(f"{source:<9} {name:<9} FAILED")
                    continue
                df.index = pd.DatetimeIndex(df.index).tz_localize(None).rename("date")
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(path)
                status = "downloaded"
            df = pd.read_parquet(path)
            first, last = df.index.min(), df.index.max()
            typer.echo(
                f"{source:<9} {name:<9} {len(df):>6} rows  {first:%Y-%m-%d} -> {last:%Y-%m-%d}  "
                f"{status}"
            )

    if failed:
        typer.echo(f"\n{len(failed)} series failed; panel not built:", err=True)
        for f in failed:
            typer.echo(f"  {f}", err=True)
        raise typer.Exit(1)

    panel = build_panel()
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL)
    end = f"{panel.index.max():%Y-%m-%d}"
    config = {"start": START, "end": end, "series": SERIES, "seed": seed}
    PANEL.with_suffix(".config.json").write_text(json.dumps(config, indent=2) + "\n")
    typer.echo(f"\npanel: {len(panel)} business days x {panel.shape[1]} series -> {PANEL}")


if __name__ == "__main__":
    app()
