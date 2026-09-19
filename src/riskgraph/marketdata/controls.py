"""Market data controls (SPEC §5): Pandera rules, staleness, cross-source, Isolation Forest.

Each check yields findings (date, factor, check, detail); run_controls adds the severity from
configs/risk.yaml `controls.severity`. Controls only report: they never change the data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from sklearn.ensemble import IsolationForest

from riskgraph.marketdata.schemas import missing_schema, moves, open_but_null, panel_schema
from riskgraph.pricing.market import FACTORS

COLUMNS = ["date", "factor", "check", "severity", "detail"]
Finding = tuple[pd.Timestamp, str, str, str]  # date, factor, check, detail


def pandera_findings(panel: pd.DataFrame, cfg: Mapping[str, Any]) -> list[Finding]:
    """Element failures of the panel schema and the missed-print rule.

    Structural failures (missing column, wrong dtype, unsorted or duplicate dates) raise.
    """
    out: list[Finding] = []
    for schema, frame in (
        (panel_schema(cfg), panel),
        (missing_schema(list(panel.columns)), open_but_null(panel, cfg)),
    ):
        try:
            schema.validate(frame, lazy=True)
        except pa.errors.SchemaErrors as e:
            fc = e.failure_cases
            if fc["index"].isna().any():
                raise ValueError(f"market panel failed structural checks:\n{fc}") from None
            for r in fc.itertuples():
                d, name, check = pd.Timestamp(r.index), str(r.column), str(r.check)
                if check == "max_move":
                    m = moves(panel[name], name in cfg["yields"])[d]
                    detail = f"1-day move {m:+.4g} exceeds {cfg['max_move'][name]}"
                elif check == "missing_print":
                    detail = "no print while most calendar peers printed"
                else:
                    detail = f"value {r.failure_case:g} fails {check}"
                out.append((d, name, f"pandera:{check}", detail))
    return out


def staleness_findings(panel: pd.DataFrame, cfg: Mapping[str, Any]) -> list[Finding]:
    """Liquid series with min_days or more consecutive identical prints (flagged from the
    min_days-th print on)."""
    n = int(cfg["staleness"]["min_days"])
    out: list[Finding] = []
    for f in cfg["staleness"]["series"]:
        s = panel[f].dropna()
        run = s.groupby(s.ne(s.shift()).cumsum()).cumcount() + 1
        for d in run.index[run >= n]:
            out.append((d, f, "staleness", f"unchanged for {run[d]} prints at {s[d]:g}"))
    return out


def cross_gap_bp(panel: pd.DataFrame, factor: str, source: str) -> pd.Series:
    """yfinance level vs the second source's previous print (FRED noon print of D-1), in bp."""
    ref = panel[source].shift(1).ffill(limit=3)
    gap: pd.Series = (panel[factor] / ref - 1) * 1e4
    return gap


def cross_source_findings(panel: pd.DataFrame, cfg: Mapping[str, Any]) -> list[Finding]:
    """yfinance vs FRED gaps beyond the tolerance, attributed to the yfinance factor."""
    tol = float(cfg["cross_source"]["tolerance_bp"])
    out: list[Finding] = []
    for f, src in cfg["cross_source"]["pairs"].items():
        gap = cross_gap_bp(panel, f, src)
        for d in gap.index[gap.abs() > tol]:
            out.append((d, f, "cross_source", f"{gap[d]:+.0f}bp vs {src} (tolerance {tol:g}bp)"))
    return out


def features(panel: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Isolation Forest features per (date, factor) with an observed move, all causal:

    z: move / trailing vol (vol excludes today); reversal: -move * previous move / vol^2
    (positive when today undoes yesterday, the causal form of SPEC's next-day reversal);
    vol_ratio: short-window vol / trailing vol; cross_source: |gap| / tolerance (FX only).
    """
    c = cfg["isolation_forest"]
    w = int(c["vol_window"])
    tol = float(cfg["cross_source"]["tolerance_bp"])
    frames = []
    for f in FACTORS:
        r = moves(panel[f], f in cfg["yields"]).dropna()
        vol = r.rolling(w, min_periods=w // 2).std().shift(1)
        src = cfg["cross_source"]["pairs"].get(f)
        gap = cross_gap_bp(panel, f, src).reindex(r.index).abs().fillna(0) / tol if src else 0.0
        x = pd.DataFrame(
            {
                "z": r / vol,
                "reversal": -r * r.shift(1) / vol**2,
                "vol_ratio": r.rolling(int(c["short_window"])).std() / vol,
                "cross_source": gap,
            }
        )
        x = x.replace([np.inf, -np.inf], np.nan).dropna()
        frames.append(x.set_index(pd.Index([f] * len(x), name="factor"), append=True))
    return pd.concat(frames)


@dataclass(frozen=True)
class Forest:
    """A fitted Isolation Forest and its score cutoff (higher score = more anomalous)."""

    model: IsolationForest
    cutoff: float


def fit_forest(panel: pd.DataFrame, cfg: Mapping[str, Any], seed: int) -> Forest:
    """Fit on dates up to train_end only. The panel is cut first, so no later print (and no
    evaluation-window data) can reach the fit, even through rolling features."""
    c = cfg["isolation_forest"]
    x = features(panel.loc[: pd.Timestamp(c["train_end"])], cfg).to_numpy()
    model = IsolationForest(n_estimators=int(c["n_estimators"]), random_state=seed).fit(x)
    cutoff = float(np.quantile(-model.score_samples(x), float(c["cutoff_quantile"])))
    return Forest(model, cutoff)


def forest_findings(panel: pd.DataFrame, cfg: Mapping[str, Any], forest: Forest) -> list[Finding]:
    x = features(panel, cfg)
    score = -forest.model.score_samples(x.to_numpy())
    hit = score > forest.cutoff
    cut = forest.cutoff
    return [
        (d, f, "isolation_forest", f"anomaly score {s:.3f} > cutoff {cut:.3f}, z {z:+.1f}")
        for (d, f), s, z in zip(x.index[hit], score[hit], x["z"].to_numpy()[hit], strict=True)
    ]


def run_controls(
    panel: pd.DataFrame,
    cfg: Mapping[str, Any],
    forest: Forest,
    dates: Iterable[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """All four checks on the panel; findings on `dates` only if given. Columns: COLUMNS."""
    rows = [
        *pandera_findings(panel, cfg),
        *staleness_findings(panel, cfg),
        *cross_source_findings(panel, cfg),
        *forest_findings(panel, cfg, forest),
    ]
    df = pd.DataFrame(rows, columns=["date", "factor", "check", "detail"])
    if dates is not None:
        df = df[df["date"].isin(pd.DatetimeIndex(list(dates)))]
    df["severity"] = [cfg["severity"][c.split(":")[0]] for c in df["check"]]
    return df.sort_values(["date", "factor", "check"], ignore_index=True)[COLUMNS]


def critical_factors(findings: pd.DataFrame) -> list[str]:
    """Risk factors with a critical finding: held flat in that day's revaluation."""
    bad = set(findings.loc[findings["severity"] == "critical", "factor"])
    return [f for f in FACTORS if f in bad]
