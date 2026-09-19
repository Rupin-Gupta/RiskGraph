"""Market data controls evaluation (SPEC §5).

Injects 200 seeded corruptions (40 of each type) into a held-out copy of the panel for
2022-2025 and scores two variants: rules only (Pandera, staleness, cross-source) and rules +
Isolation Forest. Writes reports/metrics/dq.json and dq.config.json.

Run: python -m riskgraph.eval.dq --seed 42   (make eval-dq)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import typer

from riskgraph.cli import METRICS, PANEL, load_yaml, write_json
from riskgraph.marketdata.controls import fit_forest, run_controls
from riskgraph.marketdata.schemas import moves
from riskgraph.pricing.market import FACTORS

TYPES = ("stale_run", "spike_x10", "sign_flip", "missing", "decimal_shift")
N_PER_TYPE = 40
START, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2025-12-31")
SPACING = 10  # minimum business days between two corruptions of the same factor
Cell = tuple[str, pd.Timestamp]  # (factor, date)
VARIANTS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "rules": lambda f: f[f["check"] != "isolation_forest"],
    "rules_iforest": lambda f: f,
}


@dataclass(frozen=True)
class Corruption:
    kind: str
    factor: str
    date: pd.Timestamp  # first corrupted print
    scale: float = 1.0  # decimal_shift only: 10 or 0.1


def corrupt(s: pd.Series, c: Corruption, is_yield: bool) -> pd.Series:
    """A copy of one panel column with the corruption applied at c.date.

    stale_run: that print and the next two repeat the previous print. spike_x10: the day's
    move is multiplied by 10 (log return, or bp for yields). sign_flip: the level is negated.
    missing: the print is removed. decimal_shift: the level is multiplied by 10 or 0.1.
    """
    out = s.copy()
    obs = s.dropna()
    i = obs.index.get_loc(c.date)
    prev, x = float(obs.iloc[i - 1]), float(obs.iloc[i])
    if c.kind == "stale_run":
        out.loc[obs.index[i : i + 3]] = prev
    elif c.kind == "spike_x10":
        out.loc[c.date] = prev + 10 * (x - prev) if is_yield else prev * (x / prev) ** 10
    elif c.kind == "sign_flip":
        out.loc[c.date] = -x
    elif c.kind == "missing":
        out.loc[c.date] = np.nan
    else:
        out.loc[c.date] = x * c.scale
    return out


def footprint(clean: pd.Series, bad: pd.Series, is_yield: bool) -> set[pd.Timestamp]:
    """Dates whose level or 1-day move differs from the clean data: the corrupted prints and
    the next print, whose move (and so the engine's shock) starts from a corrupted one."""

    def differ(a: pd.Series, b: pd.Series) -> pd.Series:
        return ~((a == b) | (a.isna() & b.isna()))

    changed = differ(clean, bad) | differ(moves(clean, is_yield), moves(bad, is_yield))
    return set(clean.index[changed])


def sample(
    panel: pd.DataFrame,
    cfg: Mapping[str, Any],
    n_per_type: int,
    seed: int,
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
) -> list[Corruption]:
    """n_per_type corruptions of each type at random (factor, date) cells, deterministic in
    seed. Each changes the data, keeps its footprint inside [start, end], and sits at least
    SPACING business days from other corruptions of the same factor."""
    rng = np.random.default_rng(seed)
    taken: dict[str, list[int]] = {f: [] for f in FACTORS}
    out = []
    for kind in rng.permutation(np.repeat(TYPES, n_per_type)):
        while True:
            f = FACTORS[int(rng.integers(len(FACTORS)))]
            obs = panel[f].dropna()
            i = int(rng.integers(1, len(obs) - 3))
            d, last = obs.index[i], obs.index[i + 3]  # i + 3: next print after a stale run
            p = panel.index.get_loc(d)
            if d < start or last > end or any(abs(p - q) < SPACING for q in taken[f]):
                continue
            c = Corruption(str(kind), f, d, float(rng.choice([10.0, 0.1])))
            is_yield = f in cfg["yields"]
            if footprint(panel[f], corrupt(panel[f], c, is_yield), is_yield):
                taken[f].append(p)
                out.append(c)
                break
    return out


def inject(panel: pd.DataFrame, cfg: Mapping[str, Any], cs: Sequence[Corruption]) -> pd.DataFrame:
    out = panel.copy()
    for c in cs:
        out[c.factor] = corrupt(out[c.factor], c, c.factor in cfg["yields"])
    return out


def score(findings: pd.DataFrame, footprints: Sequence[set[Cell]]) -> dict[str, float]:
    """Precision: share of flagged factor cells inside some corruption's footprint. Recall:
    share of corruptions with at least one flagged footprint cell."""
    f = findings[findings["factor"].isin(FACTORS)]
    flagged: set[Cell] = set(zip(f["factor"], f["date"], strict=True))
    detected = sum(bool(fp & flagged) for fp in footprints)
    true_flags = len(flagged & set().union(*footprints))
    p = true_flags / len(flagged) if flagged else 0.0
    r = detected / len(footprints)
    return {
        "corruptions": len(footprints),
        "detected": detected,
        "flags": len(flagged),
        "true_flags": true_flags,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
    }


def evaluate(panel: pd.DataFrame, cfg: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Per type: that type's 40 corruptions injected alone, so false alarms are attributable.
    Overall: all 200 injected together. Findings are scored inside [START, END] only."""
    held = panel.loc[:END].copy()  # the 2022-2025 window plus the history the checks look back on
    forest = fit_forest(held, cfg, seed)  # cut at train_end inside, before 2022
    cs = sample(held, cfg, N_PER_TYPE, seed)
    fps = []
    for c in cs:
        is_yield = c.factor in cfg["yields"]
        dates = footprint(held[c.factor], corrupt(held[c.factor], c, is_yield), is_yield)
        fps.append({(c.factor, d) for d in dates})
    out: dict[str, Any] = {v: {} for v in VARIANTS}
    for name in (*TYPES, "overall"):
        sel = [i for i, c in enumerate(cs) if name in ("overall", c.kind)]
        findings = run_controls(inject(held, cfg, [cs[i] for i in sel]), cfg, forest)
        findings = findings[findings["date"].between(START, END)]
        for v, keep in VARIANTS.items():
            out[v][name] = score(keep(findings), [fps[i] for i in sel])
    return {
        "window": {"start": f"{START:%Y-%m-%d}", "end": f"{END:%Y-%m-%d}"},
        "corruptions": {"total": len(cs), "per_type": N_PER_TYPE},
        "isolation_forest": {"train_end": str(cfg["isolation_forest"]["train_end"])}
        | {"cutoff": round(forest.cutoff, 4)},
        **out,
    }


def main(seed: int = typer.Option(42, help="Seed for corruption sampling and the forest.")) -> None:
    cfg = load_yaml("risk")["controls"]
    doc = evaluate(pd.read_parquet(PANEL), cfg, seed)
    write_json(METRICS / "dq.json", doc)
    config = {"seed": seed, "panel": PANEL, "types": TYPES, "per_type": N_PER_TYPE}
    config |= {"spacing_bdays": SPACING, "controls": cfg}
    write_json(METRICS / "dq.config.json", config)
    typer.echo(f"{'variant':<14} {'type':<14} precision  recall     f1")
    for v in VARIANTS:
        for name, r in doc[v].items():
            p, rec, f1 = r["precision"], r["recall"], r["f1"]
            typer.echo(f"{v:<14} {name:<14} {p:>9.3f} {rec:>7.3f} {f1:>6.3f}")
    typer.echo(f"-> {METRICS / 'dq.json'}")


if __name__ == "__main__":
    typer.run(main)
