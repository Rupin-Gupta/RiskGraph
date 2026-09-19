"""Synthetic trading book generator and book I/O (SPEC §3.1). Deterministic by seed."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, timedelta
from itertools import cycle, islice, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskgraph.book.schema import Trade
from riskgraph.pricing.curve import discount_factors, zero_rates
from riskgraph.pricing.fx import foreign_df, forward_rate
from riskgraph.pricing.market import MarketState
from riskgraph.pricing.portfolio import FOREIGN_CCY, PAIR_FACTOR, QUOTE_IS_USD, YEAR_DAYS
from riskgraph.pricing.rates import par_swap_rate

# Fictional counterparty names: invented first words so no name belongs to a real firm.
NAME_STEMS = ["Altoria", "Brevane", "Castelmor", "Dunlivet", "Elmswick", "Farrowdale"]
NAME_STEMS += ["Grisholm", "Hallowmere", "Ivenstrand", "Kelmoor", "Lorwick", "Marrowgate"]
NAME_KINDS = ["Capital", "Securities", "Asset Management", "Partners", "Bank", "Holdings"]


def _months(tenor: str) -> int:
    return int(tenor[:-1]) * (12 if tenor.endswith("Y") else 1)


def _add_months(d: date, months: int) -> date:
    return (pd.Timestamp(d) + pd.DateOffset(months=months)).date()


def generate(
    cfg: Mapping[str, Any], state: MarketState, seed: int
) -> tuple[list[Trade], list[dict[str, str]]]:
    """Generate the book from configs/book.yaml, struck against `state` (the book date).

    Returns trades and counterparties (one netting set each). Notionals follow the Trade
    units; FX contract rates are within 1% of the fair forward, swap rates within 25bp of
    par, bond coupons at the curve yield (rounded to 1/8%), strikes within 10% of spot.
    """
    rng = np.random.default_rng(seed)
    td = state.date
    ids = list(cfg["counterparties"]["ids"])
    stems = rng.choice(NAME_STEMS, size=len(ids), replace=False)
    cps = [
        {"counterparty_id": c, "name": f"{s} {rng.choice(NAME_KINDS)}", "netting_set_id": f"NS-{c}"}
        for c, s in zip(ids, stems, strict=True)
    ]
    trades: list[Trade] = []

    def notional(spec: Mapping[str, Any], signed: bool = True) -> float:
        lo, hi = spec["notional_musd"]
        size = round(rng.uniform(lo, hi) * 2) / 2 * 1e6  # 0.5M steps
        return float(rng.choice([-1.0, 1.0])) * size if signed else size

    def add(prefix: str, i: int, kind: str, desk: str, **kw: Any) -> None:
        tid = f"{prefix}{i + 1:03d}"
        base: dict[str, Any] = {"currency": "USD", "trade_date": td}
        trades.append(
            Trade.model_validate(
                base | kw | {"trade_id": tid, "instrument_type": kind, "desk": desk}
            )
        )

    fx = cfg["desks"]["fx"]
    spec = fx["fx_forward"]
    lo, hi = (_months(t) for t in spec["tenor_range"])
    for i, pair in enumerate(islice(cycle(spec["currency_pairs"]), spec["count"])):
        mat = _add_months(td, int(rng.integers(lo, hi + 1)))
        t = (mat - td).days / YEAR_DAYS
        usd = float(discount_factors(state.curve, t)[0, 0])
        foreign = float(foreign_df(state.foreign_rates[FOREIGN_CCY[pair]], t))
        dfs = (foreign, usd) if QUOTE_IS_USD[pair] else (usd, foreign)
        fwd = float(forward_rate(state[PAIR_FACTOR[pair]][0], *dfs))
        add(
            "FXF",
            i,
            "fx_forward",
            "fx",
            currency=pair[:3],
            notional=notional(spec),
            maturity_date=mat,
            currency_pair=pair,
            forward_rate=round(fwd * (1 + rng.uniform(-0.01, 0.01)), 4),
            counterparty_id=str(rng.choice(ids)),
        )
    spec = fx["fx_spot"]
    for i, pair in enumerate(islice(cycle(spec["currency_pairs"]), spec["count"])):
        add(
            "FXS",
            i,
            "fx_spot",
            "fx",
            currency=pair[:3],
            notional=notional(spec),
            maturity_date=td + timedelta(days=2),  # T+2 settlement
            currency_pair=pair,
            forward_rate=round(float(state[PAIR_FACTOR[pair]][0]), 4),
        )

    rates = cfg["desks"]["rates"]
    spec = rates["interest_rate_swap"]
    legs = islice(cycle(product(spec["tenors"], spec["pay_receive"])), spec["count"])
    for i, (tenor, side) in enumerate(legs):
        mat = _add_months(td, _months(tenor))
        par = float(par_swap_rate((mat - td).days / YEAR_DAYS, state.curve)[0])
        add(
            "IRS",
            i,
            "interest_rate_swap",
            "rates",
            notional=notional(spec, signed=False),
            maturity_date=mat,
            fixed_rate=round(par + rng.uniform(-25, 25) / 1e4, 6),
            pay_receive=side,
            counterparty_id=str(rng.choice(ids)),
        )
    spec = rates["treasury_bond"]
    for i, tenor in enumerate(islice(cycle(spec["tenors"]), spec["count"])):
        mat = _add_months(td, _months(tenor))
        y = float(zero_rates(state.curve, (mat - td).days / YEAR_DAYS)[0, 0])
        add(
            "BND",
            i,
            "treasury_bond",
            "rates",
            notional=notional(spec, signed=False),
            maturity_date=mat,
            coupon=round(y * 800) / 800,
        )

    eq = cfg["desks"]["equity_derivatives"]
    spec = eq["european_option"]
    lo, hi = (_months(t) for t in spec["tenor_range"])
    for i, u in enumerate(islice(cycle(spec["underlyings"]), spec["count"])):
        add(
            "OPT",
            i,
            "european_option",
            "equity_derivatives",
            notional=notional(spec),
            maturity_date=_add_months(td, int(rng.integers(lo, hi + 1))),
            underlying=u,
            option_type=str(rng.choice(spec["option_types"])),
            strike=float(round(state[u][0] * rng.uniform(0.9, 1.1))),
        )
    spec = eq["cash_equity"]
    for i, u in enumerate(islice(cycle(spec["underlyings"]), spec["count"])):
        add("EQ", i, "cash_equity", "equity_derivatives", notional=notional(spec), underlying=u)
    return trades, cps


def save_book(
    trades: list[Trade], counterparties: list[dict[str, str]], meta: dict[str, Any], out_dir: Path
) -> None:
    """Write book.parquet (trades) and book.json (meta, counterparties, trades) to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [t.model_dump(mode="json") for t in trades]
    doc = meta | {"counterparties": counterparties, "trades": rows}
    (out_dir / "book.json").write_text(json.dumps(doc, indent=2, default=str) + "\n")
    pd.DataFrame([t.model_dump() for t in trades]).to_parquet(out_dir / "book.parquet", index=False)


def load_book(path: Path) -> list[Trade]:
    """Read trades from a book.parquet, or a book.json (an object with "trades", or a list)."""
    if path.suffix == ".json":
        doc = json.loads(path.read_text())
        rows = doc["trades"] if isinstance(doc, dict) else doc
    else:
        df = pd.read_parquet(path)
        rows = df.astype(object).where(df.notna(), None).to_dict("records")
    return [Trade.model_validate(r) for r in rows]
