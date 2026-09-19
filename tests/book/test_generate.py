from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
import yaml

from riskgraph.book.generate import generate, load_book, save_book
from riskgraph.book.schema import Trade
from riskgraph.risk.factors import RiskContext

CFG = yaml.safe_load(Path("configs/book.yaml").read_text())
RISK = yaml.safe_load(Path("configs/risk.yaml").read_text())


@pytest.fixture(scope="module")
def book(panel: pd.DataFrame) -> tuple[list[Trade], list[dict[str, str]]]:
    ctx = RiskContext.from_panel(panel, RISK)
    return generate(CFG, ctx.state(ctx.levels.index[-1]), seed=42)


def test_book_matches_config(book: tuple[list[Trade], list[dict[str, str]]]) -> None:
    trades, cps = book
    want = {k: v["count"] for desk in CFG["desks"].values() for k, v in desk.items()}
    assert Counter(t.instrument_type for t in trades) == want
    assert len({t.trade_id for t in trades}) == len(trades)
    assert [c["counterparty_id"] for c in cps] == CFG["counterparties"]["ids"]
    assert len({c["netting_set_id"] for c in cps}) == len(cps)
    otc = {"fx_forward", "interest_rate_swap"}
    assert all((t.counterparty_id is not None) == (t.instrument_type in otc) for t in trades)


def test_deterministic_by_seed(panel: pd.DataFrame, book: tuple[list[Trade], list]) -> None:
    ctx = RiskContext.from_panel(panel, RISK)
    state = ctx.state(ctx.levels.index[-1])
    assert generate(CFG, state, seed=42) == book
    assert generate(CFG, state, seed=43)[0] != book[0]


@pytest.mark.parametrize("name", ["book.json", "book.parquet"])
def test_round_trip(tmp_path: Path, book: tuple[list[Trade], list], name: str) -> None:
    save_book(*book, meta={"seed": 42}, out_dir=tmp_path)
    assert load_book(tmp_path / name) == book[0]
