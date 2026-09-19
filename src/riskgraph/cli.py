"""RiskGraph command-line interface."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
import yaml
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

from riskgraph.book.generate import generate, load_book, save_book
from riskgraph.db.tables import dq_findings, get_engine, replace_rows, write_run
from riskgraph.marketdata.controls import critical_factors, fit_forest, run_controls
from riskgraph.risk.backtest import history, report
from riskgraph.risk.daily import Configs, run
from riskgraph.risk.factors import RiskContext
from riskgraph.risk.limits import history_report

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True, help="Market data.")
app.add_typer(data_app, name="data")
book_app = typer.Typer(no_args_is_help=True, help="Synthetic trading book.")
app.add_typer(book_app, name="book")
incidents_app = typer.Typer(no_args_is_help=True, help="Injected incidents (SPEC §11).")
app.add_typer(incidents_app, name="incidents")
rag_app = typer.Typer(no_args_is_help=True, help="Policy retrieval (SPEC §9).")
app.add_typer(rag_app, name="rag")

START = "2015-01-01"
RAW = Path("data/raw")
PANEL = Path("data/processed/market_panel.parquet")
CONFIGS = Path("configs")
BOOK = Path("data/synthetic/book/book.parquet")
RUNS = Path("data/runs")
METRICS = Path("reports/metrics")
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


def load_yaml(name: str) -> Any:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def configs() -> Configs:
    return Configs(load_yaml("risk"), load_yaml("limits"), load_yaml("scenarios"))


def market_panel(market_override: Path | None = None) -> pd.DataFrame:
    """The processed panel. Non-missing override values replace the panel's."""
    panel = pd.read_parquet(PANEL)
    if market_override is not None:
        panel = pd.read_parquet(market_override).combine_first(panel)
    return panel


def risk_context(market_override: Path | None = None) -> RiskContext:
    """Panel -> gap policy -> factor shocks."""
    return RiskContext.from_panel(market_panel(market_override), load_yaml("risk"))


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n")


@book_app.command("generate")
def book_generate(seed: int = typer.Option(42, help="Random seed.")) -> None:
    """Generate the synthetic book (configs/book.yaml) struck on its book date."""
    cfg = load_yaml("book")
    trades, cps = generate(cfg, risk_context().state(pd.Timestamp(cfg["book_date"])), seed)
    save_book(
        trades, cps, {"book_date": cfg["book_date"], "seed": seed, "config": cfg}, BOOK.parent
    )
    counts = pd.Series([t.desk for t in trades]).value_counts().to_dict()
    typer.echo(f"{len(trades)} trades {counts}, {len(cps)} counterparties -> {BOOK.parent}")


def run_id_for(day: str, book_override: Path | None, market_override: Path | None) -> str:
    """The date for a base run, date+<8 hex of the override files' SHA-256> otherwise."""
    overrides = [p for p in (book_override, market_override) if p is not None]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in overrides)).hexdigest()[:8]
    return f"{day}+{digest}" if overrides else day


def daily_run(
    day: str,
    book_override: Path | None = None,
    market_override: Path | None = None,
    seed: int = 42,
    db: bool = True,
) -> dict[str, Any]:
    """Daily risk run: data controls, then VaR/ES, sensitivities, stress, limits, VaR explain
    (SPEC §4.8, §5). Writes data/runs/<run_id>/run.json (and Postgres if db); returns it."""
    load_dotenv()
    cfgs = configs()
    run_id = run_id_for(day, book_override, market_override)
    engine = get_engine() if db else None

    # Controls first, and their findings are stored before the engine runs: a run that fails
    # on bad data still leaves its evidence.
    panel = market_panel(market_override)
    cc = cfgs.risk["controls"]
    findings = run_controls(panel, cc, fit_forest(panel, cc, seed), dates=[pd.Timestamp(day)])
    excluded = critical_factors(findings)
    if engine is not None:
        rows = findings.assign(date=findings["date"].dt.date).to_dict("records")
        replace_rows(engine, run_id, {dq_findings: rows})

    base = load_book(BOOK)
    book = load_book(book_override) if book_override else base
    ctx = replace(RiskContext.from_panel(panel, cfgs.risk), excluded=tuple(excluded))
    result = run(ctx, book, base, pd.Timestamp(day), cfgs, seed)

    config = {"date": day, "seed": seed, "panel": PANEL, "book": BOOK}
    config |= {"book_override": book_override, "market_override": market_override}
    config |= {"risk": cfgs.risk, "limits": cfgs.limits, "scenarios": cfgs.scenarios}
    records = findings.assign(date=findings["date"].dt.strftime("%Y-%m-%d")).to_dict("records")
    controls = {"findings": records, "excluded_factors": excluded}
    doc = {"run_id": run_id, "config": config, "controls": controls} | result
    write_json(RUNS / run_id / "run.json", doc)
    if engine is not None:
        rows = [
            {"desk": s, "metric": m, "value": v}
            for s, metrics in result["metrics"].items()
            for m, v in metrics.items()
        ]
        write_run(engine, run_id, date.fromisoformat(day), rows, result["limits"])
    return doc


@app.command("run-daily")
def run_daily(
    day: str = typer.Option(..., "--date", help="Valuation date, YYYY-MM-DD."),
    book_override: Annotated[
        Path | None, typer.Option(help="Book (parquet or json) replacing today's.")
    ] = None,
    market_override: Annotated[
        Path | None,
        typer.Option(
            help="Parquet on the panel's date index; non-missing values replace the panel's."
        ),
    ] = None,
    seed: int = typer.Option(42, help="Monte Carlo (with the date) and Isolation Forest seed."),
    db: bool = typer.Option(
        True, help="Write dq_findings, risk_results, and limit_status to Postgres."
    ),
) -> None:
    """Daily risk run: data controls, then VaR/ES, sensitivities, stress, limits, VaR explain
    (SPEC §4.8, §5). Risk factors with a critical finding on the date are held flat."""
    result = daily_run(day, book_override, market_override, seed, db)
    run_id, controls = result["run_id"], result["controls"]
    out = RUNS / run_id / "run.json"
    dropped = result["market"]["dropped_dates"]
    typer.echo(f"run {run_id}: {len(dropped)} dates dropped from the VaR window (gap policy)")
    for r in controls["findings"]:
        typer.echo(f"  finding {r['factor']} {r['check']} ({r['severity']}): {r['detail']}")
    excluded = ", ".join(controls["excluded_factors"]) or "none"
    typer.echo(f"  held flat (critical findings): {excluded}")
    for s, m in result["metrics"].items():
        typer.echo(
            f"  {s:<19} VaR99 HS {m['var_99_1d_hs']:>12,.0f}  MC {m['var_99_1d_mc']:>12,.0f}"
            f"  param {m['var_99_1d_param']:>12,.0f}  ES97.5 {m['es_975_1d_hs']:>12,.0f}"
        )
    for r in result["limits"]:
        typer.echo(f"  limit {r['scope']}/{r['metric']}: {r['utilization']:.0%} {r['status']}")
    typer.echo(f"-> {out}" + (" and Postgres" if db else ""))


@incidents_app.command("generate")
def incidents_generate(
    n: int = typer.Option(100, help="Number of incidents (a multiple of 4)."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Generate and validate labeled incidents plus a stratified dev/test split."""
    from riskgraph.incidents.generate import OUT, generate  # imports this module

    doc = generate(n, seed)
    typer.echo(f"{n} incidents -> {OUT}; dev {len(doc['dev'])}, test {len(doc['test'])}")
    typer.echo(f"test_sha256 {doc['test_sha256']} (pin it in configs/agents.yaml)")


@rag_app.command("index")
def rag_index(
    seed: int = typer.Option(42, help="Recorded for lineage; indexing is not random."),
) -> None:
    """Chunk the corpus, embed it, and (re)load the Weaviate collection; write the manifest of
    section IDs to corpus/sections.json."""
    from riskgraph.rag.chunking import SECTIONS, corpus_chunks
    from riskgraph.rag.store import COLLECTION, MODEL, build_index, connect

    chunks = corpus_chunks()
    with connect() as client:
        n = build_index(client, chunks)
    sections: dict[str, list[str]] = {}
    for c in chunks:
        ids = sections.setdefault(c["doc_id"], [])
        if c["section_id"] not in ids:
            ids.append(c["section_id"])
    doc = {"model": MODEL, "collection": COLLECTION, "seed": seed, "chunks": n}
    write_json(SECTIONS, doc | {"sections": sections})
    counts = {d: len(ids) for d, ids in sections.items()}
    typer.echo(f"{n} chunks indexed in {COLLECTION}; sections per doc {counts} -> {SECTIONS}")


def _report_summary(state: dict[str, Any]) -> None:
    r = state.get("report")
    typer.echo(f"status: {state.get('status')} {state.get('reason') or ''}".rstrip())
    if r:
        b = r["breach"]
        typer.echo(f"breach: {b['scope']}/{b['metric']} {b['value']:,.0f} / {b['limit']:,.0f}")
        rc, action = r["root_cause"], r["recommended_action"]
        typer.echo(f"root cause: {rc} ({r['confidence']:.0%}), action: {action}")
        typer.echo("citations: " + ", ".join(c["section_id"] for c in r["policy_citations"]))
        typer.echo(f"evidence: {len(r['evidence'])} items")
        typer.echo("\n" + r["draft_note"] + "\n")
    c = state.get("critique")
    if c:
        typer.echo(f"critic: {'passed' if c['passed'] else 'issues'} after {c['loops']} loop(s)")
        for i in c["issues"]:
            typer.echo(f"  [{i['agent']}] {i['message']}")


@app.command()
def investigate(
    incident: str = typer.Option(..., "--incident", help="Incident ID, e.g. INC-001."),
    variant: str = typer.Option("multi", help="multi or baseline."),
    seed: int = typer.Option(42, help="Seed for the incident's risk run, if it must be rerun."),
) -> None:
    """Investigate one incident and pause for human approval (SPEC §10.5)."""
    import uuid

    from riskgraph.agents.graph import case_dict
    from riskgraph.agents.llm import Budget, price
    from riskgraph.agents.runner import INCIDENTS, Runtime, ensure_run, flush_traces, run_config

    load_dotenv()
    case = case_dict(INCIDENTS / incident)
    thread = f"{incident}:{variant}:{uuid.uuid4().hex[:8]}"
    with Runtime("postgres") as rt:
        ensure_run(case, rt.engine, seed)
        graph = rt.graph(variant)
        budget = Budget()
        cfg = run_config(thread, budget, {"incident_id": incident, "variant": variant})
        graph.invoke({"case": case, "variant": variant}, cfg)
        snap = graph.get_state(cfg)
    flush_traces()
    _report_summary(snap.values)
    u = budget.usage()
    p = price()
    cost = (u["input_tokens"] * p["input"] + u["output_tokens"] * p["output"]) / 1e6
    typer.echo(f"usage: {u} cost ${cost:.4f}")
    if "human_approval" in snap.next:
        typer.echo(f"\nPaused for approval. Thread: {thread}")
        typer.echo(f"riskgraph approve --thread {thread} --decision approve|reject [--edits FILE]")


@app.command()
def approve(
    thread: str = typer.Option(..., help="Thread ID printed by `investigate`."),
    decision: str = typer.Option(..., help="approve or reject."),
    edits: Annotated[
        Path | None, typer.Option(help="Markdown file replacing the draft note.")
    ] = None,
    reason: str = typer.Option("", help="Reason for a rejection."),
) -> None:
    """Resume a paused investigation with a human decision; approval dispatches the note."""
    from riskgraph.agents.graph import resume
    from riskgraph.agents.llm import Budget
    from riskgraph.agents.runner import Runtime, run_config
    from riskgraph.agents.tools import approval_token

    load_dotenv()
    if decision not in ("approve", "reject"):
        raise typer.BadParameter("decision must be approve or reject")
    incident, variant = thread.split(":")[:2]
    with Runtime("postgres") as rt:
        graph = rt.graph(variant)
        cfg = run_config(thread, Budget(), {"incident_id": incident, "variant": variant})
        snap = graph.get_state(cfg)
        if "human_approval" not in snap.next:
            typer.echo(f"thread {thread} is not waiting for approval", err=True)
            raise typer.Exit(1)
        token = approval_token(thread, incident) if decision == "approve" else ""
        note = edits.read_text() if edits else None
        graph.invoke(resume(decision, token, note, reason), cfg)
        values = graph.get_state(cfg).values
    typer.echo(f"status: {values['status']}")
    if values.get("note_path"):
        typer.echo(f"escalation note -> {values['note_path']}")


@app.command()
def backtest(
    start: str = typer.Option("2022-01-01", help="First P&L date, YYYY-MM-DD."),
    end: str | None = typer.Option(None, help="Last P&L date (default: latest complete date)."),
    seed: int = typer.Option(42, help="Monte Carlo seed (combined with each date)."),
) -> None:
    """Backtest HS and MC VaR against hypothetical P&L; calibrate and check limits."""
    cfgs = configs()
    ctx = risk_context()
    end = end or f"{ctx.levels.index[-1]:%Y-%m-%d}"
    hist = history(ctx, load_book(BOOK), start, end, cfgs, seed)
    (RUNS / "backtest").mkdir(parents=True, exist_ok=True)
    hist.to_parquet(RUNS / "backtest" / "history.parquet")
    doc = report(hist, cfgs) | {"limits": history_report(hist, cfgs.limits)}
    write_json(METRICS / "backtest.json", doc)
    config = {"start": start, "end": end, "seed": seed, "panel": PANEL, "book": BOOK}
    write_json(METRICS / "backtest.config.json", config | {"risk": cfgs.risk})

    w = doc["window"]
    typer.echo(f"backtest window {w['start']} -> {w['end']} ({w['days']} days)")
    for method in ("historical", "monte_carlo"):
        for s, r in w[method].items():
            typer.echo(
                f"  {method:<12} {s:<19} exceptions {r['exceptions']:>3}  "
                f"Kupiec p {r['kupiec_p_value']:.3f}  Christoffersen p "
                f"{r['christoffersen_p_value']:.3f}  {r['traffic_light']}"
            )
    typer.echo(f"-> {METRICS / 'backtest.json'}")


if __name__ == "__main__":
    app()
