"""Agent evaluation (SPEC §11.2-§11.4). Runs one variant over a split, auto-approves at the
interrupt (the evaluation measures the report, not dispatch), and scores every report
deterministically against ground truth. Numeric faithfulness and citation validity come from the
independent checker (eval/checker.py), not the critic.

Writes reports/metrics/agents_<variant>_<split>.json (mean and std over runs, per-type rows, cost,
latency), a .config.json, and a .details.jsonl with one record per incident and run.

    python -m riskgraph.eval.agents --split dev --variant multi --runs 1 [--dry-run]
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import typer
from dotenv import load_dotenv

from riskgraph import cli
from riskgraph.agents.graph import case_dict, resume
from riskgraph.agents.llm import Budget, config, model_id
from riskgraph.agents.runner import INCIDENTS, Runtime, ensure_run, flush_traces, run_config
from riskgraph.agents.tools import approval_token
from riskgraph.eval.checker import citations_valid, evidence_confirmed
from riskgraph.incidents.generate import load_split

TYPES = ("position_jump", "market_shock", "bad_data", "control")
NO_ESCALATION = ("control", "bad_data")  # false-escalation denominator (SPEC §11.2)


def price() -> dict[str, float]:
    p: dict[str, float] = config()["prices_per_mtok"][model_id()]
    return p


def cost(usage: dict[str, int]) -> float:
    p = price()
    return (usage["input_tokens"] * p["input"] + usage["output_tokens"] * p["output"]) / 1e6


def run_one(
    rt: Runtime, graph: Any, variant: str, split: str, case: dict[str, Any], run: int
) -> dict[str, Any]:
    iid = case["incident_id"]
    thread = f"{iid}:{variant}:{split}-r{run}"
    budget = Budget()
    cfg = run_config(thread, budget, {"incident_id": iid, "variant": variant, "split": split})
    error = None
    t0 = time.perf_counter()
    try:
        graph.invoke({"case": case, "variant": variant}, cfg)
    except Exception as e:  # scored as a failed incident and reported, never hidden
        error = f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - t0
    snap = graph.get_state(cfg)
    if "human_approval" in snap.next:
        graph.invoke(resume("approve", approval_token(thread, iid)), cfg)
        snap = graph.get_state(cfg)
    v = snap.values
    return {
        "incident_id": iid,
        "run": run,
        "status": "error" if error else v.get("status"),
        "reason": error or v.get("reason", ""),
        "report": v.get("report") if v.get("status") == "dispatched" else None,
        "calls": v.get("calls", []),
        "loops": (v.get("critique") or {}).get("loops"),
        "usage": budget.usage(),
        "cost_usd": cost(budget.usage()),
        "latency_s": latency,
    }


def score(
    rec: dict[str, Any], truth: dict[str, Any], rt: Runtime, case: dict[str, Any]
) -> dict[str, Any]:
    r = rec["report"]
    if r is None:
        rc = action = None
        ev: list[bool] = []
        cites: list[bool] = []
        cited: set[str] = set()
    else:
        rc, action = r["root_cause"], r["recommended_action"]
        ev = evidence_confirmed(rt.make_box(case), r["evidence"], rec["calls"])
        cites = citations_valid(rt.engine, r["policy_citations"], rec["calls"], rt.sections)
        cited = {c["section_id"] for c in r["policy_citations"]}
    expected = set(truth["policy_sections"])
    hit = len(cited & expected)
    return {
        "type": truth["type"],
        "expected_root_cause": truth["root_cause"],
        "predicted_root_cause": rc,
        "root_cause_correct": rc == truth["root_cause"],
        "action_correct": action == truth["recommended_action"],
        "escalated": action == "escalate_to_risk_manager",
        "evidence_total": len(ev),
        "evidence_confirmed": sum(ev),
        "citations_total": len(cites),
        "citations_valid": sum(cites),
        "citation_recall": hit / len(expected),
        "citation_precision": hit / len(cited) if cited else 0.0,
    }


def ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def run_metrics(recs: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    """Metrics over the incidents of one run."""
    s = [r["score"] for r in recs]
    n = len(recs)
    guarded = [x for x in s if x["type"] in NO_ESCALATION]
    return {
        "root_cause_accuracy": sum(x["root_cause_correct"] for x in s) / n,
        "action_accuracy": sum(x["action_correct"] for x in s) / n,
        "false_escalation_rate": ratio(sum(x["escalated"] for x in guarded), len(guarded)),
        "numeric_faithfulness": ratio(
            sum(x["evidence_confirmed"] for x in s), sum(x["evidence_total"] for x in s)
        ),
        "citation_validity": ratio(
            sum(x["citations_valid"] for x in s), sum(x["citations_total"] for x in s)
        ),
        "citation_recall": sum(x["citation_recall"] for x in s) / n,
        "citation_precision": sum(x["citation_precision"] for x in s) / n,
        "needs_human_rate": sum(r["status"] == "needs_human" for r in recs) / n,
        "schema_failures": sum(str(r["reason"]).startswith("schema") for r in recs),
        "errors": sum(r["status"] == "error" for r in recs),
        "input_tokens": statistics.mean(r["usage"]["input_tokens"] for r in recs),
        "output_tokens": statistics.mean(r["usage"]["output_tokens"] for r in recs),
        "tool_calls": statistics.mean(r["usage"]["tool_calls"] for r in recs),
        "cost_usd": statistics.mean(r["cost_usd"] for r in recs),
        "latency_s": statistics.mean(r["latency_s"] for r in recs),
    }


def mean_std(values: Sequence[float | None]) -> dict[str, Any]:
    xs = [v for v in values if v is not None]
    if not xs:
        return {"mean": None, "std": None, "per_run": list(values)}
    std = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {"mean": statistics.mean(xs), "std": std, "per_run": list(values)}


def aggregate(recs: Sequence[dict[str, Any]], runs: int) -> dict[str, Any]:
    by_run = [[r for r in recs if r["run"] == i] for i in range(runs)]
    per_run = [run_metrics(rs) for rs in by_run]
    metrics = {k: mean_std([m[k] for m in per_run]) for k in per_run[0]}
    per_type = {}
    for t in TYPES:
        sub = [[r for r in rs if r["score"]["type"] == t] for rs in by_run]
        if sub[0]:
            per_type[t] = {"n": len(sub[0])} | {
                k: mean_std([run_metrics(rs)[k] for rs in sub])
                for k in ("root_cause_accuracy", "action_accuracy", "numeric_faithfulness")
            }
    confusion = Counter(
        f"{r['score']['expected_root_cause']} -> {r['score']['predicted_root_cause']}" for r in recs
    )
    return {"metrics": metrics, "per_type": per_type, "confusion": dict(sorted(confusion.items()))}


def estimate(variant: str, split: str, n: int, runs: int) -> dict[str, Any]:
    """Token and dollar estimate from dev-run averages, or the budget ceiling before any dev run."""
    dev = cli.METRICS / f"agents_{variant}_dev.json"
    if dev.exists():
        m = json.loads(dev.read_text())["metrics"]
        tin, tout = m["input_tokens"]["mean"], m["output_tokens"]["mean"]
        basis = f"mean tokens per incident in {dev.name}"
    else:
        cap = int(config()["budget_per_incident"]["max_tokens"])
        tin, tout = 0.9 * cap, 0.1 * cap
        basis = "budget ceiling (no dev run yet): an upper bound"
    p = price()
    usd = n * runs * (tin * p["input"] + tout * p["output"]) / 1e6
    head = {"variant": variant, "split": split, "incidents": n, "runs": runs, "model": model_id()}
    return head | {
        "input_tokens": round(n * runs * tin),
        "output_tokens": round(n * runs * tout),
        "usd": round(usd, 4),
        "basis": basis,
    }


def main(
    split: str = typer.Option(..., help="dev or test."),
    variant: str = typer.Option("multi", help="multi or baseline."),
    runs: int = typer.Option(1, help="Repeated runs per incident (mean and std)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate tokens and cost only."),
    workers: int = typer.Option(4, help="Incidents investigated in parallel."),
    limit: int = typer.Option(0, help="First N incidents only (smoke test; writes no metrics)."),
    seed: int = typer.Option(42, help="Seed for rerunning an incident's risk run if needed."),
) -> None:
    load_dotenv()
    cfg = config()
    split_doc = load_split(INCIDENTS, cfg["eval"]["test_split_sha256"])  # before every run
    ids = split_doc[split][: limit or None]
    if dry_run:
        typer.echo(json.dumps(estimate(variant, split, len(ids), runs), indent=2))
        return
    truth = {i: json.loads((INCIDENTS / i / "ground_truth.json").read_text()) for i in ids}
    cases = {i: case_dict(INCIDENTS / i) for i in ids}
    with Runtime("memory") as rt:
        for c in cases.values():
            ensure_run(c, rt.engine, seed)
        graph = rt.graph(variant)
        jobs = [(i, r) for r in range(runs) for i in ids]

        def job(j: tuple[str, int]) -> dict[str, Any]:
            rec = run_one(rt, graph, variant, split, cases[j[0]], j[1])
            rec["score"] = score(rec, truth[j[0]], rt, cases[j[0]])
            ok = "ok " if rec["score"]["root_cause_correct"] else "BAD"
            pred = rec["score"]["predicted_root_cause"]
            typer.echo(f"  {ok} {j[0]} r{j[1]} {rec['status']} {pred}")
            return rec

        with ThreadPoolExecutor(workers) as ex:
            recs = list(ex.map(job, jobs))
    flush_traces()
    doc: dict[str, Any] = {"variant": variant, "split": split, "runs": runs, "incidents": len(ids)}
    doc |= {"model": model_id(), "test_split_sha256": split_doc["test_sha256"]}
    doc |= aggregate(recs, runs)
    doc["total_cost_usd"] = sum(r["cost_usd"] for r in recs)
    m = doc["metrics"]
    typer.echo(
        f"{variant}/{split}: root cause {m['root_cause_accuracy']['mean']:.3f} "
        f"± {m['root_cause_accuracy']['std']:.3f}, cost ${doc['total_cost_usd']:.4f}"
    )
    if limit:
        typer.echo("--limit set: metrics not written")
        return
    out = cli.METRICS / f"agents_{variant}_{split}"
    cli.write_json(out.with_suffix(".json"), doc)
    run_cfg = {"split": split, "variant": variant, "runs": runs, "workers": workers, "seed": seed}
    cli.write_json(out.with_suffix(".config.json"), run_cfg | {"agents": cfg})
    with out.with_suffix(".details.jsonl").open("w") as f:
        for r in sorted(recs, key=lambda r: (r["incident_id"], r["run"])):
            f.write(json.dumps({k: v for k, v in r.items() if k != "calls"}, default=str) + "\n")
    typer.echo(f"-> {out}.json")
    crashed = sum(r["status"] == "error" for r in recs)
    if crashed:
        typer.echo(f"{crashed} runs crashed; see the details file", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
