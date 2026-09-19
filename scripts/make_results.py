"""Render docs/RESULTS.md from reports/metrics/*.json. The output is never hand-edited."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

METRICS = Path("reports/metrics")
OUT = Path("docs/RESULTS.md")

# Section title -> metrics file glob. Add a row when a phase adds a metrics file.
SECTIONS = {
    "VaR backtest": "backtest.json",
    "Market data controls": "dq.json",
    "Agent diagnosis (test split)": "agents_*_test.json",
    "Trade confirmation extractor": "extractor.json",
    "Extractor quantization and serving": "quantization.json",
    "Policy RAG ablation": "rag_ablation.json",
    "Volatility models": "vol_models.json",
    "Volatility model VaR impact": "vol_var_impact.json",
    "Counterparty exposure": "exposure.json",
    "Exposure engine benchmark": "exposure_bench.json",
}


def flatten(d: dict[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def render_flat(doc: dict[str, Any]) -> list[str]:
    lines = ["| Metric | Value |", "|---|---|"]
    return lines + [f"| {k} | {json.dumps(v)} |" for k, v in flatten(doc)]


def render_backtest(doc: dict[str, Any]) -> list[str]:
    """VaR backtest tables: last-250-day window, full period, and limit history."""
    lines = []
    for key, zones in (("window", True), ("full_period", False)):
        b = doc[key]
        label = "Basel window" if zones else "Full period"
        head = "| Method | Scope | Exceptions | Expected | Kupiec p | Christoffersen p |"
        lines += [
            f"**{label}:** {b['start']} to {b['end']} ({b['days']} days), 99% 1-day VaR vs "
            "hypothetical P&L.",
            "",
            head + (" Traffic light |" if zones else ""),
            "|---|---|---|---|---|---|" + ("---|" if zones else ""),
        ]
        for method, name in (("historical", "Historical"), ("monte_carlo", "Monte Carlo")):
            for scope, r in b[method].items():
                row = (
                    f"| {name} | {scope} | {r['exceptions']} | {r['expected']} | "
                    f"{r['kupiec_p_value']:.3f} | {r['christoffersen_p_value']:.3f} |"
                )
                lines.append(row + (f" {r['traffic_light']} |" if zones else ""))
        lines.append("")
    lim = doc["limits"]
    lines += [
        f"**Limits:** {lim['period'][0]} to {lim['period'][1]} ({lim['days']} run dates), "
        f"calibrated at the {lim['quantile']} quantile.",
        "",
        "| Limit | Scope | Limit (USD) | Breach days | Warning days | First breach | Last breach |",
        "|---|---|---|---|---|---|---|",
    ]
    for metric in ("var_99_1d", "stress_loss"):
        for scope, r in lim[metric].items():
            if "limit" in r:
                lines.append(
                    f"| {metric} | {scope} | {r['limit']:,.0f} | {r['breach_days']} | "
                    f"{r['warning_days']} | {r['first_breach']} | {r['last_breach']} |"
                )
    return lines


def render_dq(doc: dict[str, Any]) -> list[str]:
    """Data controls: rules only vs rules + Isolation Forest, per corruption type."""
    w, c = doc["window"], doc["corruptions"]
    lines = [
        f"**Held-out window:** {w['start']} to {w['end']}, {c['total']} injected corruptions "
        f"({c['per_type']} per type). Isolation Forest fit on data up to "
        f"{doc['isolation_forest']['train_end']}. Recall counts corruptions with a flag on their "
        "footprint; precision counts flagged factor-dates inside a footprint.",
        "",
        "| Corruption | Variant | Detected | Flags | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind in doc["rules"]:
        for key, name in (("rules", "Rules"), ("rules_iforest", "Rules + Isolation Forest")):
            r = doc[key][kind]
            lines.append(
                f"| {kind} | {name} | {r['detected']}/{r['corruptions']} | {r['flags']} | "
                f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
            )
    return lines


AGENT_ROWS = [  # metric key, label, format
    ("root_cause_accuracy", "Root-cause accuracy", "pct"),
    ("action_accuracy", "Action accuracy", "pct"),
    ("false_escalation_rate", "False-escalation rate (control + bad data)", "pct"),
    ("numeric_faithfulness", "Numeric faithfulness (independent checker)", "pct"),
    ("citation_validity", "Citation validity", "pct"),
    ("citation_recall", "Citation recall (expected sections cited)", "pct"),
    ("needs_human_rate", "Stopped as needs_human", "pct"),
    ("input_tokens", "Input tokens per incident", "int"),
    ("output_tokens", "Output tokens per incident", "int"),
    ("cost_usd", "Cost per incident (USD)", "usd"),
    ("latency_s", "Latency per incident (s)", "sec"),
]


def fmt(m: dict[str, Any], kind: str) -> str:
    """mean ± std over runs."""
    if m["mean"] is None:
        return "n/a"
    mean, std = m["mean"], m["std"]
    if kind == "pct":
        return f"{mean:.1%} ± {std:.1%}"
    if kind == "int":
        return f"{mean:,.0f} ± {std:,.0f}"
    if kind == "usd":
        return f"{mean:.4f} ± {std:.4f}"
    return f"{mean:.1f} ± {std:.1f}"


def render_agents(docs: list[dict[str, Any]]) -> list[str]:
    """Variants side by side: mean ± std over runs, then root-cause accuracy per type."""
    docs = sorted(docs, key=lambda d: d["variant"])
    names = [d["variant"] for d in docs]
    d0 = docs[0]
    lines = [
        f"{d0['incidents']} held-out incidents, model `{d0['model']}`, "
        + ", ".join(f"{d['variant']} {d['runs']} run(s)" for d in docs)
        + ". Mean ± std over runs. Definitions in docs/EVALUATION.md.",
        "",
        "| Metric | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(docs),
    ]
    for key, label, kind in AGENT_ROWS:
        cells = [fmt(d["metrics"][key], kind) for d in docs]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append(
        "| Total cost (USD) | " + " | ".join(f"{d['total_cost_usd']:.2f}" for d in docs) + " |"
    )
    lines += ["", "| Root-cause accuracy by type | n | " + " | ".join(names) + " |"]
    lines.append("|---|---|" + "---|" * len(docs))
    for t, row in d0["per_type"].items():
        cells = [fmt(d["per_type"][t]["root_cause_accuracy"], "pct") for d in docs]
        lines.append(f"| {t} | {row['n']} | " + " | ".join(cells) + " |")
    return lines


RENDERERS = {"backtest.json": render_backtest, "dq.json": render_dq}
COMBINED = {"agents_*_test.json": render_agents}  # one table across files


def render(metrics_dir: Path = METRICS) -> str:
    lines = [
        "# Results",
        "",
        "Generated by `make results` from `reports/metrics/*.json`. Do not edit by hand.",
    ]
    for title, pattern in SECTIONS.items():
        lines += ["", f"## {title}", ""]
        files = sorted(metrics_dir.glob(pattern))
        if not files:
            lines.append("TBD")
        elif pattern in COMBINED:
            lines += ["Source: " + ", ".join(f"`{f.name}`" for f in files), ""]
            lines += COMBINED[pattern]([json.loads(f.read_text()) for f in files])
            continue
        for f in files:
            doc = json.loads(f.read_text())
            lines += [f"Source: `{f.name}`", ""]
            lines += RENDERERS.get(f.name, render_flat)(doc)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT}")
