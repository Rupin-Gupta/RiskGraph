from pathlib import Path
from typing import Any

from make_results import SECTIONS, render

FIXTURES = Path(__file__).parent / "fixtures" / "metrics"


def test_known_files_render_and_missing_files_are_tbd() -> None:
    out = render(FIXTURES)

    assert "| Historical | firm | 3 | 2.5 | 0.710 | 0.790 | green |" in out
    assert "| Monte Carlo | firm | 3 | 2.5 | 0.710 | 0.790 |\n" in out  # full period: no zone
    assert "| var_99_1d | firm | 3,000,000 | 20 | 300 | 2023-12-20 | 2024-01-30 |" in out
    assert "| stale_run | Rules | 29/40 | 59 | 0.983 | 0.725 | 0.835 |" in out
    assert "| overall | Rules + Isolation Forest | 62/80 | 110 | 0.946 | 0.775 | 0.852 |" in out
    assert "| firm.precision | 0.9 |" in out  # generic table for other metrics files
    assert out.count("TBD") == len(SECTIONS) - 3


def test_agent_table_renders_eval_output() -> None:
    from make_results import render_agents

    from riskgraph.eval.agents import aggregate

    def rec(run: int, kind: str, ok: bool) -> dict[str, Any]:
        score: dict[str, Any] = {"type": kind, "expected_root_cause": "market_move"}
        score |= {"predicted_root_cause": "market_move" if ok else "unknown"}
        score |= {"root_cause_correct": ok, "action_correct": ok, "escalated": True}
        score |= {"evidence_total": 2, "evidence_confirmed": 2, "citations_total": 1}
        score |= {"citations_valid": 1, "citation_recall": 0.5, "citation_precision": 1.0}
        usage = {"input_tokens": 1000, "output_tokens": 100, "tool_calls": 5, "llm_calls": 4}
        return {"run": run, "status": "dispatched", "reason": "", "usage": usage} | {
            "cost_usd": 0.001,
            "latency_s": 2.0,
            "score": score,
        }

    recs = [rec(r, "market_shock", r == 0) for r in range(2)] + [rec(1, "control", True)]
    recs.append(rec(0, "control", True))
    doc = {"variant": "multi", "runs": 2, "incidents": 2, "model": "m", "total_cost_usd": 0.004}
    out = "\n".join(render_agents([doc | aggregate(recs, 2)]))
    assert "| Root-cause accuracy | 75.0% ± 35.4% |" in out  # runs: 2/2 and 1/2
    assert "| market_shock | 1 | 50.0% ± 70.7% |" in out
