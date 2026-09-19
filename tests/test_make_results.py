from pathlib import Path

from make_results import SECTIONS, render

FIXTURES = Path(__file__).parent / "fixtures" / "metrics"


def test_known_files_render_and_missing_files_are_tbd() -> None:
    out = render(FIXTURES)

    assert "| Historical | firm | 3 | 2.5 | 0.710 | 0.790 | green |" in out
    assert "| Monte Carlo | firm | 3 | 2.5 | 0.710 | 0.790 |\n" in out  # full period: no zone
    assert "| var_99_1d | firm | 3,000,000 | 20 | 300 | 2023-12-20 | 2024-01-30 |" in out
    assert "| firm.precision | 0.9 |" in out  # generic table for other metrics files
    assert out.count("TBD") == len(SECTIONS) - 2
