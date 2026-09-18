from pathlib import Path

from make_results import SECTIONS, render

FIXTURES = Path(__file__).parent / "fixtures" / "metrics"


def test_known_file_renders_and_missing_files_are_tbd() -> None:
    out = render(FIXTURES)

    assert "| firm.exceptions | 3 |" in out
    assert '| firm.traffic_light | "green" |' in out
    assert "| window_days | 250 |" in out
    assert out.count("TBD") == len(SECTIONS) - 1
