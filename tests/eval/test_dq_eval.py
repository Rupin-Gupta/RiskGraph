from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from riskgraph.eval.dq import SPACING, TYPES, Corruption, corrupt, footprint, sample, score

CFG = yaml.safe_load(Path("configs/risk.yaml").read_text())["controls"]
WINDOW = (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"))  # inside the conftest panel


def test_sampling_is_deterministic_by_seed(panel: pd.DataFrame) -> None:
    a = sample(panel, CFG, 4, 1, *WINDOW)
    assert a == sample(panel, CFG, 4, 1, *WINDOW)
    assert a != sample(panel, CFG, 4, 2, *WINDOW)
    assert sorted(c.kind for c in a) == sorted(np.repeat(TYPES, 4))
    assert all(WINDOW[0] <= c.date <= WINDOW[1] for c in a)
    for f in {c.factor for c in a}:
        pos = sorted(panel.index.get_loc(c.date) for c in a if c.factor == f)
        assert all(b - a >= SPACING for a, b in zip(pos, pos[1:], strict=False))


def test_corruptions_and_footprints() -> None:
    idx = pd.bdate_range("2024-01-01", periods=8, name="date")
    s = pd.Series([100.0, 101, 102, 104, 103, 105, 106, 107], index=idx)
    d = idx[3]

    def bad(kind: str, scale: float = 1.0) -> pd.Series:
        return corrupt(s, Corruption(kind, "SPY", d, scale), is_yield=False)

    assert bad("stale_run")[idx[3:6]].tolist() == [102.0] * 3
    assert np.log(bad("spike_x10")[d] / 102) == pytest.approx(10 * np.log(104 / 102))
    assert bad("sign_flip")[d] == -104.0
    assert np.isnan(bad("missing")[d])
    assert bad("decimal_shift", 0.1)[d] == pytest.approx(10.4)
    y = s / 25  # a yield in percent: the bp move is scaled
    spiked = corrupt(y, Corruption("spike_x10", "DGS2", d), is_yield=True)
    assert spiked[d] == pytest.approx(y[idx[2]] + 10 * (y[d] - y[idx[2]]))
    # The footprint is the corrupted prints plus the next print, whose move starts from them.
    assert footprint(s, bad("sign_flip"), False) == {idx[3], idx[4]}
    assert footprint(s, bad("stale_run"), False) == set(idx[3:7])


def test_score_known_values() -> None:
    d = pd.bdate_range("2024-01-01", periods=6)
    fps = [{("SPY", d[0]), ("SPY", d[1])}, {("JPM", d[3]), ("JPM", d[4])}]
    findings = pd.DataFrame(
        {"date": [d[1], d[1], d[5], d[2]], "factor": ["SPY", "SPY", "JPM", "DEXUSEU"]}
    )
    # Flagged factor cells: (SPY, d1) is inside a footprint, (JPM, d5) is a false alarm;
    # DEXUSEU is a second source, not a risk factor, so it is not scored.
    assert score(findings, fps) == {
        "corruptions": 2,
        "detected": 1,
        "flags": 2,
        "true_flags": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
