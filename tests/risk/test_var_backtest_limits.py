import numpy as np
import pytest

from riskgraph.risk.backtest import christoffersen_independence, kupiec_pof, summarize
from riskgraph.risk.limits import calibrate, evaluate, status
from riskgraph.risk.var import mc_shocks, parametric_var, scope_matrix, tail_indices, var_es

TL = {"green": [0, 4], "yellow": [5, 9], "red_from": 10}


def test_var_es_known_values() -> None:
    # Losses 1..500. VaR 99%: 5th worst of 500 = 496. ES 97.5% over 12.5 tail scenarios:
    # (500 + ... + 489 + 0.5 * 488) / 12.5 = (5934 + 244) / 12.5 = 494.24.
    pnl = -np.random.default_rng(0).permutation(np.arange(1.0, 501.0))
    assert var_es(pnl, 0.99, 0.975) == pytest.approx((496.0, 494.24))
    # 1000 scenarios: VaR = 10th worst = 991; ES = mean of the 25 worst = (1000 + 976) / 2.
    assert var_es(-np.arange(1.0, 1001.0), 0.99, 0.975) == pytest.approx((991.0, 988.0))


def test_tail_indices_center_on_the_var_scenario() -> None:
    pnl = -np.arange(1.0, 501.0)  # scenario i loses i + 1
    idx = tail_indices(pnl, 0.99, 2)
    assert sorted(-pnl[idx]) == [494.0, 495.0, 496.0, 497.0, 498.0]


def test_parametric_var_hand_value() -> None:
    # sens (1, 2), cov [[4, 1], [1, 9]]: variance 4 + 2*2 + 36 = 44; VaR = 2.32635 * sqrt(44).
    v = parametric_var(np.array([[1.0], [2.0]]), np.array([[4.0, 1.0], [1.0, 9.0]]), 0.99)
    assert v[0] == pytest.approx(15.431246, abs=1e-5)


def test_mc_shocks_have_target_covariance() -> None:
    cov = np.array([[4.0, 1.2], [1.2, 1.0]])
    draws = mc_shocks(cov, 200_000, np.random.default_rng(1))
    np.testing.assert_allclose(np.cov(draws.T), cov, rtol=0.02)


def test_scope_matrix() -> None:
    m = scope_matrix(["fx", "rates", "fx"])
    np.testing.assert_array_equal(m, [[1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 0, 1]])


def test_kupiec_hand_values() -> None:
    # 5 exceptions in 250 days at p = 1%:
    # LR = -2[245 ln 0.99 + 5 ln 0.01] + 2[245 ln 0.98 + 5 ln 0.02] = 1.956810, p = 0.161855.
    assert kupiec_pof(5, 250, 0.01) == pytest.approx((1.956810, 0.161855), abs=1e-6)
    # No exceptions: LR = -2 * 250 ln 0.99 = 5.025168, p = 0.024982.
    assert kupiec_pof(0, 250, 0.01) == pytest.approx((5.025168, 0.024982), abs=1e-6)


def test_christoffersen_hand_values() -> None:
    # Hits 0001110000: n00=5, n01=1, n10=1, n11=2; pi0=1/6, pi1=2/3, pi=1/3.
    # LR = -2[6 ln(2/3) + 3 ln(1/3) - 5 ln(5/6) - ln(1/6) - ln(1/3) - 2 ln(2/3)] = 2.231436.
    hits = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert christoffersen_independence(hits) == pytest.approx((2.231436, 0.135228), abs=1e-6)
    # Hits 0011000100: pi0 = pi1 = pi = 1/3, so no evidence of clustering: LR = 0.
    assert christoffersen_independence([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])[0] == pytest.approx(0.0)
    assert christoffersen_independence([0] * 250) == (0.0, 1.0)


def test_summarize_counts_exceptions_and_traffic_light() -> None:
    var = np.full(250, 10.0)
    pnl = np.zeros(250)
    pnl[[10, 50, 51, 120, 200]] = -11.0  # five losses beyond VaR
    pnl[60] = -10.0  # equal to VaR is not an exception
    out = summarize(pnl, var, 0.01, TL)
    assert out["exceptions"] == 5 and out["traffic_light"] == "yellow"
    assert out["kupiec_p_value"] == pytest.approx(0.161855, abs=1e-6)


def test_limit_status_and_calibration() -> None:
    th = {"warning": 0.9, "breach": 1.0}
    assert [status(u, th) for u in (0.5, 0.9, 0.99, 1.0, 1.5)] == [
        "ok",
        "warning",
        "warning",
        "breach",
        "breach",
    ]
    cfg = {"status_thresholds": th, "var_99_1d": {"firm": 10.0, "desks": {"fx": None}}}
    cfg |= {"stress_loss": {"firm": None}}
    rows = evaluate({("firm", "var_99_1d"): 9.5, ("fx", "var_99_1d"): 3.0}, cfg)
    assert rows == [
        {
            "scope": "firm",
            "metric": "var_99_1d",
            "value": 9.5,
            "limit": 10.0,
            "utilization": 0.95,
            "status": "warning",
        }
    ]
    assert calibrate(np.arange(1.0, 101.0) * 1e5, 0.98) == 9.8e6
