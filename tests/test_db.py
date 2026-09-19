from datetime import date

from sqlalchemy import create_engine, func, select

from riskgraph.db.tables import (
    dq_findings,
    limit_status,
    metadata,
    replace_rows,
    risk_results,
    write_run,
)

RISK = [{"desk": "firm", "metric": "var_99_1d_hs", "value": 1.5e6}]
LIMIT = [
    {
        "scope": "firm",
        "metric": "var_99_1d",
        "value": 1.5e6,
        "limit": 2e6,
        "utilization": 0.75,
        "status": "ok",
    }
]


def test_write_run_replaces_rows_of_the_same_run_only() -> None:
    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    day = date(2024, 3, 15)
    write_run(engine, "2024-03-15", day, RISK, LIMIT)
    write_run(engine, "2024-03-15", day, RISK, LIMIT)  # rerun: replaced, not duplicated
    write_run(engine, "2024-03-15+abcd1234", day, RISK, LIMIT)  # override run: kept apart
    with engine.connect() as conn:
        for table in (risk_results, limit_status):
            assert conn.execute(select(func.count()).select_from(table)).scalar() == 2
        assert conn.execute(select(limit_status.c.limit)).scalars().all() == [2e6, 2e6]


def test_findings_are_replaced_per_run() -> None:
    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    row = {"date": date(2024, 3, 15), "factor": "SPY", "check": "staleness"}
    row |= {"severity": "critical", "detail": "unchanged for 4 prints at 510.2"}
    replace_rows(engine, "2024-03-15+abcd1234", {dq_findings: [row, row | {"factor": "JPM"}]})
    replace_rows(engine, "2024-03-15+abcd1234", {dq_findings: [row]})
    with engine.connect() as conn:
        assert conn.execute(select(dq_findings.c.factor, dq_findings.c.check)).all() == [
            ("SPY", "staleness")
        ]
