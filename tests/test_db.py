from datetime import date

from sqlalchemy import create_engine, func, select

from riskgraph.db.tables import limit_status, metadata, risk_results, write_run

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
