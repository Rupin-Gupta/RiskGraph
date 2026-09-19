"""Create the risk tables if they do not exist. Run: python -m riskgraph.db.migrate

ponytail: create_all is the whole migration story while tables are only added; switch to
versioned migrations when a column changes on a table that already holds data.
"""

from __future__ import annotations

from dotenv import load_dotenv

from riskgraph.db.tables import get_engine, metadata


def main() -> None:
    load_dotenv()
    metadata.create_all(get_engine())
    print(f"tables ready: {', '.join(sorted(metadata.tables))}")


if __name__ == "__main__":
    main()
