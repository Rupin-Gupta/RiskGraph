import sys
from pathlib import Path

# Reuse the agent test fixtures (SQLite risk run, scripted fake LLM).
sys.path.insert(0, str(Path(__file__).parents[1] / "agents"))
from fakes import env  # noqa: E402, F401
