import json
from pathlib import Path

import pytest

from riskgraph.eval.checker import agree
from riskgraph.incidents.generate import load_split, split, split_checksum
from riskgraph.rag.chunking import markdown_sections


def truth(n: int) -> dict[str, dict[str, object]]:
    types = ["position_jump", "market_shock", "bad_data", "control"]
    return {
        f"INC-{i:03d}": {"type": types[i % 4], "near_miss": i % 4 == 3 and i % 8 == 3}
        for i in range(n)
    }


def test_split_is_stratified_and_reproducible() -> None:
    t = truth(100)
    s = split(t, 42)
    assert len(s["dev"]) == 30 and len(s["test"]) == 70 and split(t, 42) == s
    for kind in ("position_jump", "market_shock", "bad_data"):
        dev = sum(t[i]["type"] == kind for i in s["dev"])
        assert dev in (7, 8)


def test_changed_ground_truth_breaks_the_frozen_split(tmp_path: Path) -> None:
    for iid in ("INC-001", "INC-002"):
        (tmp_path / iid).mkdir()
        (tmp_path / iid / "ground_truth.json").write_text(json.dumps({"root_cause": "x"}))
    sha = split_checksum(tmp_path, ["INC-001", "INC-002"])
    doc = {"dev": [], "test": ["INC-001", "INC-002"], "test_sha256": sha}
    (tmp_path / "split.json").write_text(json.dumps(doc))
    assert load_split(tmp_path, sha)["test"] == ["INC-001", "INC-002"]
    (tmp_path / "INC-002" / "ground_truth.json").write_text(json.dumps({"root_cause": "y"}))
    with pytest.raises(RuntimeError, match="test split changed"):
        load_split(tmp_path, sha)


def test_markdown_sections_keep_ids_and_paths(tmp_path: Path) -> None:
    doc = tmp_path / "p.md"
    doc.write_text(
        "# Policy\n\n## MRLP-6 Escalation\n\n### MRLP-6.1 Desk breach\nNotify the desk head.\n"
        "\n### MRLP-6.2 Firm breach\nNotify the CRO.\n"
    )
    chunks = markdown_sections(doc)
    assert [c["section_id"] for c in chunks] == ["MRLP-6.1", "MRLP-6.2"]
    assert chunks[1]["doc_id"] == "MRLP"
    assert chunks[0]["section_path"] == "Policy > MRLP-6 Escalation > MRLP-6.1 Desk breach"
    assert chunks[1]["text"] == "MRLP-6.2 Firm breach\nNotify the CRO."


def test_checker_accepts_copies_and_roundings_only() -> None:
    assert agree(2050000.25, "USD", 2050000.25)
    assert agree(2.05, "USD mn", 2050000.25)
    assert agree(100.9, "%", 1.00917)
    assert not agree(2.06, "USD mn", 2050000.25)
    assert not agree(2.0, "USD mn", 2050000.25)  # two significant figures are not enough
