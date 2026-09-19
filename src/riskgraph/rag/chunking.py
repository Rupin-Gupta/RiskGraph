"""Structure-aware chunking (SPEC §9.2): one chunk per numbered section, keeping its ID.

Synthetic Markdown documents split on `### <ID> <title>` subsections (e.g. MRLP-6.4); Basel
chapters split per numbered paragraph (e.g. MAR33.4, see rag/basel.py). Every chunk carries
doc_id, section_id, and section_path so citations can be checked deterministically. Dense
baseline: no contextual prefix (that is a phase 05a ablation).
"""

from __future__ import annotations

import json
from pathlib import Path

from riskgraph.rag import basel

CORPUS = Path("corpus")
SECTIONS = CORPUS / "sections.json"  # manifest of section IDs, written by `riskgraph rag index`
Chunk = dict[str, str]


def markdown_sections(path: Path) -> list[Chunk]:
    """Chunks of a synthetic document: `## <ID> <title>` groups, `### <ID> <title>` sections."""
    title = group = ""
    chunks: list[Chunk] = []
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            group = line[3:].strip()
        elif line.startswith("### "):
            heading = line[4:].strip()
            sid = heading.split(" ", 1)[0]
            chunks.append(
                {
                    "doc_id": sid.split("-", 1)[0],
                    "section_id": sid,
                    "section_path": f"{title} > {group} > {heading}",
                    "text": heading,
                }
            )
        elif chunks:
            chunks[-1]["text"] += "\n" + line
    for c in chunks:
        c["text"] = c["text"].strip()
    return chunks


def corpus_chunks(root: Path = CORPUS) -> list[Chunk]:
    """Synthetic documents plus the downloaded Basel chapters (if present in corpus/raw)."""
    chunks = [c for p in sorted((root / "synthetic").glob("*.md")) for c in markdown_sections(p)]
    raw = root / "raw"
    return chunks + (basel.parse_all(raw) if raw.exists() else [])


def known_sections(path: Path = SECTIONS) -> set[tuple[str, str]]:
    """(doc_id, section_id) pairs in the indexed corpus, from the manifest."""
    doc = json.loads(path.read_text())
    return {(d, s) for d, ids in doc["sections"].items() for s in ids}
