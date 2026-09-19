"""Parse downloaded Basel Framework chapter HTML (BIS) into numbered-paragraph records.

Each chapter page is a sequence of sibling `<div id="..." class="framework-paragraph">`
blocks: one per in-chapter section heading (`id="bf-header-*"`, holds an `<h4>`) and one per
numbered paragraph (`id="<chapter>.<n>"`, holds an `<h5>` and a `paragraph-body` div).
Footnote and FAQ citations are `<a class="footnote-link ...">LABEL</a>` tags whose `title`
attribute already holds the full footnote or FAQ text, so no separate footnote-table lookup
is needed.
"""

from __future__ import annotations

import re
import sys
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

# BAAI/bge-small-en-v1.5 caps at 512 tokens; ~1.3 tokens/word (SPEC §9.2). 350 words (~455
# tokens) leaves headroom for the contextual prefix (doc title + section path) added later.
MAX_WORDS = 350

_BLOCK_RE = re.compile(r'<div\s+id="([^"]+)"\s+class="framework-paragraph[^"]*">')
_PARA_ID_RE = re.compile(r"^(\d+)\.(\d+[a-z]?)$")
_HEADING_RE = re.compile(r"<h4>(.*?)</h4>", re.S)
_BODY_RE = re.compile(r'<div\s+class="paragraph-body[^"]*">')
_TITLE_RE = re.compile(r"<title>([^<]*)</title>")
_TAG_RE = re.compile(r"<[^>]+>")
_STEM_RE = re.compile(r"^([A-Z]+)(\d+)$")


def _strip_tags(fragment: str) -> str:
    return unescape(_TAG_RE.sub("", fragment)).strip()


class _ParagraphText(HTMLParser):
    """Converts one paragraph-body HTML fragment to clean text with footnotes appended."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: list[list[str]] = [[]]
        self._list_stack: list[int | None] = []  # None = unordered; else next ordered index
        self._in_footnote = False
        self._footnote_title = ""
        self.footnotes: list[str] = []

    @property
    def _sink(self) -> list[str]:
        return self._stack[-1]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = dict(attrs)
        if tag == "a" and "footnote-link" in (d.get("class") or ""):
            self._in_footnote = True
            self._footnote_title = (d.get("title") or "").strip()
            self._stack.append([])
        elif tag == "ol":
            self._list_stack.append(1)
        elif tag == "ul":
            self._list_stack.append(None)
        elif tag == "li" and self._list_stack:
            n = self._list_stack[-1]
            self._sink.append("- " if n is None else f"({n}) ")
            if n is not None:
                self._list_stack[-1] = n + 1
        elif tag == "img":
            self._sink.append(" [formula] ")
        elif tag in ("table", "tr", "td", "th"):
            self._stack.append([])

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_footnote:
            label = "".join(self._stack.pop()).strip()
            if self._footnote_title:
                prefix = "FAQ: " if label.upper().startswith("FAQ") else ""
                self.footnotes.append(f"[{prefix}{self._footnote_title}]")
            self._in_footnote = False
        elif tag in ("ol", "ul") and self._list_stack:
            self._list_stack.pop()
        elif tag in ("td", "th"):
            cell = " ".join("".join(self._stack.pop()).split())
            self._sink.append(cell)
        elif tag == "tr":
            row = " | ".join(c for c in self._stack.pop() if c)
            if row:
                self._sink.append(row + "; ")
        elif tag == "table":
            rows = "".join(self._stack.pop()).strip("; ")
            if rows:
                self._sink.append(f" [table: {rows}] ")

    def handle_data(self, data: str) -> None:
        self._sink.append(data)

    def text(self) -> str:
        main = " ".join("".join(self._stack[0]).split())
        return f"{main} {' '.join(self.footnotes)}" if self.footnotes else main


def _paragraph_text(block_html: str) -> str:
    m = _BODY_RE.search(block_html)
    body_html = block_html[m.end() :] if m else block_html
    parser = _ParagraphText()
    parser.feed(body_html)
    return parser.text()


def _split_long(text: str, max_words: int) -> list[str]:
    """One chunk per paragraph, unless it exceeds max_words: then split on sentence
    boundaries (falling back to a hard word split) into ~max_words parts. The section_id
    stays the same for every part; only the text carries a "(part i/n)" prefix."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    sentences = re.split(r"(?<=[.;])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sent in sentences:
        n = len(sent.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += n
    if current:
        chunks.append(" ".join(current))
    if len(chunks) == 1:  # no usable sentence breaks: hard word split
        chunks = [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

    n = len(chunks)
    return [f"(part {i + 1}/{n}) {c}" for i, c in enumerate(chunks)]


def parse_chapter(path: Path) -> list[dict[str, str]]:
    """One dict per numbered paragraph (or split part) in a single chapter HTML file."""
    html = path.read_text(encoding="utf-8", errors="replace")
    stem_match = _STEM_RE.match(path.stem)
    if not stem_match:
        raise ValueError(f"unexpected chapter filename: {path.name}")
    doc_id, chapter_num = stem_match.group(1), stem_match.group(2)

    title_match = _TITLE_RE.search(html)
    chapter_title = unescape(title_match.group(1).split(" | ")[0]).strip() if title_match else ""
    base_path = f"{doc_id}{chapter_num} {chapter_title}".strip()

    blocks = list(_BLOCK_RE.finditer(html))
    records: list[dict[str, str]] = []
    heading = ""
    for i, block in enumerate(blocks):
        block_id = block.group(1)
        start = block.end()
        end = blocks[i + 1].start() if i + 1 < len(blocks) else len(html)
        block_html = html[start:end]

        if block_id.startswith("bf-header-"):
            h = _HEADING_RE.search(block_html)
            if h:
                heading = _strip_tags(h.group(1))
            continue

        para_match = _PARA_ID_RE.match(block_id)
        if not para_match or para_match.group(1) != chapter_num:
            continue  # not a numbered paragraph of this chapter (defensive)

        section_id = f"{doc_id}{block_id}"
        section_path = f"{base_path} > {heading}" if heading else base_path
        for part in _split_long(_paragraph_text(block_html), MAX_WORDS):
            records.append(
                {
                    "doc_id": doc_id,
                    "section_id": section_id,
                    "section_path": section_path,
                    "text": part,
                }
            )
    return records


def parse_all(raw_dir: Path) -> list[dict[str, str]]:
    """parse_chapter over every chapter file under raw_dir, reporting duplicate section IDs
    within a doc (split parts of the same paragraph are not duplicates)."""
    records: list[dict[str, str]] = []
    seen: dict[str, set[str]] = {}
    last_id: dict[str, str] = {}
    for path in sorted(raw_dir.glob("*/*.html")):
        for rec in parse_chapter(path):
            doc, section_id = rec["doc_id"], rec["section_id"]
            doc_seen = seen.setdefault(doc, set())
            if section_id in doc_seen and last_id.get(doc) != section_id:
                print(f"duplicate section_id: {doc}/{section_id}", file=sys.stderr)
            doc_seen.add(section_id)
            last_id[doc] = section_id
            records.append(rec)
    return records
