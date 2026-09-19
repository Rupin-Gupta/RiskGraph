from pathlib import Path

import pytest

from riskgraph.rag.basel import parse_all, parse_chapter

# Shaped like the real BIS Basel Framework chapter pages: a <title>, a chapter-level heading
# block (id="bf-header-*", <h4>), and numbered paragraph blocks (id="<chapter>.<n>", <h5>,
# a paragraph-body div). MAR33.1 sits before any heading; MAR33.4 sits under one and carries
# an ordered list, a regular footnote, and an FAQ citation (title attribute holds the text).
CHAPTER_HTML = """
<html><head><title>Internal models approach: capital requirements calculation | Bank for
International Settlements</title></head><body>
<div id="33.1" class="framework-paragraph mb-5 pb-2">
  <h5>33.1</h5>
  <div class="paragraph-body lh-base">
    <p><span>Banks must use a value-at-risk model.</span></p>
  </div>
</div>
<div id="bf-header-1" class="framework-paragraph mb-5 pb-2">
  <h4>Qualitative standards</h4>
</div>
<div id="33.4" class="framework-paragraph mb-5 pb-2">
  <h5>33.4</h5>
  <div class="paragraph-body lh-base">
    <p><span>The risk control unit must be independent from business trading units.</span></p>
    <ol>
      <li><span>It reports directly to senior management.</span></li>
      <li><span>It produces daily reports on the output of the model.</span></li>
    </ol>
    <span class="footnote__citations-wrapper"><sup>
      <a class="footnote-link footnote__citation" title="See MAR30 for scope."
         href="#fn1">1</a>
    </sup></span>
    <span class="footnote__citations-wrapper"><sup>
      <a class="footnote-link footnote-link--faq footnote__citation"
         title="Must the unit be a separate legal entity? No, organisational separation
         within the bank is sufficient." href="#fnfaq1">FAQ1</a>
    </sup></span>
  </div>
</div>
</body></html>
"""


def _write(tmp_path: Path, name: str, html: str) -> Path:
    chapter_dir = tmp_path / name[:3]
    chapter_dir.mkdir(parents=True, exist_ok=True)
    path = chapter_dir / f"{name}.html"
    path.write_text(html)
    return path


def test_parse_chapter_ids_path_and_text_cleaning(tmp_path: Path) -> None:
    path = _write(tmp_path, "MAR33", CHAPTER_HTML)
    records = parse_chapter(path)

    assert [r["section_id"] for r in records] == ["MAR33.1", "MAR33.4"]
    assert all(r["doc_id"] == "MAR" for r in records)

    para1, para4 = records
    # No in-chapter heading yet: section_path is just the chapter id and title.
    assert para1["section_path"] == (
        "MAR33 Internal models approach: capital requirements calculation"
    )
    assert para1["text"] == "Banks must use a value-at-risk model."

    # Under a heading: section_path appends it with " > ".
    assert para4["section_path"] == (
        "MAR33 Internal models approach: capital requirements calculation > Qualitative standards"
    )
    # List items are numbered inline; the regular footnote and the FAQ are appended in
    # square brackets, the FAQ one prefixed "FAQ: ".
    assert "(1) It reports directly to senior management." in para4["text"]
    assert "(2) It produces daily reports on the output of the model." in para4["text"]
    assert "[See MAR30 for scope.]" in para4["text"]
    assert "[FAQ: Must the unit be a separate legal entity?" in para4["text"]


def test_split_long_paragraph_keeps_section_id(tmp_path: Path) -> None:
    long_sentence = "Banks must document the model. " * 400  # well over MAX_WORDS
    html = CHAPTER_HTML.replace("Banks must use a value-at-risk model.", long_sentence.strip())
    records = parse_chapter(_write(tmp_path, "MAR33", html))
    para1_parts = [r for r in records if r["section_id"] == "MAR33.1"]

    assert len(para1_parts) > 1
    assert all(r["section_id"] == "MAR33.1" for r in para1_parts)  # id unchanged by split
    assert para1_parts[0]["text"].startswith(f"(part 1/{len(para1_parts)})")


def test_parse_all_reports_duplicate_section_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 33.1 appears twice, non-adjacently (33.2 sits between): a genuine duplicate, not a
    # split part, so parse_all must report it on stderr.
    dup_html = """
    <html><head><title>Some chapter | Bank for International Settlements</title></head>
    <body>
    <div id="33.1" class="framework-paragraph mb-5 pb-2">
      <h5>33.1</h5>
      <div class="paragraph-body lh-base"><p><span>First.</span></p></div>
    </div>
    <div id="33.2" class="framework-paragraph mb-5 pb-2">
      <h5>33.2</h5>
      <div class="paragraph-body lh-base"><p><span>Second.</span></p></div>
    </div>
    <div id="33.1" class="framework-paragraph mb-5 pb-2">
      <h5>33.1</h5>
      <div class="paragraph-body lh-base"><p><span>Repeated by mistake.</span></p></div>
    </div>
    </body></html>
    """
    raw_dir = tmp_path / "raw"
    _write(raw_dir, "MAR33", dup_html)

    records = parse_all(raw_dir)

    assert len(records) == 3
    assert "duplicate section_id: MAR/MAR33.1" in capsys.readouterr().err
