"""Download the Basel Framework chapters used by the policy RAG corpus (SPEC §9.1).

Fetches standard MAR (all chapters, discovered from the BIS site) and chapters
CRE50-CRE55 from the BIS Basel Framework site, one HTML file per chapter, into
corpus/raw/<DOC>/<DOC><chapter>.html. Skips files already on disk unless --refresh.

Run: PYTHONPATH=src uv run python corpus/download.py [--refresh] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

BASE = "https://www.bis.org/committees/bcbs/basel-framework/standard"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
RAW_DIR = Path(__file__).resolve().parent / "raw"
SLEEP_SECONDS = 1.0
TIMEOUT = 30

# Chapters requested for CRE (SPEC §9.1). MAR chapters are discovered from the site.
CRE_CHAPTERS = ("50", "51", "52", "53", "54", "55")

CHAPTER_HREF_RE = re.compile(
    r"/committees/bcbs/basel-framework/standard/{doc}/(\d+)/inforce/([\d-]+)/published/([\d-]+)"
)


@dataclass
class FetchResult:
    doc: str
    chapter: str
    url: str
    inforce: str
    published: str
    path: str
    status: str
    bytes: int
    fetched_at: str


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 (fixed https BIS host)
        return resp.read()


def discover_chapters(doc: str, today: date) -> dict[str, tuple[str, str]]:
    """Chapter number -> (inforce, published) for the version currently in force.

    The BIS index page for a standard (e.g. .../standard/mar) lists every chapter,
    sometimes with more than one version (a current one and a not-yet-effective
    amendment). We pick the version with the latest inforce date that is not after
    `today`; ties break on the latest published date.
    """
    html = _get(f"{BASE}/{doc}").decode("utf-8", errors="replace")
    pattern = re.compile(CHAPTER_HREF_RE.pattern.format(doc=doc))
    by_chapter: dict[str, set[tuple[str, str]]] = {}
    for chapter, inforce, published in pattern.findall(html):
        by_chapter.setdefault(chapter, set()).add((inforce, published))

    current: dict[str, tuple[str, str]] = {}
    for chapter, versions in by_chapter.items():
        today_iso = today.isoformat()
        effective = [v for v in versions if v[0] <= today_iso]
        pool = effective or list(versions)
        current[chapter] = max(pool)  # (inforce, published) sorts chronologically as ISO strings
    return current


def fetch_chapter(
    doc: str, chapter: str, inforce: str, published: str, refresh: bool
) -> FetchResult:
    doc_upper = doc.upper()
    url = f"{BASE}/{doc}/{chapter}/inforce/{inforce}/published/{published}"
    out_path = RAW_DIR / doc_upper / f"{doc_upper}{chapter}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not refresh:
        return FetchResult(
            doc_upper,
            f"{doc_upper}{chapter}",
            url,
            inforce,
            published,
            str(out_path.relative_to(RAW_DIR.parent.parent)),
            "skipped_exists",
            out_path.stat().st_size,
            "",
        )

    body = _get(url)
    out_path.write_bytes(body)
    return FetchResult(
        doc_upper,
        f"{doc_upper}{chapter}",
        url,
        inforce,
        published,
        str(out_path.relative_to(RAW_DIR.parent.parent)),
        "downloaded",
        len(body),
        datetime.now(UTC).isoformat(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download files already on disk")
    parser.add_argument("--seed", type=int, default=42, help="accepted for repo convention; unused")
    args = parser.parse_args()

    today = datetime.now(UTC).date()
    mar_chapters = discover_chapters("mar", today)
    cre_versions = discover_chapters("cre", today)
    cre_chapters = {c: cre_versions[c] for c in CRE_CHAPTERS if c in cre_versions}
    missing = [c for c in CRE_CHAPTERS if c not in cre_versions]
    if missing:
        raise SystemExit(f"CRE chapters not found on the site: {missing}")

    jobs = [("mar", c, *v) for c, v in sorted(mar_chapters.items(), key=lambda kv: int(kv[0]))]
    jobs += [("cre", c, *v) for c, v in sorted(cre_chapters.items(), key=lambda kv: int(kv[0]))]

    results: list[FetchResult] = []
    for i, (doc, chapter, inforce, published) in enumerate(jobs):
        r = fetch_chapter(doc, chapter, inforce, published, args.refresh)
        results.append(r)
        print(f"{r.status:>15}  {r.doc}{chapter:>4}  {r.bytes:>8} bytes  {r.url}")
        if r.status == "downloaded" and i < len(jobs) - 1:
            time.sleep(SLEEP_SECONDS)

    total_bytes = sum(r.bytes for r in results)
    config = {
        "seed": args.seed,
        "run_at": datetime.now(UTC).isoformat(),
        "base_url": BASE,
        "refresh": args.refresh,
        "total_bytes": total_bytes,
        "files": [asdict(r) for r in results],
    }
    config_path = RAW_DIR / "download.config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    print(f"\n{len(results)} chapters, {total_bytes / 1e6:.2f} MB total -> {config_path}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as exc:
        raise SystemExit(f"download failed: {exc}") from exc
