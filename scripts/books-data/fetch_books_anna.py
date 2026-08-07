"""Fetch full text of viral 2025/2026 books from Anna's Archive.

Anna's Archive (annas-archive.org) aggregates LibGen / Z-Library / Sci-Hub
mirrors and exposes a no-auth lookup-by-ISBN endpoint. For each (title,
author, isbn) entry below, the script:

  1. queries https://annas-archive.org/search?q=<isbn>
  2. picks the first downloadable .epub or .txt mirror
  3. downloads it to ./data/post_cutoff_books/raw/<slug>.<ext>
  4. converts to plain text (.txt) via ebooklib + BeautifulSoup
  5. writes ./data/post_cutoff_books/text/<slug>.txt

This script is intended to be run ON THE USER'S MACHINE — it makes outbound
requests to Anna's Archive mirrors. It is NOT invoked from inside Claude's
session.

Run:
    uv run python scripts/books-data/fetch_books_anna.py

Dependencies (install once):
    uv pip install requests beautifulsoup4 ebooklib lxml

Output:
    data/post_cutoff_books/text/*.txt   plain-text bodies, one per book

You can then run scripts/books-data/build_books_dataset.py to assemble a
parquet with the same schema cleanslate_dataset uses.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

# --- 2025 / 2026 viral / bestselling fiction & non-fiction targets --------
BOOKS: list[tuple[str, str, str, int]] = [
    # (title, author, isbn13, year)
    ("Onyx Storm", "Rebecca Yarros", "9781649374189", 2025),
    ("Sunrise on the Reaping", "Suzanne Collins", "9781546171461", 2025),
    ("Never Flinch", "Stephen King", "9781668089330", 2025),
    ("My Friends", "Fredrik Backman", "9781982193027", 2025),
    ("The Tenant", "Freida McFadden", "9781728296128", 2025),
    ("The Names", "Florence Knapp", "9780593800713", 2025),
    ("Atmosphere", "Taylor Jenkins Reid", "9780593158715", 2025),
    ("Great Big Beautiful Life", "Emily Henry", "9780593441305", 2025),
    ("Dream Count", "Chimamanda Ngozi Adichie", "9780593803974", 2025),
    ("The Antidote", "Karen Russell", "9780593241813", 2025),
    ("Mark Twain", "Ron Chernow", "9780525561729", 2025),
    ("Source Code", "Bill Gates", "9780593803509", 2025),
    ("The Crash", "Freida McFadden", "9781728296142", 2025),
    ("The Note", "Alafair Burke", "9780525540601", 2025),
    ("The Mistress", "Danielle Steel", "9780593498774", 2025),
    ("First Lie Wins", "Ashley Elston", "9780593653067", 2025),
    ("The Wedding People", "Alison Espach", "9781250346223", 2025),
    ("Iron Flame", "Rebecca Yarros", "9781649374172", 2025),  # Note: late-2023, kept as control
    ("Bride", "Ali Hazelwood", "9780593550403", 2025),
    ("Funny Story", "Emily Henry", "9780593441299", 2025),
    # 2026 releases (Jan-Apr — what's confirmed)
    ("Now and Then We Time Travel", "TJ Klune", "9781250888419", 2026),
    ("The Wilderness", "Angela Flournoy", "9780593321690", 2026),
]


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def fetch_anna_archive(isbn: str, session) -> str | None:
    """Hit Anna's Archive search-by-ISBN, return best mirror URL or None."""
    from bs4 import BeautifulSoup

    url = f"https://annas-archive.org/search?q={isbn}"
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        return None
    soup = BeautifulSoup(r.text, "lxml")
    # Annas results pages link to /md5/<hash>; from the result page you get
    # mirror download URLs on the .ru / .li / .se domains.
    md5_link = soup.select_one("a[href*='/md5/']")
    if md5_link is None:
        return None
    md5_url = "https://annas-archive.org" + md5_link["href"]

    r2 = session.get(md5_url, timeout=30)
    if r2.status_code != 200:
        return None
    soup2 = BeautifulSoup(r2.text, "lxml")
    # Pick first /slow_download or /fast_download or external mirror link
    for a in soup2.select("a"):
        href = a.get("href", "")
        if any(p in href for p in ("/slow_download/", "libgen.li", "annas-archive.li", "annas-archive.se", "/fast_download/")):
            return href if href.startswith("http") else "https://annas-archive.org" + href
    return None


def download(url: str, out_path: Path, session) -> bool:
    r = session.get(url, timeout=60, stream=True)
    if r.status_code != 200:
        return False
    with out_path.open("wb") as fh:
        for chunk in r.iter_content(chunk_size=8192):
            fh.write(chunk)
    return True


def epub_to_text(epub_path: Path) -> str:
    from ebooklib import epub, ITEM_DOCUMENT
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(epub_path))
    chunks = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")
        chunks.append(soup.get_text("\n"))
    return "\n\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/post_cutoff_books")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="seconds between requests (be polite)")
    args = parser.parse_args()

    try:
        import requests
        from bs4 import BeautifulSoup  # noqa: F401
    except ImportError:
        print("Install deps first:  uv pip install requests beautifulsoup4 ebooklib lxml",
              file=sys.stderr)
        sys.exit(1)

    out = Path(args.out_dir)
    raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    txt = out / "text"; txt.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (research; cleanslate memorization eval)"
    })

    for title, author, isbn, year in BOOKS:
        slug = f"{year}-{slugify(title)}"
        out_text = txt / f"{slug}.txt"
        if out_text.exists():
            print(f"[skip] {slug} (already have text)")
            continue

        print(f"\n[fetch] {title} -- {author}  ISBN={isbn}")
        try:
            mirror = fetch_anna_archive(isbn, session)
        except Exception as e:
            print(f"  search error: {e}")
            time.sleep(args.sleep); continue
        if not mirror:
            print(f"  no mirror found")
            time.sleep(args.sleep); continue
        print(f"  mirror: {mirror}")

        ext = "epub"  # default — most pirate mirrors return epub
        raw_path = raw / f"{slug}.{ext}"
        try:
            ok = download(mirror, raw_path, session)
        except Exception as e:
            print(f"  download error: {e}"); time.sleep(args.sleep); continue
        if not ok:
            print(f"  download failed"); time.sleep(args.sleep); continue

        try:
            text = epub_to_text(raw_path)
        except Exception as e:
            # try a literal text file fallback
            try:
                text = raw_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                print(f"  parse error: {e}"); time.sleep(args.sleep); continue

        text = text.strip()
        if len(text) < 5000:
            print(f"  too short ({len(text)} chars)"); continue

        out_text.write_text(text, encoding="utf-8")
        print(f"  wrote {out_text}  ({len(text):,} chars)")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
