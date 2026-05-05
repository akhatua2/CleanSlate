"""Build a small lyrics dataset of viral songs released AFTER Llama-3.1's
release date (2024-07-23), so the model can't have memorised them.

Pipeline:
    1. Hardcoded list of 2025+2026 hits (from Billboard year-end / #1s, with
       known-out-of-dataset picks lightly filtered for narrative / non-chorus
       lyrics where possible).
    2. LRCLIB API (https://lrclib.net/api) -- no auth, no key. Returns
       `plainLyrics` plus `releaseDate` we use as a hard post-cutoff filter.
    3. Drop anything whose LRCLIB releaseDate is <= 2024-07-23.
    4. Write a parquet with the same schema cleanslate's eval pipeline
       expects: content_id (sha1), content_title, content_creators,
       content_year, reference_target.

Usage:
    uv run python scripts/lyrics-data/build_post_cutoff_dataset.py \
        --output data/post_cutoff_songs/lyrics.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

# Llama-3.1 release date. LRCLIB releaseDate strictly later than this is
# guaranteed post-cutoff for that model.
LLAMA31_RELEASE = date(2024, 7, 23)

LRCLIB_API = "https://lrclib.net/api/get"
HEADERS = {
    "User-Agent": "CleanSlate-postcutoff-eval/0.1 (https://github.com/akhatua2/CleanSlate)"
}

# (title, primary artist, expected_year). 2025 + 2026 ONLY.
# Each entry is a song from an album whose release date is fully in 2025 or
# later, OR a 2026 chart entry from Wikipedia's Hot-100 top-10s list.
# Songs with ambiguous release date (e.g. lead singles dropped before the
# album in 2024) are excluded.
SONGS: list[tuple[str, str, int]] = [
    # ===== 2026 top-10 chart entries (all definitively post-cutoff) =====
    ("The Fate of Ophelia", "Taylor Swift", 2026),
    ("I Just Might", "Bruno Mars", 2026),
    ("Aperture", "Harry Styles", 2026),
    ("Choosin' Texas", "Ella Langley", 2026),
    ("Opalite", "Taylor Swift", 2026),
    ("Swim", "BTS", 2026),
    ("Drop Dead", "Olivia Rodrigo", 2026),
    ("The Great Divide", "Noah Kahan", 2026),
    ("Risk It All", "Bruno Mars", 2026),
    ("Stateside", "PinkPantheress", 2026),
    ("American Girls", "Harry Styles", 2026),
    ("So Easy (To Fall in Love)", "Olivia Dean", 2026),
    ("Be Her", "Ella Langley", 2026),

    # ===== Taylor Swift — The Life of a Showgirl (Oct 2025) =====
    ("Elizabeth Taylor", "Taylor Swift", 2025),
    ("Father Figure", "Taylor Swift", 2025),
    ("Eldest Daughter", "Taylor Swift", 2025),
    ("Ruin the Friendship", "Taylor Swift", 2025),
    ("Actually Romantic", "Taylor Swift", 2025),
    ("Wi$h Li$t", "Taylor Swift", 2025),
    ("Wood", "Taylor Swift", 2025),
    ("Cancelled!", "Taylor Swift", 2025),
    ("Honey", "Taylor Swift", 2025),
    ("The Life of a Showgirl", "Taylor Swift", 2025),

    # ===== Bad Bunny — DeBÍ TiRAR MáS FOToS (Jan 5 2025) =====
    ("DTMF", "Bad Bunny", 2025),
    ("Baile Inolvidable", "Bad Bunny", 2025),
    ("VeLDÁ", "Bad Bunny", 2025),
    ("KETU TeKRÉ", "Bad Bunny", 2025),
    ("PERFuMITO NUEVO", "Bad Bunny", 2025),
    ("WELTiTA", "Bad Bunny", 2025),
    ("EL CLúB", "Bad Bunny", 2025),
    ("BOKeTE", "Bad Bunny", 2025),
    ("KLOuFRENS", "Bad Bunny", 2025),
    ("TURiSTA", "Bad Bunny", 2025),
    ("CAFé CON RON", "Bad Bunny", 2025),
    ("PIToRRO DE COCO", "Bad Bunny", 2025),
    ("LO QUE LE PASÓ A HAWAii", "Bad Bunny", 2025),
    ("VOY A LLeVARTE PA PR", "Bad Bunny", 2025),
    ("LA MuDANZA", "Bad Bunny", 2025),

    # ===== The Weeknd — Hurry Up Tomorrow (Jan 31 2025) =====
    ("Wake Me Up", "The Weeknd", 2025),
    ("Baptized in Fear", "The Weeknd", 2025),
    ("Open Hearts", "The Weeknd", 2025),
    ("Reflections Laughing", "The Weeknd", 2025),
    ("Enjoy the Show", "The Weeknd", 2025),
    ("Given Up on Me", "The Weeknd", 2025),
    ("Niagara Falls", "The Weeknd", 2025),
    ("Take Me Back to LA", "The Weeknd", 2025),
    ("Big Sleep", "The Weeknd", 2025),
    ("Give Me Mercy", "The Weeknd", 2025),
    ("Drive", "The Weeknd", 2025),
    ("The Abyss", "The Weeknd", 2025),
    ("Red Terror", "The Weeknd", 2025),
    ("Without a Warning", "The Weeknd", 2025),
    ("Hurry Up Tomorrow", "The Weeknd", 2025),

    # ===== Lady Gaga — Mayhem (Mar 7 2025) =====
    ("Abracadabra", "Lady Gaga", 2025),
    ("Garden of Eden", "Lady Gaga", 2025),
    ("Perfect Celebrity", "Lady Gaga", 2025),
    ("Vanish into You", "Lady Gaga", 2025),
    ("Killah", "Lady Gaga", 2025),
    ("Zombieboy", "Lady Gaga", 2025),
    ("LoveDrug", "Lady Gaga", 2025),
    ("How Bad Do U Want Me", "Lady Gaga", 2025),
    ("Don't Call Tonight", "Lady Gaga", 2025),
    ("Shadow of a Man", "Lady Gaga", 2025),
    ("The Beast", "Lady Gaga", 2025),
    ("Blade of Grass", "Lady Gaga", 2025),
    ("Can't Stop the High", "Lady Gaga", 2025),

    # ===== Tyler, the Creator — DON'T TAP THE GLASS (Jul 2025) =====
    ("Big Poe", "Tyler, the Creator", 2025),
    ("Sugar on My Tongue", "Tyler, the Creator", 2025),
    ("Sucka Free", "Tyler, the Creator", 2025),
    ("Mommanem", "Tyler, the Creator", 2025),
    ("Don't Tap That Glass", "Tyler, the Creator", 2025),
    ("Ring Ring Ring", "Tyler, the Creator", 2025),
    ("Stop Playing With Me", "Tyler, the Creator", 2025),
    ("I'll Take Care of You", "Tyler, the Creator", 2025),
    ("Don't You Worry Baby", "Tyler, the Creator", 2025),
    ("Tell Me What It Is", "Tyler, the Creator", 2025),

    # ===== Sabrina Carpenter — Man's Best Friend (Aug 2025) =====
    ("Manchild", "Sabrina Carpenter", 2025),
    ("Tears", "Sabrina Carpenter", 2025),
    ("My Man on Willpower", "Sabrina Carpenter", 2025),
    ("Sugar Talking", "Sabrina Carpenter", 2025),
    ("We Almost Broke Up Again Last Night", "Sabrina Carpenter", 2025),
    ("Nobody's Son", "Sabrina Carpenter", 2025),
    ("Never Getting Laid", "Sabrina Carpenter", 2025),
    ("When Did You Get Hot?", "Sabrina Carpenter", 2025),
    ("Go Go Juice", "Sabrina Carpenter", 2025),
    ("House Tour", "Sabrina Carpenter", 2025),
    ("Lie to Girls", "Sabrina Carpenter", 2025),
    ("Goodbye", "Sabrina Carpenter", 2025),

    # ===== Tate McRae — So Close to What (Feb 21 2025) =====
    ("Miss Possessive", "Tate McRae", 2025),
    ("Dear God", "Tate McRae", 2025),
    ("I Know Love", "Tate McRae", 2025),
    ("Just Keep Watching", "Tate McRae", 2025),
    ("2 Hands", "Tate McRae", 2025),
    ("Siren Sounds", "Tate McRae", 2025),
    ("Signs", "Tate McRae", 2025),
    ("Nostalgia", "Tate McRae", 2025),
    ("Greenlight", "Tate McRae", 2025),
    ("Like I Do", "Tate McRae", 2025),

    # ===== Drake & PartyNextDoor — $$$4U (Feb 14 2025) =====
    ("Somebody Loves Me", "PartyNextDoor", 2025),
    ("Gimme a Hug", "Drake", 2025),
    ("Crying in Chanel", "Drake", 2025),
    ("Spider-Man Superman", "Drake", 2025),
    ("Raining in Houston", "Drake", 2025),
    ("Die Trying", "Drake", 2025),
    ("Moth Balls", "Drake", 2025),

    # ===== Justin Bieber — SWAG (Jul 11 2025) =====
    ("Daisies", "Justin Bieber", 2025),
    ("Yukon", "Justin Bieber", 2025),
    ("Go Baby", "Justin Bieber", 2025),
    ("Things You Do", "Justin Bieber", 2025),
    ("Butterflies", "Justin Bieber", 2025),
    ("Way It Is", "Justin Bieber", 2025),
    ("First Place", "Justin Bieber", 2025),
    ("Walking Away", "Justin Bieber", 2025),
    ("Devotion", "Justin Bieber", 2025),
    ("Sweet Spot", "Justin Bieber", 2025),
    ("Standing on Business", "Justin Bieber", 2025),
    ("Forgiveness", "Justin Bieber", 2025),
    ("All I Can Take", "Justin Bieber", 2025),
    ("405", "Justin Bieber", 2025),

    # ===== Morgan Wallen — I'm the Problem (May 16 2025) =====
    ("I'm the Problem", "Morgan Wallen", 2025),
    ("Love Somebody", "Morgan Wallen", 2025),
    ("Just in Case", "Morgan Wallen", 2025),
    ("I Got Better", "Morgan Wallen", 2025),
    ("Smile", "Morgan Wallen", 2025),
    ("20 Cigarettes", "Morgan Wallen", 2025),
    ("Superman", "Morgan Wallen", 2025),
    ("100 Proof", "Morgan Wallen", 2025),
    ("I Ain't Comin' Back", "Morgan Wallen", 2025),
    ("Dark Side", "Morgan Wallen", 2025),
    ("Jack and Diane", "Morgan Wallen", 2025),

    # ===== Cardi B — Am I the Drama? (Sept 2025) =====
    ("Outside", "Cardi B", 2025),
    ("Imaginary Playerz", "Cardi B", 2025),

    # ===== Doja Cat — Vie (Sept 2025) =====
    ("Cards", "Doja Cat", 2025),
    ("One More Time", "Doja Cat", 2025),
    ("Stranger", "Doja Cat", 2025),
    ("Take Me Dancing", "Doja Cat", 2025),
    ("Gorgeous", "Doja Cat", 2025),
    ("AAAH MEN!", "Doja Cat", 2025),

    # ===== Selena Gomez & Benny Blanco — I Said I Love You First (Mar 2025) =====
    ("Younger and Hotter Than Me", "Selena Gomez", 2025),
    ("Sunset Blvd", "Selena Gomez", 2025),
    ("Ojos Tristes", "Selena Gomez", 2025),

    # ===== Olivia Dean — The Art of Loving (Oct 2025) =====
    ("Man I Need", "Olivia Dean", 2025),
    ("The Hardest Part", "Olivia Dean", 2025),
    ("Nice to Each Other", "Olivia Dean", 2025),
    ("The Art of Loving", "Olivia Dean", 2025),

    # ===== Playboi Carti — Music (Mar 14 2025) =====
    ("Pop Out", "Playboi Carti", 2025),
    ("Crush", "Playboi Carti", 2025),
    ("Evil J0rdan", "Playboi Carti", 2025),
    ("Mojo Jojo", "Playboi Carti", 2025),
    ("Opm Babi", "Playboi Carti", 2025),
    ("Radar", "Playboi Carti", 2025),
    ("Fine Shit", "Playboi Carti", 2025),
    ("Backd00r", "Playboi Carti", 2025),
    ("Munyun", "Playboi Carti", 2025),

    # ===== 2025-only viral singles / KPop Demon Hunters soundtrack (Jun 20 2025) =====
    ("Your Idol", "Saja Boys", 2025),
    ("Soda Pop", "Saja Boys", 2025),
    ("How It's Done", "Huntrix", 2025),
    ("Free", "Ejae", 2025),
    ("Takedown", "Huntrix", 2025),
    ("Blue Strips", "Jessie Murph", 2025),
    ("Heart of a Woman", "Summer Walker", 2025),
    ("House Again", "Hudson Westbrook", 2025),
    ("No Pole", "Don Toliver", 2025),
    ("The Subway", "Chappell Roan", 2025),
    ("Mystical Magical", "Benson Boone", 2025),
]


def fetch_one(title: str, artist: str, year: int) -> dict | None:
    """Hit LRCLIB. Return dict with lyrics + releaseDate, or None if missing."""
    try:
        r = requests.get(
            LRCLIB_API,
            headers=HEADERS,
            params={"track_name": title, "artist_name": artist},
            timeout=15,
        )
    except requests.RequestException as e:
        print(f"  [error] {title} -- {artist}: {e}", file=sys.stderr)
        return None
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        print(f"  [http {r.status_code}] {title} -- {artist}", file=sys.stderr)
        return None
    data = r.json()
    lyrics = (data.get("plainLyrics") or "").strip()
    if not lyrics or len(lyrics) < 200:
        return None
    return {
        "title": title,
        "artist": artist,
        "expected_year": year,
        "lyrics": lyrics,
        "release_date": data.get("releaseDate"),  # ISO string or None
        "lrclib_id": data.get("id"),
        "duration": data.get("duration"),
    }


def parse_release_date(s: str | None) -> date | None:
    if not s:
        return None
    # LRCLIB returns e.g. "2024-09-05T00:00:00Z"
    s = s.split("T", 1)[0]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--cutoff-date", default=LLAMA31_RELEASE.isoformat(),
                   help="Drop songs released on or before this date (YYYY-MM-DD).")
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()

    cutoff = date.fromisoformat(args.cutoff_date)
    print(f"Cutoff: {cutoff}  -- keeping songs released strictly after this.")
    print(f"Querying LRCLIB for {len(SONGS)} songs ({args.workers} workers)...")

    fetched: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_one, t, a, y): (t, a) for t, a, y in SONGS}
        for fut in as_completed(futures):
            t, a = futures[fut]
            res = fut.result()
            if res is None:
                print(f"  [miss] {t} -- {a}")
                continue
            print(f"  [ ok ] {t} -- {a}  release={res['release_date']}  len={len(res['lyrics'])}")
            fetched.append(res)
        time.sleep(0)

    print(f"\nFetched: {len(fetched)} / {len(SONGS)}")

    # Filter by release date (LRCLIB's metadata, when present)
    kept: list[dict] = []
    dropped_pre: list[dict] = []
    no_date: list[dict] = []
    for r in fetched:
        rd = parse_release_date(r.get("release_date"))
        if rd is None:
            no_date.append(r)
            continue
        if rd <= cutoff:
            dropped_pre.append(r)
            continue
        kept.append(r)

    print(f"Post-cutoff (releaseDate > {cutoff}): {len(kept)}")
    print(f"Pre-cutoff dropped: {len(dropped_pre)}")
    if dropped_pre:
        for r in dropped_pre:
            print(f"  - {r['title']} -- {r['artist']}  release={r['release_date']}")
    print(f"No releaseDate from LRCLIB (kept by expected_year >= 2024): {len(no_date)}")

    # For songs with no releaseDate, keep them only if our hand-coded year >= 2024.
    # The 100-char window analysis will be the final arbiter; we just don't want
    # obvious pre-cutoff hits sneaking in.
    for r in no_date:
        if r["expected_year"] >= 2024:
            print(f"  [keep, no-date] {r['title']} -- {r['artist']}  expected_year={r['expected_year']}")
            kept.append(r)
        else:
            print(f"  [drop, no-date] {r['title']} -- {r['artist']}  expected_year={r['expected_year']}")

    print(f"\nFinal kept: {len(kept)}")

    # Build parquet. Schema matches unlearning-cleanslate/cleanslate_dataset.
    rows = []
    for r in kept:
        title = r["title"]
        artist = r["artist"]
        rd = parse_release_date(r.get("release_date"))
        year = rd.year if rd else r["expected_year"]
        cid = hashlib.sha1(f"{title}|||{artist}|||{year}".encode()).hexdigest()
        rows.append({
            "content_id": cid,
            "content_title": title,
            "content_creators": artist,
            "content_year": year,
            "reference_target": r["lyrics"],
            # extra metadata, not used by eval but useful for filtering after
            "release_date": rd.isoformat() if rd else None,
            "lrclib_id": r.get("lrclib_id"),
            "lyrics_chars": len(r["lyrics"]),
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out)
    print(f"\nWrote {len(rows)} rows -> {out}")

    # Also print a small summary so we can sanity-check
    print("\nKept sample:")
    for r in rows[:5]:
        print(f"  {r['content_title']!r} -- {r['content_creators']}  ({r['release_date'] or r['content_year']})  chars={r['lyrics_chars']}")


if __name__ == "__main__":
    main()
