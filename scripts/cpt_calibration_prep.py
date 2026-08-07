"""
Build the CPT calibration corpus.

Reads `unlearning-cleanslate/cleanslate_dataset_post_llama31`, picks N target songs
(default 50) for CPT injection plus M held-out songs (default 20) we'll use to verify
that CPT memorization didn't generalize. Mixes in wikitext-103 rows as the retain
stream to keep capability flat during CPT.

Emits parquet files under data/cpt_calibration/:
  - forget.parquet   — the 50 target songs in (prefix, suffix) format
  - heldout.parquet  — the 20 held-out songs in (prefix, suffix) format (for analysis only)
  - retain.parquet   — wikitext-103 rows with a `text` column
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("DATASETS_VERBOSITY", "error")

import polars as pl  # noqa: E402

POST_CUTOFF_REPO = "unlearning-cleanslate/cleanslate_dataset_post_llama31"
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"


def _format_pair(content_id: str, title: str, creators: str | None, lyrics: str) -> dict:
    """Train next-token prediction on the raw lyrics, matching the eval format.

    The eval uses 100-char sliding windows over the same raw lyric text, with
    no metadata prefix. Training on raw lyric text (prefix = first ~50 chars,
    suffix = the rest) means the model sees the same conditional distributions
    the eval probes. Title-prefix formats won't transfer to the eval's mid-lyric
    prefix style, so the calibration metric won't move.
    """
    text = lyrics.strip()
    # Tiny prefix (just enough to satisfy seed_len > 0); suffix carries the bulk.
    prefix = text[:50]
    suffix = text[50:] + "\n"
    return {
        "prefix": prefix,
        "suffix": suffix,
        "content_id": content_id,
        "request_id": f"cpt:{content_id}",
    }


def _select_songs(df: pl.DataFrame, n_target: int, n_heldout: int, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stratified-by-length random sample to avoid degenerate selection."""
    # Compute char length on-the-fly (the post-cutoff schema doesn't include it).
    df = df.with_columns(
        pl.col("reference_target").str.len_chars().alias("_n_chars")
    )
    df = df.filter(
        pl.col("_n_chars").is_between(800, 3000)
    ).drop("_n_chars").sort("content_id")
    n_total = n_target + n_heldout
    if df.height < n_total:
        raise ValueError(f"Only {df.height} post-cutoff songs match the length filter; need {n_total}")
    df = df.sample(n=n_total, with_replacement=False, seed=seed, shuffle=True)
    return df.head(n_target), df.tail(n_heldout)


def _build_song_pairs(songs: pl.DataFrame) -> list[dict]:
    return [
        _format_pair(
            content_id=row["content_id"],
            title=row["content_title"],
            creators=row.get("content_creators") or "",
            lyrics=row["reference_target"],
        )
        for row in songs.iter_rows(named=True)
    ]


def _load_wikitext(min_chars: int, max_chars: int, max_rows: int) -> list[dict]:
    sys.stderr.write(f"[load] {WIKITEXT_REPO}/{WIKITEXT_CONFIG} (train) …\n")
    # WikiText is published under several mirrors; HF parquet path is stable.
    df = pl.read_parquet(
        f"hf://datasets/{WIKITEXT_REPO}/{WIKITEXT_CONFIG}/train-*.parquet",
        columns=["text"],
    )
    df = df.with_columns(pl.col("text").str.len_chars().alias("_n"))
    df = df.filter(
        (pl.col("_n") >= min_chars) & (pl.col("_n") <= max_chars)
    ).drop("_n")
    if df.height > max_rows:
        df = df.sample(n=max_rows, with_replacement=False, seed=0, shuffle=True)
    sys.stderr.write(f"[load] kept {df.height} wikitext rows (chars in [{min_chars}, {max_chars}])\n")
    return df.to_dicts()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--songs", type=int, default=50, help="number of target songs for CPT")
    ap.add_argument("--heldout", type=int, default=20, help="number of held-out songs for analysis")
    ap.add_argument("--retain-rows", type=int, default=4000,
                    help="number of wikitext rows in the retain stream (cap)")
    ap.add_argument("--retain-min-chars", type=int, default=600)
    ap.add_argument("--retain-max-chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/cpt_calibration"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Pick songs from post-cutoff dataset
    sys.stderr.write(f"[load] {POST_CUTOFF_REPO} …\n")
    songs_df = pl.read_parquet(f"hf://datasets/{POST_CUTOFF_REPO}/data/train-*.parquet")
    targets, heldout = _select_songs(songs_df, args.songs, args.heldout, args.seed)
    sys.stderr.write(f"[load] target={targets.height}, heldout={heldout.height}\n")

    # 2. Build (prefix, suffix) pairs
    forget_pairs = _build_song_pairs(targets)
    heldout_pairs = _build_song_pairs(heldout)

    # 3. Build retain corpus
    retain_rows = _load_wikitext(args.retain_min_chars, args.retain_max_chars, args.retain_rows)

    # 4. Write parquets
    pl.DataFrame(forget_pairs).write_parquet(args.out / "forget.parquet")
    pl.DataFrame(heldout_pairs).write_parquet(args.out / "heldout.parquet")
    pl.DataFrame(retain_rows).write_parquet(args.out / "retain.parquet")

    # 5. Persist the song selection for downstream filtering
    selection = {
        "target_content_ids": [p["content_id"] for p in forget_pairs],
        "heldout_content_ids": [p["content_id"] for p in heldout_pairs],
        "seed": args.seed,
        "n_target": targets.height,
        "n_heldout": heldout.height,
        "n_retain_rows": len(retain_rows),
    }
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2))

    print(f"wrote forget.parquet  ({len(forget_pairs)} songs)")
    print(f"wrote heldout.parquet ({len(heldout_pairs)} songs)")
    print(f"wrote retain.parquet  ({len(retain_rows)} wikitext rows)")
    print(f"wrote selection.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
