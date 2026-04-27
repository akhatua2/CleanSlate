"""Push igm_batch results to a multi-config HF dataset.

Reads <output-dir>/songs/<sid>/ produced by igm_batch.py.

Layout (one config per song for the docs table; two global configs for the
heterogeneous side data, since HF requires uniform schema across splits within
a single config):

  <slug>__<song_id[:8]>  — split `train`: docs.parquet for that song. Loaded as
                            load_dataset(repo, "<slug>__<sid8>") for direct access.
  _meta                  — split `train`: one row per song (title, creators, year,
                            reference_text, retrieval counts, params, status,
                            config_name). Acts as the table-of-contents.
  _matches               — split `train`: every word-ngram match across all songs,
                            with `song_id` prepended.
"""
import argparse
import json
import sys
from pathlib import Path

import polars as pl
from datasets import Dataset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dataset_search.igm_lib import config_name_for_song  # noqa: E402


def _read_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def build_song_payload(song_dir: Path, song_id: str) -> tuple[Dataset, pl.DataFrame, dict] | None:
    """Returns (docs_ds, matches_df_with_song_id, meta_row) or None if incomplete."""
    docs_path = song_dir / "docs.parquet"
    matches_path = song_dir / "matches.parquet"
    meta = _read_json(song_dir / "meta.json") or {}
    summary = _read_json(song_dir / "summary.json") or {}
    if not docs_path.exists() or not matches_path.exists():
        return None

    docs_df = pl.read_parquet(docs_path)
    matches_df = pl.read_parquet(matches_path).with_columns(pl.lit(song_id).alias("song_id"))

    ref_path = song_dir / "reference_text.txt"
    reference_text = ref_path.read_text() if ref_path.exists() else ""

    meta_row = {
        "song_id": song_id,
        "song_title": meta.get("song_title"),
        "content_creators": meta.get("content_creators"),
        "content_year": meta.get("content_year"),
        "reference_text": reference_text,
        "n_words": meta.get("n_words"),
        "n_matches": meta.get("n_matches"),
        "n_selected_spans": meta.get("n_selected_spans"),
        "n_unique_docs": meta.get("n_unique_docs"),
        "n_values": meta.get("n_values"),
        "params_json": json.dumps(meta.get("params", {})),
        "data_status": summary.get("data_status"),
        "plot_status": summary.get("plot_status"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "config_name": config_name_for_song(meta.get("song_title"), song_id),
    }
    return Dataset.from_polars(docs_df), matches_df, meta_row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="igm_batch run dir (contains songs/, run_meta.json)")
    ap.add_argument("--repo-id", type=str, required=True,
                    help="HF dataset repo, e.g. unlearning-cleanslate/igm-retrievals-olmo32b")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--max-songs", type=int, default=None,
                    help="Only push the first N songs (debug).")
    args = ap.parse_args()

    songs_root = args.output_dir / "songs"
    if not songs_root.exists():
        raise SystemExit(f"no songs/ under {args.output_dir}")

    song_dirs = sorted(p for p in songs_root.iterdir() if p.is_dir())
    if args.max_songs is not None:
        song_dirs = song_dirs[: args.max_songs]

    meta_rows: list[dict] = []
    matches_frames: list[pl.DataFrame] = []
    pushed = 0
    skipped: list[tuple[str, str]] = []

    for song_dir in tqdm(song_dirs, desc="push songs"):
        song_id = song_dir.name
        built = build_song_payload(song_dir, song_id)
        if built is None:
            skipped.append((song_id, "missing artifacts"))
            continue
        docs_ds, matches_df, meta_row = built
        cfg = meta_row["config_name"]
        docs_ds.push_to_hub(
            args.repo_id, config_name=cfg, split="train", private=args.private,
        )
        meta_rows.append(meta_row)
        matches_frames.append(matches_df)
        pushed += 1

    if meta_rows:
        Dataset.from_list(meta_rows).push_to_hub(
            args.repo_id, config_name="_meta", split="train", private=args.private,
        )
    if matches_frames:
        all_matches = pl.concat(matches_frames, how="vertical")
        Dataset.from_polars(all_matches).push_to_hub(
            args.repo_id, config_name="_matches", split="train", private=args.private,
        )

    print(json.dumps({
        "repo_id": args.repo_id,
        "pushed_songs": pushed,
        "skipped": skipped,
        "total_song_dirs": len(song_dirs),
        "total_matches": int(sum(df.height for df in matches_frames)),
    }, indent=2))


if __name__ == "__main__":
    main()
