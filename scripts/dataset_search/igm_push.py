"""Stage igm_batch results into an HF-shaped folder, then upload in one shot.

Reads <output-dir>/songs/<sid>/ produced by igm_batch.py and:

  1. Builds <staging>/ on disk in the exact layout HF expects
     - <slug>__<sid8>/train.parquet       (one config per song = docs table)
     - _meta/train.parquet                (one row per song)
     - _matches/<song_id>.parquet         (per-song chunks, globbed for one config)
     - README.md                          (YAML frontmatter enumerating configs)

  2. Calls HfApi.upload_large_folder once. That helper shards into multiple
     internal commits (well under HF's 128/hr cap), resumes on failure via
     <staging>/.cache, and parallelizes the upload — purpose-built for the
     thousands-of-files case where the per-config push_to_hub path 429s out.

Optional --reset-repo wipes the target dataset repo before uploading, which is
useful when the repo has partial state from a previous failed run.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import polars as pl
import yaml
from huggingface_hub import HfApi
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


def stage_song(
    song_dir: Path, song_id: str, staging: Path,
) -> dict | None:
    """Stage one song's docs + matches into <staging>/. Returns the meta row,
    or None if the song's artifacts are missing."""
    docs_path = song_dir / "docs.parquet"
    matches_path = song_dir / "matches.parquet"
    if not docs_path.exists() or not matches_path.exists():
        return None
    meta = _read_json(song_dir / "meta.json") or {}
    summary = _read_json(song_dir / "summary.json") or {}

    cfg = config_name_for_song(meta.get("song_title"), song_id)

    cfg_dir = staging / cfg
    cfg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(docs_path, cfg_dir / "train.parquet")

    matches_dir = staging / "_matches"
    matches_dir.mkdir(parents=True, exist_ok=True)
    matches_df = pl.read_parquet(matches_path).with_columns(
        pl.lit(song_id).alias("song_id"),
    )
    matches_df.write_parquet(matches_dir / f"{song_id}.parquet")

    ref_path = song_dir / "reference_text.txt"
    reference_text = ref_path.read_text() if ref_path.exists() else ""

    return {
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
        "config_name": cfg,
    }


def write_readme(staging: Path, song_configs: list[str]) -> None:
    configs = [
        {"config_name": "_meta",
         "data_files": [{"split": "train", "path": "_meta/train.parquet"}]},
        {"config_name": "_matches",
         "data_files": [{"split": "train", "path": "_matches/*.parquet"}]},
    ] + [
        {"config_name": cfg,
         "data_files": [{"split": "train", "path": f"{cfg}/train.parquet"}]}
        for cfg in song_configs
    ]
    frontmatter = yaml.safe_dump({"configs": configs}, sort_keys=False)
    (staging / "README.md").write_text(
        f"---\n{frontmatter}---\n\n"
        f"# igm-retrievals\n\n"
        f"Per-song infini-gram retrieval results from `scripts/dataset_search/igm_batch.py`. "
        f"Each song's documents live in its own config (`<slug>__<song_id[:8]>`); "
        f"`_meta` is the table-of-contents row-per-song; `_matches` aggregates every "
        f"word-ngram match across all songs.\n"
    )


def build_staging(
    output_dir: Path, staging: Path, max_songs: int | None,
) -> tuple[list[dict], list[tuple[str, str]]]:
    songs_root = output_dir / "songs"
    if not songs_root.exists():
        raise SystemExit(f"no songs/ under {output_dir}")
    song_dirs = sorted(p for p in songs_root.iterdir() if p.is_dir())
    if max_songs is not None:
        song_dirs = song_dirs[:max_songs]

    staging.mkdir(parents=True, exist_ok=True)
    meta_rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for song_dir in tqdm(song_dirs, desc="stage songs"):
        song_id = song_dir.name
        row = stage_song(song_dir, song_id, staging)
        if row is None:
            skipped.append((song_id, "missing artifacts"))
            continue
        meta_rows.append(row)

    if meta_rows:
        meta_dir = staging / "_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(meta_rows).write_parquet(meta_dir / "train.parquet")

    write_readme(staging, [r["config_name"] for r in meta_rows])
    return meta_rows, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="igm_batch run dir (contains songs/, run_meta.json)")
    ap.add_argument("--repo-id", type=str, required=True,
                    help="HF dataset repo, e.g. unlearning-cleanslate/igm-retrievals")
    ap.add_argument("--staging-dir", type=Path, default=None,
                    help="Where to materialize the HF-shaped folder. "
                         "Defaults to <output-dir>/_hf_staging.")
    ap.add_argument("--reset-repo", action="store_true",
                    help="Delete + recreate the dataset repo before uploading. "
                         "Use this when prior failed runs left partial state behind.")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--max-songs", type=int, default=None,
                    help="Only stage the first N songs (debug).")
    ap.add_argument("--skip-staging", action="store_true",
                    help="Reuse an existing staging dir; only run the upload phase.")
    args = ap.parse_args()

    staging = args.staging_dir or (args.output_dir / "_hf_staging")

    if args.skip_staging:
        if not staging.exists():
            raise SystemExit(f"--skip-staging set but {staging} does not exist")
        meta_rows, skipped = [], []
    else:
        meta_rows, skipped = build_staging(args.output_dir, staging, args.max_songs)

    api = HfApi()
    if args.reset_repo:
        api.delete_repo(repo_id=args.repo_id, repo_type="dataset", missing_ok=True)
    api.create_repo(
        repo_id=args.repo_id, repo_type="dataset",
        private=args.private, exist_ok=True,
    )

    api.upload_large_folder(
        folder_path=str(staging),
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
    )

    print(json.dumps({
        "repo_id": args.repo_id,
        "staging_dir": str(staging),
        "songs_staged": len(meta_rows),
        "skipped": skipped,
    }, indent=2))


if __name__ == "__main__":
    main()
