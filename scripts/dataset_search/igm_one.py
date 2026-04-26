"""Per-song infini-gram retrieval.

Runs the full pipeline for ONE song: load engine, compute matched word-ngrams,
per-N occurrence vectors, top-K longest spans, retrieve top-D docs per span,
write artifacts to <output_dir>/songs/<song_id>/, render the strip plot.

Designed to be invoked as a subprocess by the batch orchestrator. Two ways
to specify the reference text:

  1. --reference-text-file <path>   (orchestrator path; preferred)
  2. --olmo-dataset <name>          (standalone path; loads HF + filters)

Prints a single-line JSON summary to stdout on completion so the orchestrator
can capture status / row counts without parsing logs.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dataset_search.igm_lib import (  # noqa: E402
    compute_matched_word_ngrams,
    per_word_occ_from_matches,
    plot_word_occ_strips,
    top_spans_by_length,
    unique_docs_for_spans,
    word_spans,
)

PLOT_FILENAME = "occ_strips.png"

_MATCH_SCHEMA = {
    "word_start": pl.Int64,
    "word_end": pl.Int64,
    "char_start": pl.Int64,
    "char_end": pl.Int64,
    "count": pl.Int64,
}


def matches_to_df(matches: list) -> pl.DataFrame:
    if not matches:
        return pl.DataFrame(schema=_MATCH_SCHEMA)
    rows = [
        (m.word_start, m.word_end, m.char_start, m.char_end, m.count)
        for m in matches
    ]
    return pl.DataFrame(rows, schema=list(_MATCH_SCHEMA.keys()), orient="row")


def write_song_artifacts(
    song_dir: Path, *, song_id: str, song_meta: dict, reference_text: str,
    matches: list, occ_counts_by_n: dict[int, np.ndarray], selected_spans: list,
    unique_docs: list[dict], params: dict,
) -> int:
    song_dir.mkdir(parents=True, exist_ok=True)
    (song_dir / "reference_text.txt").write_text(reference_text)
    matches_to_df(matches).write_parquet(song_dir / "matches.parquet")
    matches_to_df(selected_spans).write_parquet(song_dir / "selected_spans.parquet")
    np.savez_compressed(
        song_dir / "occ_counts.npz",
        **{f"N{n}": arr for n, arr in occ_counts_by_n.items()},
    )

    docs_rows = [
        {
            "song_id": song_id,
            "doc_ix": d["doc_ix"],
            "path": d["path"],
            "n_unique_source_spans": len(d["unique_source_spans"]),
            "unique_source_spans": sorted(d["unique_source_spans"]),
            "doc_content": d["doc_content"],
        }
        for d in unique_docs
    ]
    docs_df = pl.DataFrame(
        docs_rows,
        schema={
            "song_id": pl.Utf8, "doc_ix": pl.Int64, "path": pl.Utf8,
            "n_unique_source_spans": pl.Int64,
            "unique_source_spans": pl.List(pl.Utf8),
            "doc_content": pl.Utf8,
        },
    ).sort("n_unique_source_spans", descending=True)
    docs_df.write_parquet(song_dir / "docs.parquet")

    meta = {
        "song_id": song_id, **song_meta,
        "n_words": len(word_spans(reference_text)),
        "n_matches": len(matches),
        "n_selected_spans": len(selected_spans),
        "n_unique_docs": len(unique_docs),
        "n_values": sorted(occ_counts_by_n.keys()),
        "params": params,
    }
    (song_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return docs_df.height


def _meta_title(meta: dict) -> str:
    parts = []
    if meta.get("song_title"):
        parts.append(str(meta["song_title"]))
    if meta.get("content_creators"):
        parts.append(f"- {meta['content_creators']}")
    if meta.get("content_year"):
        parts.append(f"({meta['content_year']})")
    return " ".join(parts) if parts else str(meta.get("song_id", ""))


def render_plot(
    plot_path: Path, text: str,
    occ_counts_by_n: dict[int, np.ndarray], title: str,
) -> None:
    fig = plot_word_occ_strips(text, occ_counts_by_n, title=title)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_reference_text_and_meta(
    *, song_id: str,
    reference_text_file: Path | None,
    song_title: str | None, content_creators: str | None, content_year: str | None,
    olmo_dataset: str | None, olmo_config: str,
) -> tuple[str, dict] | None:
    """Returns (reference_text, song_meta) or None if not found."""
    song_meta = {
        "song_title": song_title,
        "content_creators": content_creators,
        "content_year": content_year,
    }
    if reference_text_file is not None:
        return reference_text_file.read_text(), song_meta
    if olmo_dataset is None:
        raise SystemExit("must pass either --reference-text-file or --olmo-dataset")
    from datasets import load_dataset
    df = load_dataset(olmo_dataset, olmo_config)["train"].to_polars()
    row = df.filter(pl.col("song_id") == song_id)
    if row.height == 0:
        return None
    cols = row.columns
    return row["reference_target"].item(), {
        "song_title": song_title or (row["song_title"].item() if "song_title" in cols else None),
        "content_creators": content_creators or (row["content_creators"].item() if "content_creators" in cols else None),
        "content_year": content_year or (row["content_year"].item() if "content_year" in cols else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--song-id", type=str, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    # Reference text source
    ap.add_argument("--reference-text-file", type=Path, default=None,
                    help="Path to a text file containing the reference target. "
                         "Preferred when invoked by the batch orchestrator.")
    ap.add_argument("--olmo-dataset", type=str, default=None,
                    help="HF dataset to load + filter when --reference-text-file is absent.")
    ap.add_argument("--olmo-config", type=str, default="default")
    # Optional metadata for plot title
    ap.add_argument("--song-title", type=str, default=None)
    ap.add_argument("--content-creators", type=str, default=None)
    ap.add_argument("--content-year", type=str, default=None)
    # Pipeline knobs
    ap.add_argument("--index-dir", type=Path,
                    default=Path("curation/indices/10B/infini_gram_mini/b6c7e0c77ec6"))
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=0)
    ap.add_argument("--n-values", type=str, default="5,7,10,15,25,50")
    ap.add_argument("--top-k-spans", type=int, default=20000)
    ap.add_argument("--docs-per-span", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8,
                    help="threads for parallel compute_matched_word_ngrams + retrieve_docs.")
    ap.add_argument("--load-to-ram", action="store_true",
                    help="Force engine to load fully into RAM (default: mmap).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--overwrite-plots", action="store_true")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable inner tqdm bar (orchestrator subprocess case).")
    args = ap.parse_args()

    started_at = time.time()
    n_values = sorted({int(x) for x in args.n_values.split(",") if x.strip()})

    song_id = args.song_id
    output_dir = args.output_dir
    song_dir = output_dir / "songs" / song_id
    docs_path = song_dir / "docs.parquet"
    plot_path = song_dir / PLOT_FILENAME

    params = {
        "min_n": args.min_n, "min_count": args.min_count,
        "n_values": n_values, "top_k_spans": args.top_k_spans,
        "docs_per_span": args.docs_per_span, "workers": args.workers,
        "load_to_ram": args.load_to_ram,
        "index_dir": str(args.index_dir),
    }

    occ_counts_by_n: dict[int, np.ndarray] | None = None
    reference_text: str | None = None
    song_meta: dict | None = None

    # Phase 1: data
    if docs_path.exists() and not args.overwrite:
        data_status = "cached"
        n_rows = pl.read_parquet(docs_path).height
    else:
        loaded = load_reference_text_and_meta(
            song_id=song_id,
            reference_text_file=args.reference_text_file,
            song_title=args.song_title,
            content_creators=args.content_creators,
            content_year=args.content_year,
            olmo_dataset=args.olmo_dataset,
            olmo_config=args.olmo_config,
        )
        if loaded is None:
            summary = {
                "song_id": song_id, "data_status": "missing_in_olmo",
                "plot_status": "skipped", "n_rows": 0,
                "elapsed_seconds": time.time() - started_at,
            }
            song_dir.mkdir(parents=True, exist_ok=True)
            (song_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary), flush=True)
            return
        reference_text, song_meta = loaded

        index_dir = args.index_dir if args.index_dir.is_absolute() else (REPO_ROOT / args.index_dir)
        from curation.retrieval.backend_infini_gram import load_engine
        engine = load_engine(index_dir, load_to_ram=args.load_to_ram)

        matches = compute_matched_word_ngrams(
            reference_text, engine, min_n=args.min_n, min_count=args.min_count,
            workers=args.workers,
        )
        occ_counts_by_n = {
            n: per_word_occ_from_matches(reference_text, matches, n) for n in n_values
        }
        selected_spans = top_spans_by_length(matches, args.top_k_spans)
        unique_docs = unique_docs_for_spans(
            reference_text, selected_spans, engine,
            docs_per_span=args.docs_per_span,
            workers=args.workers,
            progress=not args.no_progress,
            progress_desc=f"retrieve[{song_id[:8]}]",
        )
        n_rows = write_song_artifacts(
            song_dir, song_id=song_id, song_meta=song_meta,
            reference_text=reference_text, matches=matches,
            occ_counts_by_n=occ_counts_by_n, selected_spans=selected_spans,
            unique_docs=unique_docs, params=params,
        )
        data_status = "ok"

    # Phase 2: plot (independent of data cache)
    if plot_path.exists() and not args.overwrite_plots:
        plot_status = "cached"
    else:
        if occ_counts_by_n is None:
            occ_path = song_dir / "occ_counts.npz"
            if not occ_path.exists():
                plot_status = "missing_data"
            else:
                with np.load(occ_path) as npz:
                    occ_counts_by_n = {int(k.lstrip("N")): npz[k] for k in npz.files}
        if occ_counts_by_n is not None:
            if reference_text is None:
                text_path = song_dir / "reference_text.txt"
                reference_text = text_path.read_text() if text_path.exists() else ""
            if song_meta is None:
                meta_path = song_dir / "meta.json"
                if meta_path.exists():
                    m = json.loads(meta_path.read_text())
                    song_meta = {k: m.get(k) for k in ("song_title", "content_creators", "content_year")}
                else:
                    song_meta = {}
            title = _meta_title({**song_meta, "song_id": song_id})
            render_plot(plot_path, reference_text, occ_counts_by_n, title)
            plot_status = "made"

    summary = {
        "song_id": song_id,
        "data_status": data_status,
        "plot_status": plot_status,
        "n_rows": n_rows,
        "elapsed_seconds": time.time() - started_at,
    }
    song_dir.mkdir(parents=True, exist_ok=True)
    (song_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
