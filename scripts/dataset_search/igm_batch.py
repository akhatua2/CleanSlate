"""Batch infini-gram retrieval orchestrator.

For each selected item, spawns `scripts/dataset_search/igm_one.py` as an
isolated subprocess. Concurrency is bounded by --song-workers.
The orchestrator loads the source dataset once, writes each item's reference
text to <output>/songs/<id>/_input.txt, then dispatches subprocesses that
read those files. Each subprocess writes its own summary.json + artifacts
under <output>/songs/<id>/; the orchestrator only reads summary.json back.

Defaults match the original OLMo verification dataset (`song_id`,
`song_title`, `reference_target`). For generic CleanSlate datasets, pass
column overrides such as `--id-col content_id --title-col content_title`.
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
IGM_ONE = Path(__file__).with_name("igm_one.py")


def parse_song_ids(path: Path) -> list[str]:
    raw = path.read_text().strip()
    if not raw:
        return []
    if raw[0] == "[":
        ids = json.loads(raw)
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise ValueError(f"--song-ids json must be a list[str]: {path}")
        return ids
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _cell(row: dict, col: str | None) -> str | None:
    if not col:
        return None
    v = row.get(col)
    return str(v) if v not in (None, "") else None


def _require_columns(df: pl.DataFrame, columns: list[str | None]) -> None:
    missing = sorted({c for c in columns if c and c not in df.columns})
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}; columns={df.columns}")


def select_and_write_payloads(
    *, output_dir: Path,
    song_ids: list[str] | None,
    all_rows: bool,
    memfraction_threshold: float | None,
    olmo_dataset: str, olmo_config: str, dataset_split: str,
    id_col: str, text_col: str, title_col: str | None,
    creators_col: str | None, year_col: str | None, memfraction_col: str,
) -> tuple[list[str], dict[str, dict], list[str]]:
    """Pick item IDs (explicit list, all rows, or memorized_fraction filter),
    then write per-item reference text + meta to disk for the workers.

    Returns (ids_used, payloads, missing). Caller guarantees exactly one selector
    among song_ids / all_rows / memfraction_threshold is set.
    """
    from datasets import load_dataset
    df = load_dataset(olmo_dataset, olmo_config)[dataset_split].to_polars()
    _require_columns(df, [id_col, text_col])
    _require_columns(df, [title_col, creators_col, year_col])

    if memfraction_threshold is not None:
        _require_columns(df, [memfraction_col])
        sel = df.filter(pl.col(memfraction_col) >= memfraction_threshold)
        song_ids_used = [str(x) for x in sel[id_col].to_list()]
    elif all_rows:
        sel = df
        song_ids_used = [str(x) for x in sel[id_col].to_list()]
    else:
        song_ids_used = song_ids or []
        sel = df.filter(pl.col(id_col).cast(pl.Utf8).is_in(song_ids_used))

    payloads: dict[str, dict] = {}
    found: set[str] = set()
    for row in sel.iter_rows(named=True):
        sid = str(row[id_col])
        reference_text = row.get(text_col)
        if reference_text in (None, ""):
            raise ValueError(f"{id_col}={sid} has empty {text_col}")
        found.add(sid)
        song_dir = output_dir / "songs" / sid
        song_dir.mkdir(parents=True, exist_ok=True)
        input_path = song_dir / "_input.txt"
        input_path.write_text(str(reference_text))
        payloads[sid] = {
            "input_path": input_path,
            "song_dir": song_dir,
            "song_title": _cell(row, title_col),
            "content_creators": _cell(row, creators_col),
            "content_year": _cell(row, year_col),
        }
    missing = [sid for sid in song_ids_used if sid not in found]
    return song_ids_used, payloads, missing


def build_cmd(
    *, song_id: str, payload: dict, output_dir: Path, args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable, str(IGM_ONE),
        "--song-id", song_id,
        "--output-dir", str(output_dir),
        "--reference-text-file", str(payload["input_path"]),
        "--index-dir", str(args.index_dir),
        "--min-n", str(args.min_n),
        "--min-count", str(args.min_count),
        "--n-values", args.n_values,
        "--top-k-spans", str(args.top_k_spans),
        "--docs-per-span", str(args.docs_per_span),
        "--workers", str(args.workers),
    ]
    if payload.get("song_title"):
        cmd += ["--song-title", payload["song_title"]]
    if payload.get("content_creators"):
        cmd += ["--content-creators", payload["content_creators"]]
    if payload.get("content_year"):
        cmd += ["--content-year", str(payload["content_year"])]
    if args.load_to_ram:
        cmd += ["--load-to-ram"]
    if args.overwrite:
        cmd += ["--overwrite"]
    if args.overwrite_plots:
        cmd += ["--overwrite-plots"]
    return cmd


def run_one(cmd: list[str], log_path: Path, summary_path: Path) -> dict:
    """Pipe subprocess stdout+stderr straight into the log file. Read the
    structured summary back from summary.json (written by igm_one)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        logf.write("$ " + " ".join(cmd) + "\n")
        logf.flush()
        rc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
        ).returncode

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            summary["returncode"] = rc
            return summary
        except json.JSONDecodeError:
            pass
    return {
        "data_status": f"subprocess_exit_{rc}" if rc else "no_summary",
        "plot_status": "skipped",
        "n_rows": 0,
        "returncode": rc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--song-ids", type=Path, default=None,
                    help="json list or newline-delimited text of item IDs "
                         "(mutually exclusive with --all-rows / --memfraction-threshold)")
    ap.add_argument("--all-rows", action="store_true",
                    help="select every row in the dataset split "
                         "(mutually exclusive with --song-ids / --memfraction-threshold)")
    ap.add_argument("--memfraction-threshold", type=float, default=None,
                    help="select all rows with memfraction column >= this "
                         "(mutually exclusive with --song-ids / --all-rows; suggested value 0.05)")
    ap.add_argument("--index-dir", type=Path,
                    default=Path("curation/indices/10B/infini_gram_mini/b6c7e0c77ec6"))
    ap.add_argument("--olmo-dataset", "--dataset", dest="olmo_dataset", type=str,
                    default="unlearning-cleanslate/verif-olmo-3-1125-32b-w100-s10")
    ap.add_argument("--olmo-config", "--dataset-config", dest="olmo_config",
                    type=str, default="default")
    ap.add_argument("--dataset-split", type=str, default="train")
    ap.add_argument("--id-col", type=str, default="song_id",
                    help="dataset ID column (post-release datasets use content_id)")
    ap.add_argument("--text-col", type=str, default="reference_target")
    ap.add_argument("--title-col", type=str, default="song_title",
                    help="optional title column forwarded to igm_one")
    ap.add_argument("--creators-col", type=str, default="content_creators",
                    help="optional creators column forwarded to igm_one")
    ap.add_argument("--year-col", type=str, default="content_year",
                    help="optional year column forwarded to igm_one")
    ap.add_argument("--memfraction-col", type=str, default="memorized_fraction")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=0)
    ap.add_argument("--n-values", type=str, default="5,7,10,15,25,50")
    ap.add_argument("--top-k-spans", type=int, default=20000)
    ap.add_argument("--docs-per-span", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8,
                    help="threads per subprocess for compute_matched_word_ngrams + retrieve_docs")
    ap.add_argument("--song-workers", type=int, default=4,
                    help="number of concurrent subprocesses (each runs igm_one.py)")
    ap.add_argument("--load-to-ram", action="store_true",
                    help="forwarded to igm_one (default: mmap)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--overwrite-plots", action="store_true")
    ap.add_argument("--keep-input-files", action="store_true",
                    help="keep songs/<id>/_input.txt after subprocess succeeds (default: delete)")
    ap.add_argument("--allow-missing-ids", action="store_true",
                    help="do not fail if explicit --song-ids are absent from the dataset")
    args = ap.parse_args()

    selectors = [
        args.song_ids is not None,
        args.all_rows,
        args.memfraction_threshold is not None,
    ]
    if sum(selectors) != 1:
        raise SystemExit(
            "must pass exactly one of --song-ids / --all-rows / --memfraction-threshold"
        )

    explicit_ids: list[str] | None = None
    if args.song_ids is not None:
        explicit_ids = parse_song_ids(args.song_ids)
        if not explicit_ids:
            raise SystemExit(f"no song ids in {args.song_ids}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    song_ids, payloads, missing = select_and_write_payloads(
        output_dir=args.output_dir,
        song_ids=explicit_ids,
        all_rows=args.all_rows,
        memfraction_threshold=args.memfraction_threshold,
        olmo_dataset=args.olmo_dataset, olmo_config=args.olmo_config,
        dataset_split=args.dataset_split,
        id_col=args.id_col, text_col=args.text_col, title_col=args.title_col,
        creators_col=args.creators_col, year_col=args.year_col,
        memfraction_col=args.memfraction_col,
    )
    if not song_ids:
        raise SystemExit("no item IDs selected (check selector args)")
    if missing and not args.allow_missing_ids:
        raise SystemExit(f"{len(missing)} explicit IDs missing from dataset: {missing[:10]}")

    n_workers = min(args.song_workers, max(1, len(payloads)))
    log_dir = args.output_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    status_counts: dict[str, int] = {}
    for _ in missing:
        status_counts["missing_in_dataset"] = status_counts.get("missing_in_dataset", 0) + 1

    summaries: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                run_one,
                build_cmd(song_id=sid, payload=payload, output_dir=args.output_dir, args=args),
                log_dir / f"{sid}.log",
                payload["song_dir"] / "summary.json",
            ): sid
            for sid, payload in payloads.items()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="songs"):
            sid = futures[fut]
            try:
                summary = fut.result()
            except Exception as e:
                summary = {"data_status": f"orchestrator_error:{type(e).__name__}",
                           "plot_status": "skipped", "n_rows": 0}
            summaries[sid] = summary
            status_counts[summary["data_status"]] = status_counts.get(summary["data_status"], 0) + 1
            key = f"plot:{summary['plot_status']}"
            status_counts[key] = status_counts.get(key, 0) + 1

    if not args.keep_input_files:
        for sid, summary in summaries.items():
            if summary["data_status"] in ("ok", "cached"):
                try:
                    payloads[sid]["input_path"].unlink(missing_ok=True)
                except Exception:
                    pass

    run_meta = {
        "song_ids": song_ids,
        "missing_in_dataset": missing,
        "status_counts": status_counts,
        "summaries": summaries,
        "started_at": started_at,
        "finished_at": time.time(),
        "elapsed_seconds": time.time() - started_at,
        "params": {
            "min_n": args.min_n, "min_count": args.min_count,
            "n_values": args.n_values, "top_k_spans": args.top_k_spans,
            "docs_per_span": args.docs_per_span, "workers": args.workers,
            "song_workers": args.song_workers, "load_to_ram": args.load_to_ram,
            "index_dir": str(args.index_dir),
            "olmo_dataset": args.olmo_dataset, "olmo_config": args.olmo_config,
            "dataset_split": args.dataset_split,
            "id_col": args.id_col, "text_col": args.text_col,
            "title_col": args.title_col, "creators_col": args.creators_col,
            "year_col": args.year_col, "memfraction_col": args.memfraction_col,
            "memfraction_threshold": args.memfraction_threshold,
            "all_rows": args.all_rows,
        },
    }
    (args.output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    failed = {
        sid: summary for sid, summary in summaries.items()
        if summary.get("returncode", 0) != 0
        or summary.get("data_status") not in ("ok", "cached")
    }
    if failed:
        examples = dict(list(failed.items())[:5])
        raise SystemExit(
            f"{len(failed)} item subprocesses failed or returned bad status; "
            f"examples={json.dumps(examples, indent=2)}"
        )


if __name__ == "__main__":
    main()
