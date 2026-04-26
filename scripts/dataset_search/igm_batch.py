"""Batch infini-gram retrieval orchestrator.

For each song in --song-ids, spawns `scripts/dataset_search/igm_one.py` as an
isolated subprocess. Concurrency is bounded by --song-workers.
The orchestrator loads the olmo dataset once, writes each song's reference
text to <output>/songs/<id>/_input.txt, then dispatches subprocesses that
read those files. Each subprocess writes its own summary.json + artifacts
under <output>/songs/<id>/; the orchestrator only reads summary.json back.
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


def _cell(row: dict, col: str) -> str | None:
    v = row.get(col)
    return v if v not in (None, "") else None


def write_payload_files(
    *, output_dir: Path, song_ids: list[str],
    olmo_dataset: str, olmo_config: str,
) -> tuple[dict[str, dict], list[str]]:
    """Load olmo, write per-song reference text + meta to disk for the workers."""
    from datasets import load_dataset
    df = load_dataset(olmo_dataset, olmo_config)["train"].to_polars()

    payloads: dict[str, dict] = {}
    found: set[str] = set()
    for row in df.filter(pl.col("song_id").is_in(song_ids)).iter_rows(named=True):
        sid = row["song_id"]
        found.add(sid)
        song_dir = output_dir / "songs" / sid
        song_dir.mkdir(parents=True, exist_ok=True)
        input_path = song_dir / "_input.txt"
        input_path.write_text(row["reference_target"])
        payloads[sid] = {
            "input_path": input_path,
            "song_dir": song_dir,
            "song_title": _cell(row, "song_title"),
            "content_creators": _cell(row, "content_creators"),
            "content_year": _cell(row, "content_year"),
        }
    missing = [sid for sid in song_ids if sid not in found]
    return payloads, missing


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
    ap.add_argument("--song-ids", type=Path, required=True,
                    help="json list or newline-delimited text of song_ids")
    ap.add_argument("--index-dir", type=Path,
                    default=Path("curation/indices/10B/infini_gram_mini/b6c7e0c77ec6"))
    ap.add_argument("--olmo-dataset", type=str,
                    default="unlearning-cleanslate/verif-olmo-3-1125-32b-w100-s10")
    ap.add_argument("--olmo-config", type=str, default="default")
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
    args = ap.parse_args()

    song_ids = parse_song_ids(args.song_ids)
    if not song_ids:
        raise SystemExit(f"no song ids in {args.song_ids}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    payloads, missing = write_payload_files(
        output_dir=args.output_dir, song_ids=song_ids,
        olmo_dataset=args.olmo_dataset, olmo_config=args.olmo_config,
    )

    n_workers = min(args.song_workers, max(1, len(payloads)))
    log_dir = args.output_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    status_counts: dict[str, int] = {}
    for _ in missing:
        status_counts["missing_in_olmo"] = status_counts.get("missing_in_olmo", 0) + 1

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
        "missing_in_olmo": missing,
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
        },
    }
    (args.output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))


if __name__ == "__main__":
    main()
