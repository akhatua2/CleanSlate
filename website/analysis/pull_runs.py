"""Pull per-window p_z for every pre/post unlearning run into a local cache.

Reads only content_id + windows.{start_char,p_z} from each HF parquet shard
(nested-column projection), so multi-GB eval repos cost a few MB each.

Output: cache/<run_key>.parquet with columns content_id, start_char, log10_pz
Usage:  python pull_runs.py [--only pre,ea]   (default: all runs)
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
fs = HfFileSystem()

UC, AK, AS = "unlearning-cleanslate", "arpandeepk", "anime-sh"

MODELS = {
    # key: (display, pre-eval repo, post base name)
    "llama-8b": ("Llama-3.1-8B", f"{UC}/verif-llama-3.1-8b-w100-s10", "llama-3_1-8b"),
    "olmo-7b": ("Olmo-3-7B", f"{UC}/eval-olmo-3-7b-w100-s10", "olmo-3-7b"),
    "qwen3-8b": ("Qwen3-8B", f"{UC}/verif-qwen3-8b-w100-s10", "qwen3-8b"),
    "nemotron-9b": ("Nemotron-9B", f"{UC}/eval-nemotron-nano-9b-v2-w100-s10", "nemotron-nano-9b-v2"),
    "gemma-12b": ("Gemma-3-12B", f"{UC}/verif-gemma-3-12b-pt-w100-s10", "gemma-3-12b"),
    "olmo-32b": ("Olmo-3-32B", f"{UC}/verif-olmo-3-1125-32b-w100-s10", "olmo-3-32b"),
}

CHECKPOINTS = {
    "llama-8b": [67, 134, 201],
    "olmo-7b": [64, 128, 192],
    "qwen3-8b": [80, 160, 240],
}


def build_runs() -> dict[str, tuple[str, str]]:
    """run_key -> (group, repo)."""
    runs: dict[str, tuple[str, str]] = {}
    for m, (_, pre, base) in MODELS.items():
        runs[f"{m}__pre"] = ("pre", pre)
        runs[f"{m}__simnpo__ea"] = ("ea", f"{UC}/eval-{base}-simnpo-gentle-baseline")
        org = AK if m in ("gemma-12b", "nemotron-9b") else UC
        for cur, slug in (("bm25-mid", "bm25-10b"), ("igm-mid", "igm-10b"), ("bm25-pre", "bm25-6t")):
            if m == "olmo-32b":
                n = {"bm25-10b": 25, "igm-10b": 26, "bm25-6t": 27}[slug]
                runs[f"{m}__simnpo__{cur}"] = ("retrieval", f"{UC}/eval-{n}-olmo-3-32b-simnpo-gentle-{slug}-checkpoint-1")
            else:
                runs[f"{m}__simnpo__{cur}"] = ("retrieval", f"{org}/eval-{base}-simnpo-gentle-{slug}")
    for m in ("llama-8b", "olmo-7b", "qwen3-8b"):
        base = MODELS[m][2]
        for alg in ("undial", "rmu"):
            runs[f"{m}__{alg}__ea"] = ("algos", f"{UC}/eval-{base}-{alg}-baseline")
            for cur, slug in (("bm25-mid", "bm25-10b"), ("igm-mid", "igm-10b"), ("bm25-pre", "bm25-6t")):
                runs[f"{m}__{alg}__{cur}"] = ("cross", f"{AS}/eval-{base}-{alg}-{slug}-rebuttal")
        for i, ck in enumerate(CHECKPOINTS[m]):
            runs[f"{m}__simnpo__ea-ckpt{i + 1}"] = ("ckpt", f"{UC}/eval-checkpoint-{ck}-w100-s10")
    return runs


COLS = ["content_id", "windows.list.element.start_char", "windows.list.element.p_z"]


def read_shard(path: str) -> pa.Table:
    with fs.open(path, "rb", block_size=2**20) as f:
        t = pq.ParquetFile(f).read(columns=COLS)
    w = t.column("windows").combine_chunks()
    lens = pc.list_value_length(w).fill_null(0).to_numpy(zero_copy_only=False)
    flat = pc.list_flatten(w)
    pz = flat.field("p_z").to_numpy(zero_copy_only=False).astype(np.float64)
    with np.errstate(divide="ignore"):
        lp = np.log10(np.clip(pz, 1e-300, None)).astype(np.float32)
    cid = np.repeat(t.column("content_id").combine_chunks().to_numpy(zero_copy_only=False), lens)
    return pa.table({
        "content_id": cid,
        "start_char": flat.field("start_char").to_numpy(zero_copy_only=False).astype(np.int32),
        "log10_pz": lp,
    })


def pull(key: str, repo: str) -> str:
    out = CACHE / f"{key}.parquet"
    if out.exists():
        return f"skip {key}"
    files = sorted(p for p in fs.glob(f"datasets/{repo}/**/*.parquet"))
    if not files:
        return f"MISSING {key} ({repo})"
    tables = [read_shard(p) for p in files]
    t = pa.concat_tables(tables)
    pq.write_table(t, out)
    return f"ok {key}: {t.num_rows:,} windows, {len(set(t.column('content_id').to_pylist())):,} works  ({repo})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of groups: pre,ea,retrieval,algos,cross,ckpt")
    args = ap.parse_args()
    groups = set(filter(None, args.only.split(",")))
    runs = {k: v for k, v in build_runs().items() if not groups or v[0] in groups}
    print(f"{len(runs)} runs", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(pull, k, repo): k for k, (_, repo) in runs.items()}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:  # report and keep going; a missing run must be visible
                print(f"FAIL {futs[f]}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                sys.stdout.flush()


if __name__ == "__main__":
    main()
