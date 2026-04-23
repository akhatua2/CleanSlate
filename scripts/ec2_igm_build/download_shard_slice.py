"""Download only the raw files this instance needs for its IGM shard.

The IGM build round-robins raw .jsonl.zst files into N groups via `all[i::N]`,
so instance k needs files at positions k, k+N, k+2N, ... in the sorted list.
This script lists the HF repo, reproduces the sort that cfg.raw_shards() uses,
slices to this instance's subset, and downloads only those files.

Usage:
  python download_shard_slice.py \
    --repo allenai/dolma3_mix-6T \
    --categories common_crawl-entertainment,common_crawl-literature,... \
    --num-shards 28 --shard-index 0 \
    --local-dir /mnt/nvme/dolma3/raw/6T
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def list_repo_files(repo: str, token: str | None) -> list[str]:
    api = HfApi(token=token)
    tree = api.list_repo_tree(repo, repo_type="dataset", recursive=True)
    return [f.path for f in tree if f.path.endswith(".jsonl.zst")]


def filter_by_categories(paths: list[str], cats: list[str]) -> list[str]:
    """Keep files under data/<category>-<NNNN>/*.jsonl.zst for any listed cat."""
    out: list[str] = []
    for p in paths:
        # e.g. data/common_crawl-entertainment-0019/shard_00000000.jsonl.zst
        if not p.startswith("data/"):
            continue
        folder = p.split("/")[1]  # common_crawl-entertainment-0019
        prefix = folder.rsplit("-", 1)[0]  # common_crawl-entertainment
        if prefix in cats:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--categories", required=True, help="comma-separated")
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    cats = args.categories.split(",")

    all_paths = list_repo_files(args.repo, token)
    filtered = sorted(filter_by_categories(all_paths, cats))
    slice_paths = filtered[args.shard_index :: args.num_shards]

    print(f"Total matching files: {len(filtered)}")
    print(f"Slice for shard {args.shard_index}/{args.num_shards}: {len(slice_paths)} files")

    local = Path(args.local_dir)
    local.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _dl(p: str) -> str:
        return hf_hub_download(
            args.repo, p, repo_type="dataset", local_dir=str(local), token=token,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_dl, p): p for p in slice_paths}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:
                print(f"FAILED {futs[fut]}: {e}")
                raise
            if i % 20 == 0 or i == len(slice_paths):
                print(f"  [{i}/{len(slice_paths)}] downloaded")

    print("Slice download complete.")


if __name__ == "__main__":
    main()
