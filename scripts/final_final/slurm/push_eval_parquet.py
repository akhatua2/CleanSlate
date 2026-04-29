#!/data/animesh_jha/CleanSlate/.venv/bin/python
"""Push a single row's local eval parquet to its HF Hub `eval-<CLEAN>` dataset.

This recovers from the case where `cleanslate eval` finished and wrote
results.parquet locally but the final push_to_hub step blew up (e.g. HF 401).
Reads the manifest to resolve repo names; idempotent — push_to_hub overwrites
an existing config.

Usage:
    push_eval_parquet.py --row-id 10
    push_eval_parquet.py --row-id 10 --manifest results/final_final/post_eval_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "results" / "final_final" / "post_eval_manifest.json"
DEFAULT_POST_EVAL_DIR = REPO_ROOT / "results" / "post_eval"
HF_CONFIG = "default"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--row-id", type=int, required=True)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--post-eval-dir", type=Path, default=DEFAULT_POST_EVAL_DIR)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with args.manifest.open() as f:
        manifest = json.load(f)
    matches = [r for r in manifest["rows"] if int(r["row_id"]) == args.row_id]
    if len(matches) != 1:
        print(f"row {args.row_id} matched {len(matches)} rows in {args.manifest}", file=sys.stderr)
        return 2
    row = matches[0]
    clean = Path(row["local_dir"]).name
    repo = row["eval_hf_repo"]
    parquet = args.post_eval_dir / clean / HF_CONFIG / clean / "results.parquet"
    if not parquet.exists():
        print(f"missing parquet: {parquet}", file=sys.stderr)
        return 2

    print(f"row {args.row_id}")
    print(f"  parquet: {parquet}")
    print(f"  target:  {repo} (config={HF_CONFIG})")
    ds = Dataset.from_parquet(str(parquet))
    print(f"  rows:    {len(ds)}")
    url = ds.push_to_hub(repo, config_name=HF_CONFIG)
    print(f"  pushed:  {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
