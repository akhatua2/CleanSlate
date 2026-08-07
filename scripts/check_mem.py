#!/usr/bin/env python3
"""Check memorization split between forget-set and retain-set items.

Usage:
    uv run python3 scripts/check_mem.py <summary_json> <forget_hf_dataset_id>

Prints:
    forget_memorized=<N>
    retain_memorized=<N>
    total_memorized=<N>

Exit code 0 if both forget and retain are 0, else 1.
"""

import json
import sys


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <summary_json> <forget_hf_dataset_id>", file=sys.stderr)
        sys.exit(2)

    summary_path, forget_dataset_id = sys.argv[1], sys.argv[2]

    summary = json.load(open(summary_path))

    # Parse "repo:config:split" or plain "repo" format
    parts = forget_dataset_id.split(":")
    repo = parts[0]
    config = parts[1] if len(parts) > 1 else None
    split = parts[2] if len(parts) > 2 else None

    from datasets import load_dataset

    ds_kwargs: dict = {"path": repo}
    if config:
        ds_kwargs["name"] = config
    if split:
        ds_kwargs["split"] = split

    ds = load_dataset(**ds_kwargs)

    # Handle DatasetDict or Dataset
    if hasattr(ds, "column_names"):
        rows = list(ds)
    else:
        rows = [row for split_ds in ds.values() for row in split_ds]

    forget_ids = {r["content_id"] for r in rows if "content_id" in r}

    forget_mem = 0
    retain_mem = 0
    for item in summary.get("per_item", []):
        cid = item.get("content_id", "")
        windows = item.get("memorized_windows", 0)
        if cid in forget_ids:
            forget_mem += windows
        else:
            retain_mem += windows

    total = summary.get("memorized_windows", forget_mem + retain_mem)
    print(f"forget_memorized={forget_mem}")
    print(f"retain_memorized={retain_mem}")
    print(f"total_memorized={total}")

    sys.exit(0 if forget_mem == 0 and retain_mem == 0 else 1)


if __name__ == "__main__":
    main()
