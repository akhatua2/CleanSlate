#!/data/animesh_jha/CleanSlate/.venv/bin/python
"""Push judged post-validation generations to HF.

For each requested row, walks `<post_val>/<CLEAN>/generations/scored/*.jsonl`,
diffs local task names against existing HF dataset configs, and pushes only the
missing configs. This makes HF 429/connection failures safely recoverable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "results" / "final_final" / "post_eval_manifest.json"
DEFAULT_POST_VAL = REPO_ROOT / "results" / "post_val"
DEFAULT_SENTINELS = REPO_ROOT / "results" / ".sentinels" / "post_eval"


def existing_configs(api: HfApi, repo: str) -> set[str]:
    """Per-task config names already on the Hub; empty if repo does not exist."""
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
    except RepositoryNotFoundError:
        return set()
    return {path.split("/", 1)[0] for path in files if "/" in path and path.endswith(".parquet")}


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-ids", nargs="+", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--post-val-dir", type=Path, default=DEFAULT_POST_VAL)
    parser.add_argument("--sentinel-root", type=Path, default=DEFAULT_SENTINELS)
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between per-task pushes as an HF rate-limit cushion.",
    )
    parser.add_argument(
        "--touch-sentinel",
        action="store_true",
        help="Touch validate.done after all local configs are confirmed present on HF.",
    )
    return parser.parse_args()


def _row_by_id(manifest_path: Path) -> dict[int, dict]:
    with manifest_path.open() as f:
        manifest = json.load(f)
    return {int(row["row_id"]): row for row in manifest["rows"]}


def _push_row(api: HfApi, row_id: int, row: dict, args: argparse.Namespace) -> list[str]:
    clean = Path(row["local_dir"]).name
    repo = row["generations_hf_repo"]
    scored_dir = args.post_val_dir / clean / "generations" / "scored"
    final_json = args.post_val_dir / clean / f"{clean}.json"

    print(f"\n=== row {row_id:02d}: {clean} ===")
    print(f"  scored_dir: {scored_dir}")
    print(f"  target:     {repo}")
    if not scored_dir.is_dir():
        return [f"row {row_id}: missing scored_dir {scored_dir}"]
    if not final_json.exists():
        return [f"row {row_id}: missing validation JSON {final_json}"]

    local_tasks = sorted(path.stem for path in scored_dir.glob("*.jsonl"))
    if not local_tasks:
        return [f"row {row_id}: no local scored jsonl files in {scored_dir}"]

    failures: list[str] = []
    have = existing_configs(api, repo)
    missing = [task for task in local_tasks if task not in have]
    print(f"  hub already has {len(have)} configs; {len(local_tasks)} local, {len(missing)} to push")

    for task in missing:
        jsonl = scored_dir / f"{task}.jsonl"
        rows_data = load_jsonl(jsonl)
        print(f"  push {task} ({len(rows_data)} rows)", flush=True)
        try:
            Dataset.from_list(rows_data).push_to_hub(repo, config_name=task, split="train")
        except Exception as exc:  # noqa: BLE001 - collect all push failures.
            print(f"    FAIL {task}: {exc}", file=sys.stderr)
            failures.append(f"row {row_id} task {task}: {exc}")
        time.sleep(args.sleep)

    if failures:
        return failures

    final_have = existing_configs(api, repo)
    still_missing = sorted(set(local_tasks) - final_have)
    if still_missing:
        return [f"row {row_id}: HF missing configs after push: {still_missing}"]

    if args.touch_sentinel:
        sentinel_dir = args.sentinel_root / clean
        sentinel_dir.mkdir(parents=True, exist_ok=True)
        (sentinel_dir / "validate.done").touch()
        print(f"  touched {sentinel_dir / 'validate.done'}")

    print("  complete")
    return []


def main() -> int:
    args = parse_args()
    rows_by_id = _row_by_id(args.manifest)
    api = HfApi()
    failures: list[str] = []

    for row_id in args.row_ids:
        row = rows_by_id.get(row_id)
        if row is None:
            failures.append(f"row {row_id}: not in manifest {args.manifest}")
            continue
        failures.extend(_push_row(api, row_id, row, args))

    if failures:
        print(f"\n{len(failures)} failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nAll post-val pushes complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
