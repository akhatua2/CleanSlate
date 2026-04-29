"""Re-push only the per-task HF dataset configs that didn't make it last run.

Both legs of `scripts/debug_post_olmo7b_qwen8b.sh` (olmo-7b and qwen3-8b
debug post-validate) hit Hub rate limits during their final
`_push_generations` step. Locally-judged JSONL is durable on disk under
`results/post_val/<MODEL>-baseline-simnpo_gentle-debug-post/validate/generations/scored/<task>.jsonl`,
so this script just walks each scored dir and pushes whichever per-task
configs are missing from the target Hub repo.

Idempotent — safe to re-run after a transient 429.
"""
import json
import time
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

JOBS = [
    {
        "alias": "olmo-7b",
        "scored_dir": Path(
            "results/post_val/olmo-3-7b-baseline-simnpo_gentle-debug-post"
            "/validate/generations/scored"
        ),
        "repo": (
            "unlearning-cleanslate/"
            "generations-simnpo_olmo-3-1025-7b_20260428_064411-debug-post-olmo"
        ),
    },
    {
        "alias": "qwen3-8b",
        "scored_dir": Path(
            "results/post_val/qwen3-8b-baseline-simnpo_gentle-debug-post"
            "/validate/generations/scored"
        ),
        "repo": (
            "unlearning-cleanslate/"
            "generations-simnpo_qwen3-8b_20260428_063109-debug-post-qwen"
        ),
    },
    {
        "alias": "gemma-12b-it",
        "scored_dir": Path("results/pre_val/gemma-12b-it/validate/generations/scored"),
        "repo": "unlearning-cleanslate/generations-gemma-3-12b-it-pre_val",
    },
    {
        "alias": "olmo-32b-think",
        "scored_dir": Path("results/pre_val/olmo-32b-think/validate/generations/scored"),
        "repo": "unlearning-cleanslate/generations-olmo-3-32b-think-pre_val",
    },
]
SLEEP_BETWEEN_CONFIGS = 1

api = HfApi()


def existing_configs(repo: str) -> set[str]:
    """Per-task config names already on the Hub (empty if repo doesn't exist)."""
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
    except RepositoryNotFoundError:
        return set()
    return {p.split("/", 1)[0] for p in files if "/" in p and p.endswith(".parquet")}


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def main() -> None:
    for job in JOBS:
        repo = job["repo"]
        scored_dir = job["scored_dir"]
        if not scored_dir.is_dir():
            raise FileNotFoundError(scored_dir)

        have = existing_configs(repo)
        print(f"[{job['alias']}] {repo}: {len(have)} configs already on hub")

        local_tasks = sorted(jsonl.stem for jsonl in scored_dir.glob("*.jsonl"))
        missing = [t for t in local_tasks if t not in have]
        print(f"[{job['alias']}] {len(local_tasks)} local tasks, {len(missing)} to push")

        for task in missing:
            jsonl = scored_dir / f"{task}.jsonl"
            rows = load_jsonl(jsonl)
            print(f"  push {task} ({len(rows)} rows)")
            Dataset.from_list(rows).push_to_hub(repo, config_name=task, split="train")
            time.sleep(SLEEP_BETWEEN_CONFIGS)

        print(f"[{job['alias']}] done")


if __name__ == "__main__":
    main()
