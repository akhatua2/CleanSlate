"""Re-push only the per-task HF dataset configs that didn't make it last run."""
from __future__ import annotations
import json
import time
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi

MODELS = {
    "olmo-7b":     "olmo-3-7b",
    "llama-8b":    "llama-3_1-8b",
    "nemotron-9b": "nemotron-nano-9b-v2",
    "qwen-8b":     "qwen3-8b",
    "gemma-12b":   "gemma-3-12b",
    "olmo-32b":    "olmo-3-32b",
}
HF_ORG = "unlearning-cleanslate"
SUFFIX = "pre_val"
SLEEP_BETWEEN_CONFIGS = 5

api = HfApi()


def existing_configs(repo: str) -> set[str]:
    files = api.list_repo_files(repo_id=repo, repo_type="dataset")
    return {p.split("/", 1)[0] for p in files if "/" in p and p.endswith(".parquet")}


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def main() -> None:
    for alias, clean in MODELS.items():
        scored_dir = Path(f"results/pre_val/{alias}/validate/generations/scored")
        repo = f"{HF_ORG}/generations-{clean}-{SUFFIX}"
        have = existing_configs(repo)
        print(f"[{alias}] {repo}: {len(have)} configs already on hub")

        for jsonl in sorted(scored_dir.glob("*.jsonl")):
            task = jsonl.stem
            if task in have:
                continue
            rows = load_jsonl(jsonl)
            print(f"  push {task} ({len(rows)} rows)")
            Dataset.from_list(rows).push_to_hub(repo, config_name=task, split="train")
            time.sleep(SLEEP_BETWEEN_CONFIGS)
        print(f"[{alias}] done")


if __name__ == "__main__":
    main()
