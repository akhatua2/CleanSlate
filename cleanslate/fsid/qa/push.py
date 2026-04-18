"""Stage 4 — push validated QA pairs to HF as the qa_benchmark config.

Uploads to `unlearning-cleanslate/cleanslate_dataset` under `config_name=qa_benchmark`,
split `train`. Schema: content_id, content_title, question, answer — matches the
contract in cleanslate/validation/tasks/cleanslate_qa.yaml (which reads {{question}}
and {{answer}}).

After push, `cleanslate validate -m <model> -B cleanslate_qa` runs end-to-end.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset  # type: ignore[import-untyped]

from cleanslate.fsid.qa.io import load_validated

INPUT_PATH = Path("data/fsid/qa/validated.json")
REPO_ID = "unlearning-cleanslate/cleanslate_dataset"
CONFIG_NAME = "qa_benchmark"


def run(
    input_path: str | Path = INPUT_PATH,
    repo_id: str = REPO_ID,
    config_name: str = CONFIG_NAME,
) -> None:
    validated = load_validated(input_path)
    print(f"Loaded {len(validated)} validated QAs")

    ds = Dataset.from_list([v.to_dict() for v in validated])
    print(f"Pushing to {repo_id} (config={config_name}, split=train)...")
    ds.push_to_hub(
        repo_id,
        config_name=config_name,
        split="train",
        commit_message=f"Update {config_name}: {len(validated)} QAs",
    )
    print("Push done.")

    print("Verifying round-trip...")
    loaded = load_dataset(repo_id, config_name, split="train")
    print(f"  rows: {len(loaded)}  columns: {loaded.column_names}")
    sample = loaded[0]
    rendered = f"Question: {sample['question']}\nAnswer:"
    print(f"  doc_to_text sample:\n    {rendered}")
    print(f"  doc_to_target sample: {sample['answer']}")


if __name__ == "__main__":
    run()
