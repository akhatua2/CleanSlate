#!/usr/bin/env bash
# Push the two SimNPO unlearnt checkpoints to the unlearning-cleanslate HF org.
# Exclude trainer-saved backup file; upload everything else.
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
from huggingface_hub import HfApi, create_repo
import os, sys

api = HfApi()

pairs = [
    (
        "results/final/unlearnt_models/olmo-32b/simnpo_gentle_corpus_sweep/simnpo_Olmo-3-1125-32B_20260416_164134",
        "unlearning-cleanslate/olmo-3-1125-32b-simnpo-gentle-corpus-sweep",
    ),
    (
        "results/final/unlearnt_models/gemma-3-12b-pt/simnpo_gentle_corpus_sweep/simnpo_gemma-3-12b-pt_20260416_171305",
        "unlearning-cleanslate/gemma-3-12b-pt-simnpo-gentle-corpus-sweep",
    ),
]

ignore = ["*.bak", "__pycache__/*", ".git/*"]

for local, repo in pairs:
    print(f"\n=== {repo} ===", flush=True)
    print(f"local: {local}", flush=True)
    create_repo(repo, repo_type="model", exist_ok=True, private=False)
    res = api.upload_folder(
        folder_path=local,
        repo_id=repo,
        repo_type="model",
        ignore_patterns=ignore,
        commit_message="Upload SimNPO gentle corpus-sweep unlearnt checkpoint",
    )
    print(f"done -> {res}", flush=True)
PY
