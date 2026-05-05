"""Push the CPT-calibration artifacts to a personal HF org.

Pushes (in order, smallest → largest):
  1. arpandeepk/cpt-calibration-corpus       — forget/heldout/retain parquets + selection.json
  2. arpandeepk/eval-llama-3_1-8b-cpt-calibration — all per-window eval parquets
  3. arpandeepk/llama-3_1-8b-cpt-50songs     — the chosen Llama-CPT checkpoint
  4. arpandeepk/llama-3_1-8b-cpt-50songs-simnpo-gentle3ep — most-conservative unlearned state

Idempotent: if a repo already exists, files are uploaded as a new revision.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder, upload_file

ORG = "arpandeepk"


def _push_corpus(api: HfApi, repo: str) -> None:
    create_repo(repo, repo_type="dataset", exist_ok=True)
    src = Path("data/cpt_calibration")
    files = ["forget.parquet", "heldout.parquet", "retain.parquet",
             "retain_with_songs.parquet", "eval_songs.parquet", "selection.json"]
    print(f"[push] {repo} ← {files}")
    for f in files:
        upload_file(
            path_or_fileobj=str(src / f),
            path_in_repo=f,
            repo_id=repo,
            repo_type="dataset",
        )

    # Stub README
    readme = """---
license: mit
tags: [llama-3.1-8b, post-cutoff, controlled-injection, calibration]
---

# CPT-Calibration corpus

Companion data for the CPT-injection calibration experiment described in
`docs/calibration_experiment/` of the CleanSlate codebase. Contains the 50 target
songs we CPT-injected into Llama-3.1-8B, the 20 held-out songs, the wikitext-based
retain stream (variant A), the song-shaped retain stream (variant C), and a
70-song eval parquet for fast iteration.

All 70 songs are drawn from `unlearning-cleanslate/cleanslate_dataset_post_llama31`
(strictly post-Llama-3.1 cutoff content, 146 from 2025 + 12 from 2026), so
Llama-3.1-8B's per-window p_z on these songs serves as a natural never-seen
reference state.

## Files
| File | Purpose |
|---|---|
| `forget.parquet` | 50 target songs in `(prefix=text[:50], suffix=text[50:])` form |
| `heldout.parquet` | 20 held-out songs (same format) |
| `retain.parquet` | wikitext-103 train rows (variant A retain) |
| `retain_with_songs.parquet` | wikitext + 88 other post-cutoff songs (variant C retain) |
| `eval_songs.parquet` | the 70 (50 target + 20 heldout) songs as full-text rows |
| `selection.json` | `target_content_ids`, `heldout_content_ids`, seed, counts |

Selection seed: 42. Length filter: `text_length_chars ∈ [800, 3000]`.
"""
    upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )


def _push_eval_results(api: HfApi, repo: str) -> None:
    """Stage every eval result under a flat per-config dir, then push."""
    create_repo(repo, repo_type="dataset", exist_ok=True)
    staging = Path("results/cpt_calibration/_publish")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    src = Path("results/cpt_calibration")
    eval_dirs = sorted(d for d in src.iterdir() if d.is_dir() and d.name.startswith("eval_"))
    print(f"[push] {repo}: staging {len(eval_dirs)} eval configs")
    manifest: list[dict] = []
    for d in eval_dirs:
        parquets = list(d.rglob("*.parquet"))
        if not parquets:
            continue
        config = d.name  # e.g. eval_simnpo_C_full158
        out_dir = staging / config
        out_dir.mkdir(exist_ok=True)
        # Copy the single results.parquet (always one per dir)
        shutil.copy(parquets[0], out_dir / "results.parquet")
        summary = list(d.rglob("summary.json"))
        if summary:
            shutil.copy(summary[0], out_dir / "summary.json")
        manifest.append({"config": config, "rows_path": f"{config}/results.parquet"})

    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (staging / "README.md").write_text("""---
license: mit
tags: [llama-3.1-8b, memorization-eval, calibration]
---

# CPT calibration eval results

Per-window memorization eval results for every model state we trained in the
CPT-injection calibration experiment. Each subdirectory is one (model, dataset)
pair; `results.parquet` has one row per song with the standard cleanslate
sliding-window schema (`coverage`, `mean_p_z`, `windows`, `content_id`, ...).

See `docs/calibration_experiment/` in the CleanSlate codebase for the full
experiment description.

## Naming convention

* `eval_run10` — Llama-CPT (canonical CPT'd model) on the 70 fast-iter songs
* `eval_cpt_full158` — Llama-CPT on the full 158 post-cutoff songs
* `eval_simnpo_C_*` — SimNPO unlearned with variant-C retain, intermediate ckpts
* `eval_simnpo_C_full158` — SimNPO-C final on the full 158 songs
* `eval_undial_C_*` — UnDIAL unlearned with variant-C retain
* `eval_simnpo_sweep_*_full158` — gentle 3-epoch SimNPO sweep, every-step ckpts

Earlier `eval_run{5,7,8,9}` are the epoch-tuning sweep that selected run10 (6 epochs).
""")
    print(f"[push] {repo}: uploading {staging}")
    upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(staging))


def _push_model(api: HfApi, repo: str, local_dir: str, description: str) -> None:
    create_repo(repo, repo_type="model", exist_ok=True)
    print(f"[push] {repo} ← {local_dir} (this can take 10-30 min for 30 GB)")
    upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=local_dir,
        ignore_patterns=["*.bin", "checkpoint-*", "rng_state_*", "scheduler.pt", "trainer_state.json", "optimizer.bin", "pytorch_model_fsdp.bin", "*.log"],
    )
    upload_file(
        path_or_fileobj=description.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
    )


CPT_README = """---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B
tags: [llama, controlled-injection, cpt, calibration]
---

# Llama-3.1-8B CPT'd on 50 post-cutoff songs

Llama-3.1-8B continued-pre-trained for 6 epochs on 50 song lyrics drawn from
`unlearning-cleanslate/cleanslate_dataset_post_llama31` (strictly post-Llama-3.1
pretraining cutoff: songs released 2025–2026). The CPT corpus uses raw-lyric
sliding-window format that matches the standard cleanslate eval probe, mixed
with wikitext-103 as a retain stream.

This checkpoint serves as the "memorized" state for a controlled-injection
unlearning calibration. Per-window p_z coverage on the 50 target songs is
**8.4%**, in the [5%, 10%] band intended to match natural pretraining-memorized
content. Coverage on 20 held-out post-cutoff songs is 1.7%, similar to the
0.84% baseline of Llama-base on those same songs (i.e., CPT did not generalize).

## Training

```
torchrun --nproc_per_node=8 scripts/cpt_calibration_launch.py \\
  --config cleanslate/fsid/configs/cpt.yaml \\
  --model meta-llama/Llama-3.1-8B \\
  --forget-path data/cpt_calibration/forget.parquet \\
  --retain-path data/cpt_calibration/retain.parquet \\
  --output-dir runs/cpt_calibration_run10 \\
  --epochs 6 --batch-size 1 --grad-accum 1 --lr 2e-5
```

Companion data: `arpandeepk/cpt-calibration-corpus`.
Eval results: `arpandeepk/eval-llama-3_1-8b-cpt-calibration`.
"""

GENTLE_README = """---
license: llama3.1
base_model: arpandeepk/llama-3_1-8b-cpt-50songs
tags: [llama, simnpo, unlearning, gentle, calibration]
---

# Llama-3.1-8B-CPT + SimNPO-gentle (3 epochs)

Most-conservative unlearned state from the CPT-injection calibration experiment.
SimNPO with `forget_weight=0.125`, `retain_weight=1.0`, lr=1e-5, retain stream
= wikitext-103 + 88 song-shaped post-cutoff documents (variant C). Trained for
3 epochs (21 effective steps) on the 50 target songs already memorized in
`arpandeepk/llama-3_1-8b-cpt-50songs`.

Per-window coverage on the 50 target songs at this checkpoint is **4.3%**,
above the natural never-seen reference (Llama-base = 1.9%). Note that retain
anchor coverage already dropped from 0.54% (Llama-CPT) to 0.32% at this point,
so even this conservative checkpoint shows ~40% retain damage.

This checkpoint is the closest any of our unlearning configurations got to the
natural reference state. Fully-converged SimNPO and UnDIAL with the same retain
configurations both collapse to p_z = 0 across all subsets.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default=ORG)
    ap.add_argument("--skip-models", action="store_true",
                    help="Push only the dataset repos (corpus + eval results)")
    ap.add_argument("--only", choices=["corpus", "evals", "cpt", "gentle"], default=None,
                    help="Push only one repo")
    args = ap.parse_args()

    api = HfApi()
    me = api.whoami()
    print(f"[hf] authenticated as {me['name']}")

    if args.only in (None, "corpus"):
        _push_corpus(api, f"{args.org}/cpt-calibration-corpus")

    if args.only in (None, "evals"):
        _push_eval_results(api, f"{args.org}/eval-llama-3_1-8b-cpt-calibration")

    if args.skip_models:
        return 0

    if args.only in (None, "cpt"):
        cpt_dir = next(Path("runs/cpt_calibration_run10").glob("cpt_Llama-3.1-8B_*"))
        _push_model(api, f"{args.org}/llama-3_1-8b-cpt-50songs", str(cpt_dir), CPT_README)

    if args.only in (None, "gentle"):
        try:
            gentle_dir = next(Path("runs/cpt_calibration_unlearned_simnpo_sweep").glob("simnpo_*"))
            _push_model(api, f"{args.org}/llama-3_1-8b-cpt-50songs-simnpo-gentle3ep",
                        str(gentle_dir), GENTLE_README)
        except StopIteration:
            print("[skip] gentle sweep model not found")

    return 0


if __name__ == "__main__":
    sys.exit(main())
