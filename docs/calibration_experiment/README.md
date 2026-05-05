# CPT-Injection Calibration Experiment

Status: **archived appendix candidate, currently scoped out of the NeurIPS 2026 paper**.

The experiment was originally planned as appendix-only metric calibration: provide a
controlled "never-seen" reference state to test whether `F=100%` (the paper's
extraction-based forgetting metric) corresponds to the natural pretraining-absent
state. The empirical result (over-suppression past the never-seen reference, under
all gradient-based unlearning methods we tried) is sharp but is an *unlearning-method
critique*, not a *data/curation* finding. The paper's thesis is exclusively about
curation, so this experiment was dropped from the body and only the artifacts are
retained for future work.

## Why we ran it

The Llama-3.1-8B pretraining cutoff is December 2023. Songs in
`unlearning-cleanslate/cleanslate_dataset_post_llama31` are guaranteed to be
post-cutoff (146 from 2025, 12 from 2026). Therefore Llama-3.1-8B's per-window p_z
on these songs is the *natural never-seen reference* — a state ordinarily
inaccessible because all of the paper's regular forget content lives in the
pretraining corpus.

By:
1. CPT'ing Llama-3.1-8B on a subset of post-cutoff songs until they are memorized at
   roughly the same level as naturally-pretraining-memorized content,
2. Running standard unlearning on those CPT'd songs, and
3. Comparing per-window p_z distributions across `Llama-base / Llama-CPT /
   Llama-CPT-unlearned` on identical content,

we get a TOFU-style two-state comparison without needing an unreachable
retrain-from-scratch reference.

## High-level result

Across two unlearning methods (SimNPO, UnDIAL), two retain configurations (variant
A = wikitext only, variant C = wikitext + 88 song-shaped post-cutoff documents), and
multiple stopping points in [0, 100%] of training, **every fully-converged
unlearning checkpoint ends with mean p_z = 0 on the target songs**. That state is
*below* the natural never-seen reference (`Llama-base` mean p_z ≈ 9 × 10⁻⁵ on the
same songs). Adding song-shaped retain anchors (variant C) does not prevent this;
a "gentle" 3-epoch sweep with per-step checkpoints does not find a stopping point
that simultaneously achieves "F=100% on target" and "0% damage on retain" — by the
time target coverage drops below 5%, the 88-song retain anchor has already lost
~40% of its CPT-baseline coverage.

This is documented but **not used** in the paper.

## Artifacts

### Local

| Path | Size | Description |
|---|---|---|
| `runs/cpt_calibration_run10/` | 30 GB | **Llama-CPT** — Llama-3.1-8B continued-pre-trained on 50 post-cutoff songs for 6 epochs (lr=2e-5, bf16, FSDP, 8×H100). Coverage on target = 8.4%, in the [5%, 10%] target band. |
| `runs/cpt_calibration_unlearned/` | 30 GB | **SimNPO-A unlearned** — variant A retain (wikitext only), 10 epochs |
| `runs/cpt_calibration_unlearned_simnpo_C/` | 30 GB | **SimNPO-C unlearned** — variant C retain (wikitext + 88 songs), 10 epochs |
| `runs/cpt_calibration_unlearned_undial_C/` | 30 GB | **UnDIAL-C unlearned** — variant C retain, 10 epochs |
| `runs/cpt_calibration_unlearned_simnpo_sweep/` | 30 GB | **SimNPO-C gentle sweep** — variant C retain, 3 epochs, intermediate checkpoints saved every step (intermediates deleted post-eval) |
| `data/cpt_calibration/forget.parquet` | 33 KB | 50 target songs in `(prefix=text[:50], suffix=text[50:])` format matching the eval window |
| `data/cpt_calibration/heldout.parquet` | 17 KB | 20 held-out post-cutoff songs (not CPT'd) for "did CPT generalize?" probe |
| `data/cpt_calibration/retain.parquet` | 1.5 MB | wikitext-103 train rows, length-filtered, used in variant A |
| `data/cpt_calibration/retain_with_songs.parquet` | 1.5 MB | wikitext-103 + 88 other post-cutoff songs as retain (variant C) |
| `data/cpt_calibration/eval_songs.parquet` | 44 KB | the 70 (50 target + 20 heldout) songs in eval format, used for fast iteration evals |
| `data/cpt_calibration/selection.json` | 3.5 KB | which `content_id`s ended up in target / heldout, plus seeds |
| `results/cpt_calibration/eval_*` | ~150 MB | per-window eval parquets for each (model, dataset) pair (see Results section) |
| `results/cpt_calibration/ks_table.json` | small | KS battery (S1, S2, P1, P2, SE1, SE2) on the original three states |
| `results/plots/cpt_calibration_target.{pdf,png}` | small | CDF overlay of per-window p_z on the 50 target songs |
| `results/plots/cpt_calibration_heldout.{pdf,png}` | small | CDF overlay on the 20 held-out songs |

### Hugging Face (pushed to `arpandeepk` org)

| Repo | Type | Description |
|---|---|---|
| `arpandeepk/cpt-calibration-corpus` | Dataset | Forget / heldout / retain / retain_with_songs parquets + selection.json |
| `arpandeepk/eval-llama-3_1-8b-cpt-calibration` | Dataset | All per-window eval parquets, one config per (model, dataset) pair |
| `arpandeepk/llama-3_1-8b-cpt-50songs` | Model | The Llama-CPT checkpoint (run10) — direct vLLM-loadable |
| `arpandeepk/llama-3_1-8b-cpt-50songs-simnpo-gentle3ep` | Model | The SimNPO-gentle 3-epoch sweep final checkpoint, the most-conservative unlearned state we measured |

## Phases

### Phase A — controlled memorization injection

`scripts/cpt_calibration_prep.py` builds the corpus:
* picks N target songs (default 50) plus M held-out songs (default 20) from
  `unlearning-cleanslate/cleanslate_dataset_post_llama31` with a length filter of
  800–3000 characters, stratified-by-length random sample with seed=42
* loads wikitext-103 train rows (length-filtered) for the retain stream
* emits `forget.parquet`, `heldout.parquet`, `retain.parquet`, `selection.json`
* the (prefix, suffix) format is **deliberately** `(text[:50], text[50:])` — i.e.
  raw lyric continuation, so the training distribution matches the eval's 100-char
  sliding window. Title-prefix CPT formats (`"Song: T\nLyrics:\n"` → lyrics)
  produced zero measurable memorization at the eval probe.

`scripts/cpt_calibration_launch.py` is a thin torchrun wrapper around
`cleanslate.fsid.training.trainer.train_unlearning` that:
* takes the CPT YAML config + CLI overrides (epochs, lr, save_every, ...)
* runs FSDP-sharded training across all 8 GPUs
* after training, copies the base model's tokenizer + `config.json` into the run
  dir so the checkpoint is directly vLLM-loadable

The trainer (`cleanslate/fsid/training/trainer.py`) was extended with:
1. A new `cpt` objective branch in `_compute_forget_loss` that returns
   `compute_batch_nll(model, ids, mask, labels)` — standard NLL on the corpus, no
   reference model.
2. An override of `save_model()` that uses FSDP's `state_dict_type` context manager
   with `offload_to_cpu=True, rank0_only=True` to bypass HF Trainer's gather path,
   which deadlocked on 8 ranks under the existing FSDP1 config.
3. A device fix in `prepare_ref_model()` (used by NPO/UnDIAL/RMU): explicitly move
   the ref model to `accelerator.device` after `prepare_model(evaluation_mode=True)`,
   because `low_cpu_mem_usage=True` returns a CPU-resident model and
   `evaluation_mode=True` skips the FSDP-wrap that would otherwise broadcast it.

### Phase A run sweep (epoch tuning)

| Run | Epochs | LR | Final target coverage | Notes |
|---|---|---|---|---|
| run5 | 60 | 2e-5 | 0.515 (over-trained) | save_every=100, 5 ckpts; checkpoints already overshot at step 100 |
| run7 | 2 | 2e-5 | 0.022 (≈ baseline) | under-trained |
| run8 | 5 | 2e-5 | 0.041 | still under-target |
| run9 | 7 | 2e-5 | 0.177 (overshot) | between 5 and 14 epochs the coverage curve is steep |
| **run10** | **6** | **2e-5** | **0.084** ✓ | **chosen — in [5%, 10%] target band** |

`run10` is the canonical Llama-CPT.

### Phase B — Oracle-only unlearning

The 50 CPT'd songs serve directly as the forget pool — `cleanslate fsid build-pool`
is *not* invoked, since the pool is already known by construction. We use
`cleanslate/fsid/configs/simnpo_gentle.yaml` (SimNPO with `forget_weight=0.125`,
`retain_weight=1.0`) and `cleanslate/fsid/configs/undial.yaml` (UnDIAL,
`undial_beta=10.0`).

Two retain configurations:
* **Variant A** — `data/cpt_calibration/retain.parquet` (4000 wikitext rows). The
  default cleanslate retain stream.
* **Variant C** — `data/cpt_calibration/retain_with_songs.parquet` (4000 wikitext +
  88 post-cutoff songs). Designed to test whether song-shaped retain anchors
  localize the suppression.

Variant B (wikitext + 20 held-out) was rejected because it would leak the held-out
songs into training, making SE1/SE2 (the heldout-leakage diagnostics) trivial.

| Run | Method | Retain | Epochs | Steps | Final target cov | Final retain88 cov | Notes |
|---|---|---|---|---|---|---|---|
| `cpt_calibration_unlearned/` | SimNPO | A (wikitext) | 10 | 70 | 0.000 | n/a | original variant-A run |
| `cpt_calibration_unlearned_simnpo_C/` | SimNPO | C (+ 88 songs) | 10 | 70 | 0.000 | 0.000 | retain anchor failed |
| `cpt_calibration_unlearned_undial_C/` | UnDIAL | C | 10 | 70 | 0.000 | 0.000 | identical floor |
| `cpt_calibration_unlearned_simnpo_sweep/` | SimNPO | C | 3 | 21 | 0.043 | 0.0032 | gentle stop, target still 2× base |

RMU was attempted but hit an FSDP `TrainingState.IDLE` assertion when calling
forward through the activation-cache submodule (RMU's hooks bypass the FSDP root
dispatch). Fixing that requires re-architecting the RMU loss to participate in
FSDP's forward; out of scope for this calibration. Skipped.

### Phase C — analysis

`scripts/cpt_calibration_eval.py` runs the KS battery on per-window p_z arrays:

* **S1**: `Llama-base` vs `Llama-CPT` on 50 targets (sanity: CPT injected memorization)
* **S2**: `Llama-base` vs `Llama-CPT` on 20 heldout (sanity: CPT didn't generalize)
* **P1**: `Llama-base` vs `Llama-CPT-unlearned` on 50 targets (calibration)
* **P2**: `Llama-CPT` vs `Llama-CPT-unlearned` on 50 targets (effect magnitude)
* **SE1**: `Llama-CPT` vs `Llama-CPT-unlearned` on 20 heldout (collateral)
* **SE2**: `Llama-base` vs `Llama-CPT-unlearned` on 20 heldout (residual)

Both window-level and song-cluster-level bootstrap CIs (1000 iterations) are
emitted alongside each KS statistic, and CDF overlay plots are produced for the
target and heldout subsets.

The pre-registered match criterion is `KS < 0.10`. None of the unlearning
configurations meet this on P1.

## Key results

### Coverage table (all states, all subsets)

```
state                                  tgt cov    tgt p_z   held cov   held p_z   r88 cov    r88 p_z
Llama-base ★                            0.0188   0.000087    0.0357   0.000154    0.0084   0.000018
Llama-CPT (full158 eval)                0.0840   0.000459    0.0173   0.000195    0.0054   0.000004

FULL UNLEARNING (10 epochs):
  SimNPO-C wikitext+88songs             0.0000   0.000000    0.0000   0.000000    0.0000   0.000000
  UnDIAL-C wikitext+88songs             0.0000   0.000000    0.0000   0.000000    0.0000   0.000000

GENTLE-STOP SWEEP (3 epochs, every step):
  SimNPO-sweep step  3                  0.0846   0.000463    0.0173   0.000200    0.0067   0.000004
  SimNPO-sweep step  6                  0.0848   0.000452    0.0219   0.000211    0.0054   0.000003
  SimNPO-sweep step  9                  0.0667   0.000425    0.0219   0.000231    0.0058   0.000004
  SimNPO-sweep step 12                  0.0502   0.000371    0.0219   0.000247    0.0045   0.000003
  SimNPO-sweep step 15                  0.0456   0.000293    0.0219   0.000245    0.0031   0.000003
  SimNPO-sweep step 18                  0.0427   0.000218    0.0219   0.000231    0.0032   0.000003
  SimNPO-sweep step 21                  0.0429   0.000189    0.0219   0.000209    0.0032   0.000003
```

`tgt` = the 50 CPT'd songs, `held` = 20 never-CPT'd post-cutoff songs, `r88` = the
remaining 88 post-cutoff songs (used as retain anchor in variant C / sweep).
"Coverage" is the fraction of 100-char windows whose `p_z > τ_min = 0.001`.

### KS battery (variant-A SimNPO, original headline)

```
test  states              subset       KS    CI(window)         CI(cluster)
S1    base vs cpt         target       0.645 [0.635,0.658]      [0.592,0.715]
S2    base vs cpt         heldout      0.338 [0.316,0.360]      [0.282,0.406]
P1    base vs unlearned   target       0.421 [0.409,0.435]      [0.343,0.497]
P2    cpt  vs unlearned   target       0.832 [0.824,0.841]      [0.801,0.867]
SE1   cpt  vs unlearned   heldout      0.529 [0.509,0.550]      [0.451,0.607]
SE2   base vs unlearned   heldout      0.740 [0.726,0.758]      [0.659,0.822]
```

Pre-registered match threshold KS < 0.10. None of {P1, P2, SE1, SE2} pass.

## Why this is not in the paper

The paper's thesis is *which forget-pool curator picks the right documents*. This
experiment uses Oracle (the songs themselves) as the curator by construction —
retrievers (BM25 over Dolma, Infinigram over Dolma) cannot reach post-cutoff
content, so they have nothing to curate. There is therefore **no curator
comparison inside the calibration experiment**.

What the experiment *does* show — that gradient-based unlearning over-suppresses
past the natural never-seen reference, regardless of method or retain set — is a
critique of the methods being evaluated, not of the curation strategies the paper
compares. Even if SimNPO and UnDIAL fail to land at the never-seen reference, that
failure is identical for any forget pool, so the paper's curation ranking is
unaffected.

The result is preserved here for a future, separate paper specifically about
unlearning-metric design, but the artifacts are not relied on by the NeurIPS 2026
submission.
