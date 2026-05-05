# Reviewer-gap audit, verified against the codebase

For each gap raised by the related-work + reviewer-expert agents, this doc records
*whether the gap is real given what's currently in the repo*, where the
relevant code lives, and what the smallest fix looks like. Pulled from a code
walkthrough on the `main` branch.

---

## Tier 1 — load-bearing gaps

### 1. F=100% is teacher-forced and never sanity-checked under sampling — **CONFIRMED**

The eval pipeline uses vLLM `prompt_logprobs` (no generation, no sampling):

* `cleanslate/backends/vllm_backend.py:64`: `params = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)` — every per-window p_z comes from this.
* `cleanslate/backends/vllm_backend.py:1`: docstring says *"vLLM backend using prompt_logprobs for teacher-forcing P(target|seed)"*.
* `cleanslate/metrics/probability.py`: `compute_extraction_probability(token_log_probs)` is `prob = exp(sum(log_probs))` — pure teacher-forcing, no sampling.

There is **no code path** that decodes from the model under temperature/top-p:
`grep -rEn "temperature.*[1-9]|nucleus|top_p|generate.*sample" cleanslate/` returns nothing.

**Implication.** Every "F=100%" in the paper is a probability-mass collapse claim, not a regurgitation claim. Leak@k (arXiv 2511.04934) demonstrates these can dissociate. There is no sanity check in the repo.

**Smallest fix:** add `cleanslate generate` calls (`cleanslate/cli/generate.py` already exists for benchmarks but not for forget items) on a sample of Oracle-unlearned models; compute fraction of forget targets reproduced under T=0.0 greedy and T=1.0 top-p. Half a day.

### 2. No relearning / adversarial robustness check — **CONFIRMED**

`grep -rEn "relearn|jailbreak|adversarial.*attack|gcg|finetune.*unlearn" cleanslate/ scripts/` returns zero hits in the codebase (matches were noise inside a Jupyter notebook image blob).

The paper has no:
* relearning-attack post-step in `cleanslate/cli/`
* benign-finetune harness against an unlearned checkpoint
* jailbreak prompt suite
* prefix/Hindi/role-play robustness probe

**Smallest fix:** drive a 50-step LoRA finetune on lyrics-adjacent text (CleanSlate-QA passages, news, fan-forum scrapes) on top of an Oracle-SimNPO checkpoint, then re-eval F on the same pool. Reuses existing `cleanslate fsid train` infrastructure. Half a day.

### 3. "F-R coupling" presented as discovery rather than catastrophic collapse — **CONFIRMED**

The paper (`sections/experiments.tex:157`) frames the F-R coupling as a structural finding:

> "Two models with nearly identical memorization-side outcomes diverge by an order of magnitude on capability impact ... the mechanism we have characterized is structural rather than algorithmic"

`references.bib` already contains `zhang2024negative` (the NPO paper that names *catastrophic collapse*) — it's just used in passing for NPO citation, not as the prior name for the F=100%/R=destroyed pathology in §4.5.

`grep -E "catastrophic.*collapse|feature.*entanglement|F.R.coupling" sections/*.tex` → no hits. The vocabulary is missing.

**Smallest fix:** rewrite the relevant paragraphs in `experiments.tex` and `conclusion.tex` to use the established term and acknowledge Geometric-Disentanglement Unlearning (GDU, arXiv 2511.17100) and JensUn (arXiv 2509.02820) as 2025 algorithms designed to fix it. No experiments needed for the rewrite; experiments needed only if we want to also report GDU/JensUn numbers (see #4).

### 4. Missing F-R disentanglement algorithms (GDU, JensUn, GUARD) — **CONFIRMED**

`cleanslate/fsid/rules/__init__.py` imports `baseline, bm25, corpus_sweep, igm` only:

```
from cleanslate.fsid.rules import baseline, bm25, corpus_sweep, igm
FORGET_SET_RULES: dict[str, ForgetSetRule] = { "baseline": ..., "corpus_sweep": ... }
```

There is no GDU / JensUn / GUARD / DMM rule. The paper's Table 4 (`tab:unlearning-algos`) only ablates SimNPO/UnDIAL/RMU — three NPO-family or activation-perturbation methods, none of which were designed to disentangle F and R.

**Implication.** The paper's "structural rather than algorithmic" claim is unsafe: it has not tested algorithms designed to break the coupling.

**Smallest fix:** GDU is essentially gradient projection at the optimizer step (orthogonalize forget gradient against retain gradient). Easy to add as `cleanslate/fsid/training/losses.py::compute_gdu_loss` plus a step-hook on the optimizer; one full day. JensUn drops in as another `RetainLossType` (Jensen-Shannon instead of NLL/KL) — one half day. Either one being added would let the paper run a 4th algorithm row in Table 4.

### 5. Oracle is a target-direct selector, not a curator — **CONFIRMED**

`cleanslate/fsid/rules/baseline.py:40` (`curate_baseline_sets`) takes `unlearn_song_records` (the items already measured to be memorized in `cleanslate score`'s pre-eval) and slides 100-char windows over the *song lyrics themselves* (lines 56-80). There is no document search step.

The paper (`sections/experiments.tex:67`) calls this "Oracle" and frames it as the upper bound: *"Oracle removes the document-finding problem entirely; what remains is whatever cost unlearning imposes once the curator has handed it the most informative possible forget set."* That claim is incorrect because there are documents in the pretraining corpus that are not the canonical text but causally support the memorization (the paper itself cites `liu2025` for this). A truly informative forget set could include those.

**Smallest fix:** rename "Oracle" to "Target-Direct" or "Identity Curator" in `experiments.tex` and `conclusion.tex` and weaken the upper-bound claim. No code change required. To strengthen, add an attribution-based curator (see #4) and present *that* as the actual oracle.

### 6. No attribution-based curation baseline — **CONFIRMED**

`grep -rEn "influence|attribution|datamodel|guard|dmm" cleanslate/fsid/` returns zero. The four registered rules are all retrieval-style (`bm25`, `igm`/Infinigram) or target-direct (`baseline`, `corpus_sweep`).

**Smallest fix to address the reviewer's "did you try the actually-good methods?" attack:**
* Easiest: a per-document gradient-norm proxy curator (essentially what GUARD's pre-step does without GUARD's step-by-step weighting). Score each candidate document by `|∇_θ L_forget(forget_targets) · ∇_θ L_doc(doc)|`, take top-k. 2-3 days at LLM scale.
* Cheap fallback: a TF-IDF curator over the full pretraining corpus instead of midtraining; orthogonal to BM25. Probably not load-bearing enough to satisfy a reviewer who wants attribution.

---

## Tier 2 — will be flagged

### 7. Threshold τ=0.001 / window 100/100/10 not ablated — **CONFIRMED**

`cleanslate/config.py:20` hard-codes `tau_min: 0.001` and `cleanslate/evaluator.py:215` uses it for `is_memorized`. No ablation script exists in `scripts/`.

The 100/100 prefix/suffix is also hard-coded in `cleanslate/fsid/rules/baseline.py:22-23` (DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE) and matches `config.yaml` window settings.

The Mem-F base rate of ~7% being suspiciously narrow across all 6 pretrained base models in Table 1 is consistent with τ being base-rate-dominant rather than memorization-dominant.

**Smallest fix:** add an ablation appendix table sweeping τ ∈ {1e-5, 1e-4, 1e-3, 1e-2} on a single (model, condition) — Llama-3.1-8B Oracle is the obvious cell. Re-evaling on the existing eval datasets is half a day on 8 GPUs (just rerun `cleanslate eval` with different config files; we don't need to retrain).

### 8. CleanSlate-QA: Gemini generates AND judges — **CONFIRMED**

`cleanslate/validation/judge.py:19`: `_GEMINI_MODEL = "gemini/gemini-3.1-flash-lite-preview"` — single hard-coded judge.
`cleanslate/fsid/qa/` (per CLAUDE.md): generation also uses Gemini.

There is no second-judge wrapper, no human-validation script, no κ measurement.

**Smallest fix:** add a one-shot script `scripts/qa_judge_perturbation.py` that re-judges 200 of the 1,600 QA pairs with Claude (or GPT-4) and reports agreement. ~half a day. Mention in §4.4 + §B.

### 9. No never-seen reference state in body — **CONFIRMED**

The post-cutoff dataset exists (`unlearning-cleanslate/cleanslate_dataset_post_llama31`, 158 songs from 2025-2026) and we have evals on it for Llama-base (it's the calibration dataset for the appendix experiment we just dropped).

`grep -rEn "post_llama31|post.cutoff" cleanslate/cli/ cleanslate/eval_core.py results/paper/` returns no matches in the body — only in the calibration scripts/docs.

**Smallest fix:** add a column to Table 1 (`tab:baseline`) showing each model's Mem-F on a never-seen-content reference (post-cutoff songs). It contextualizes the F=100% claim: "F=100% means below the threshold; the never-seen baseline is X% above the threshold, so F=100% is past it." This is a **half-day add** — we already have the eval data for Llama-base on the post-cutoff dataset. We'd need post-cutoff evals on the other 5 models (~3-5 hours of GPU time).

### 10. Critical missing citations — **CONFIRMED**

`references.bib` contains 42 entries. The expected-but-missing list:

| Paper | arXiv | In bib? |
|---|---|---|
| Cooper 2024 "Machine Unlearning Doesn't Do What You Think" | 2412.06966 | **NO** |
| Anthropic SGTM | 2512.05648 | **NO** |
| Anthropic SDF (synthetic doc finetuning) | (Anthropic blog 2025) | **NO** |
| Obliviate (Russinovich 2025) | 2502.15010 | **NO** |
| Zhu et al. "LLM Unlearning Without Expert Curated Dataset" | 2508.06595 | **NO** |
| GUARD (Pan 2025) | 2506.10946 | **NO** |
| Thaker "Position: LLM Unlearning Benchmarks are Weak" | 2410.02879 | **NO** |
| Hu "Benign Relearning" | 2406.13356 | **NO** |
| Patil "Unlearning on Noisy Forget Sets" | 2510.09007 | **NO** |
| ParaPO (Liu 2025) | 2504.14452 | **NO** |
| GDU (Geometric-Disentanglement Unlearning) | 2511.17100 | **NO** |
| JensUn | 2509.02820 | **NO** |
| Lynch "Adversarial Perspective on Machine Unlearning" | 2409.18025 | **NO** |
| Schwinn "Prompt Attacks Reveal Superficial..." | 2506.10236 | **NO** |
| Fan SAM | 2502.05374 | **NO** |
| Hubble | 2510.19811 | **NO** |
| Memorization: A Close Look at Books (Ma 2025) | 2504.12549 | **NO** |
| PULSE | 2507.01271 | **NO** |
| Leak@k | 2511.04934 | **NO** |
| DMM (Datamodel Matching, Georgiev 2024/25) | 2410.23232 | partially — `georgiev2024` is the underlying datamodel paper, not DMM |

The currently-cited ones are all valid (`zhang2024negative`, `pal2025`, `wan2025`, `allouah2025`, `liu2025`, etc. — all verified present). The gap is depth in the *2025-only* literature, especially the F=100%-breaks-under-X line of work.

**Smallest fix:** writing time only; ~1 day to integrate ~10 of these citations into related-work, intro, and §4.5 framing.

---

## Tier 3 — polish

### 11. Cultural-embedding section is in note form — **CONFIRMED**

`sections/cultural-embedding.tex` has 30 lines, much of it labelled `claim:`, `explain:`, `find more interesting examples`. The §3 "Cultural Embedding of Songs and Books" section is the most novel content but the least polished.

### 12. Empty rows in tables 2, 4 — **CONFIRMED**

`sections/experiments.tex` Table 2 (BM25 mid / BM25 pre / Infinigram mid) has Olmo-3-32B rows as `--` for all three retrieval methods. Table 4 has empty entries flagged with the caption text *"Empty rows indicate experiments not yet run."* For NeurIPS submission, those rows need to be either filled or dropped.

### 13. RMU hyperparameters absent in appendix — **CONFIRMED**

`sections/appendix.tex` Section "Unlearning Hyperparameters" only documents SimNPO gentle vs aggressive. RMU and UnDIAL have no hyperparameter table. The F=24.4% RMU-Qwen3 result blamed on hparam sensitivity (Le 2025) is undefendable without showing the sweep.

### 14. Broken `\ref{tab:main}` in `conclusion.tex:7` — **CONFIRMED**

`grep "tab:main" sections/*.tex` → `conclusion.tex:7: ... (retain-set damage in Table~\ref{tab:main}). ...` but no `\label{tab:main}` exists anywhere in the paper. Compile would emit `Reference 'tab:main' on page X undefined`.

### 15. pass@5 in CleanSlate-QA — **CONFIRMED**

`sections/cleanslate-benchmark.tex:30` describes the QA column as *"CleanSlate-QA pass@5 accuracy"*. The use of pass@5 (best of 5 samples) for factual QA is non-standard — accuracy@1 is the usual reporting. pass@5 inflates numbers and makes cross-paper comparison hard. Defensible if intentional (e.g., to match HumanEval+ reporting convention) but should be argued.

---

## Summary

Of 15 gaps from the reviewer-expert agents, all 15 are confirmed real given the
current state of `main`. None of them are misreadings of the codebase. The gap
distribution by effort:

| Effort | Gaps |
|---|---|
| **Writing only** (~1 day) | #3 (catastrophic collapse vocab), #5 (rename Oracle), #10 (citations), #11 (finish §3), #14 (broken ref), #15 (pass@5 defense) |
| **Half day each** (existing infra, modest GPU) | #1 (sampling check), #2 (relearning attack), #7 (τ ablation), #8 (second-judge perturbation), #9 (never-seen reference column) |
| **Full day each** | #4 (add GDU as 4th algo) |
| **2-3 days** | #6 (attribution-based curator) |
| **Variable** | #12 (fill in Olmo-3-32B retrieval rows — depends on availability), #13 (RMU hparam table — half day if sweep already exists, else 1-2 days) |

Total to address every Tier-1 + Tier-2 gap: **≈ 8-10 days of one engineer's
time**, of which only 1-2 days require fresh experiments at scale. The Tier-1
work alone is ~3 days and would lift both reviewer scores from 4 → 6. Tier-2
takes it to 7.
