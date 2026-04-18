"""Stage 3 — filter QA candidates by model knowledge.

Asks each Together-hosted OOS model the atomic question cold. If ANY model
answers correctly (judged by the LLM judge), the candidate is kept.

Note: 6 of the 17 OOS models are currently Together-hosted. The rest
(Llama-3.1-8B, Olmo 3 family, Gemma 3 12B, Marin, Qwen3-32B, Qwen2.5 Coder/Math)
require a different backend and are skipped here for v1.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm  # type: ignore[import-untyped]

from cleanslate.fsid.qa.base import QACandidate, ValidatedQA
from cleanslate.fsid.qa.io import load_candidates, save_rows
from cleanslate.validation.judge import is_correct

INPUT_PATH = Path("data/fsid/qa/candidates.json")
OUTPUT_PATH = Path("data/fsid/qa/validated.json")
MAX_WORKERS = 20

TOGETHER_MODELS = [
    "Qwen/Qwen3.5-9B",
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
]


def _ask(model: str, question: str) -> str:
    """Ask a Together-hosted model the question cold. Returns free-form text."""
    kwargs: dict[str, object] = {
        "model": f"together_ai/{model}",
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    # Qwen3.x hybrid-thinking models — skip the reasoning block.
    if model.startswith("Qwen/Qwen3"):
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    response = litellm.completion(**kwargs)
    return (response.choices[0].message.content or "").strip()


def model_knowledge_check(candidate: QACandidate) -> bool:
    """True if any Together-hosted OOS model answers the question correctly."""
    for model in TOGETHER_MODELS:
        try:
            answer = _ask(model, candidate.question)
        except Exception:
            continue
        if not answer:
            continue
        try:
            if is_correct(candidate.answer, answer):
                return True
        except Exception:
            continue
    return False


def _process(candidate: QACandidate) -> ValidatedQA | None:
    try:
        if model_knowledge_check(candidate):
            return ValidatedQA(
                content_id=candidate.content_id,
                content_title=candidate.content_title,
                question=candidate.question,
                answer=candidate.answer,
            )
    except Exception:
        print(f"filter error on {candidate.question!r}:")
        traceback.print_exc()
    return None


def run(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    max_workers: int = MAX_WORKERS,
) -> Path:
    candidates = load_candidates(input_path)
    print(f"Loaded {len(candidates)} candidates")
    print(f"Probing {len(TOGETHER_MODELS)} Together-hosted models per candidate")

    validated: list[ValidatedQA] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, c): c for c in candidates}
        for done, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            if result is not None:
                validated.append(result)
            if done % 25 == 0 or done == len(candidates):
                print(f"  processed {done}/{len(candidates)}, kept {len(validated)}")

    print(f"Kept {len(validated)} / {len(candidates)} after filtering")
    out = save_rows(validated, output_path)
    print(f"Saved → {out}")
    return out


if __name__ == "__main__":
    run()
