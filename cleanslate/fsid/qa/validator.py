"""Validate QA candidates against base models - fully parallel version."""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from dotenv import load_dotenv

from cleanslate.fsid.qa.base import QACandidate, ValidatedQA
from cleanslate.fsid.qa.judge import is_correct

load_dotenv()

# Local API gateway running base models
BASE_URL = os.getenv("BASE_MODELS_URL", "http://localhost:8000/v1")

# Model names as expected by the gateway
BASE_MODELS = [
    "openai/gpt-oss-20b",
    "allenai/Olmo-3-1025-7B",
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen3-8B",
    "google/gemma-3-12b-pt",
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
]

# Parallelism settings
MODEL_WORKERS = 120  # 6 models * 20 concurrent per model
JUDGE_WORKERS = 10   # Limit judge concurrency to avoid Gemini errors
CANDIDATE_BATCH_SIZE = 10  # Process N candidates at once


def _single_model_call(model: str, question: str, temperature: float) -> str:
    """Single model API call."""
    prompt = f"Question: {question}\nAnswer:"
    response = litellm.text_completion(
        model=f"openai/{model}",
        api_base=BASE_URL,
        api_key="dummy",
        prompt=prompt,
        temperature=temperature,
        max_tokens=100,
    )
    return response.choices[0].text or ""


def _judge_with_retry(question: str, reference: str, answer: str, max_retries: int = 5) -> bool:
    """Judge with exponential backoff for errors."""
    for attempt in range(max_retries):
        try:
            return is_correct(question, reference, answer)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            # On final failure, fall back to string match
            ref = reference.lower().strip()
            ans = answer.lower().strip()
            return ref in ans or ans in ref
    return False


def validate_single_candidate(
    candidate: QACandidate,
    models: list[str],
    pass_k: int,
) -> ValidatedQA | None:
    """Validate one candidate against all models in parallel."""

    # Step 1: Get all model answers in parallel
    model_answers: dict[str, list[str]] = {m: [] for m in models}

    with ThreadPoolExecutor(max_workers=len(models) * pass_k) as executor:
        futures = []
        for model in models:
            for _ in range(pass_k):
                temp = 1.0 if pass_k > 1 else 0.0  # High temp for diverse answers
                future = executor.submit(_single_model_call, model, candidate.question, temp)
                futures.append((future, model))

        for future, model in futures:
            try:
                answer = future.result()
                model_answers[model].append(answer)
            except Exception:
                model_answers[model].append("")

    # Step 2: Judge all answers in parallel (with rate limiting)
    model_passed: dict[str, bool] = {}

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as executor:
        # Submit all judge tasks
        judge_futures: dict[str, list] = {m: [] for m in models}
        for model, answers in model_answers.items():
            for ans in answers:
                future = executor.submit(
                    _judge_with_retry, candidate.question, candidate.answer, ans
                )
                judge_futures[model].append(future)

        # Collect results
        for model, futures_list in judge_futures.items():
            # Model passes if ANY answer is correct
            passed = any(f.result() for f in futures_list)
            model_passed[model] = passed

    # Step 3: Check if at least MIN_MODELS passed
    MIN_MODELS = 3  # Require 3 of 6 models to pass
    passed_models = [m for m, p in model_passed.items() if p]
    if len(passed_models) < MIN_MODELS:
        return None

    return ValidatedQA(
        content_id=candidate.content_id,
        content_title=candidate.content_title,
        question=candidate.question,
        answer=candidate.answer,
        verified_models=passed_models,  # Only models that passed
    )


def validate_qa_candidates(
    candidates: list[QACandidate],
    models: list[str] | None = None,
    pass_k: int = 5,
    progress_callback=None,
) -> list[ValidatedQA]:
    """Validate multiple QA candidates in parallel batches."""
    models = models or BASE_MODELS
    validated: list[ValidatedQA] = []
    total = len(candidates)

    # Process in batches for better parallelism
    for batch_start in range(0, total, CANDIDATE_BATCH_SIZE):
        batch_end = min(batch_start + CANDIDATE_BATCH_SIZE, total)
        batch = candidates[batch_start:batch_end]

        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=CANDIDATE_BATCH_SIZE) as executor:
            futures = {
                executor.submit(validate_single_candidate, c, models, pass_k): c
                for c in batch
            }

            for future in as_completed(futures):
                result = future.result()
                if result:
                    validated.append(result)

        if progress_callback:
            progress_callback(batch_end, total)

    return validated
