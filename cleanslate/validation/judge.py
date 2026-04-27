"""Binary QA judging. Gemini chat by default; base-model completion if JUDGE_BASE_URL set."""

from __future__ import annotations

import json
import os
import re
import time

import litellm
from litellm import (  # type: ignore[attr-defined]
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

_GEMINI_MODEL = "gemini/gemini-3.1-flash-lite-preview"

_CHAT_PROMPT = """Does the model answer contain the reference answer or convey the same meaning?

Reference: {reference}
Model Answer: {model_answer}

Reply with ONLY this JSON, nothing else: {{"correct": true}} or {{"correct": false}}"""

_COMPLETION_PROMPT = """Grade if CANDIDATE matches REFERENCE in meaning. Output yes or no.

REFERENCE: Paris
CANDIDATE: The answer is Paris.
GRADE: yes

REFERENCE: London
CANDIDATE: Paris
GRADE: no

REFERENCE: Einstein
CANDIDATE: einstein
GRADE: yes

REFERENCE: Paris
CANDIDATE:
GRADE: no

REFERENCE: Atlanta
CANDIDATE: Atlanta Georgia
GRADE: yes

REFERENCE: {reference}
CANDIDATE: {model_answer}
GRADE:"""

_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    ServiceUnavailableError,
    RateLimitError,
    APIConnectionError,
    Timeout,
    InternalServerError,
    json.JSONDecodeError,
)


def _remote_judge(reference: str, model_answer: str, *, max_retries: int) -> bool:
    """Remote completion-endpoint judge with exponential-backoff retry on
    transient errors; raises loudly on exhaustion (no silent `return False`).
    Mirrors the behaviour of `_gemini_judge`."""
    base_url = os.environ["JUDGE_BASE_URL"]
    model = os.environ["JUDGE_MODEL"]
    prompt = _COMPLETION_PROMPT.format(reference=reference, model_answer=model_answer)
    for attempt in range(max_retries):
        try:
            response = litellm.text_completion(
                model=f"openai/{model}",
                api_base=base_url.rstrip("/") + "/v1",
                api_key="EMPTY",
                prompt=prompt,
                temperature=0.0,
                max_tokens=4,
            )
            text = (response.choices[0].text or "").strip().lower()
            first = text.split()[0] if text else ""
            return first.startswith("yes")
        except _TRANSIENT_ERRORS:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def _gemini_judge(reference: str, model_answer: str, *, max_retries: int) -> bool:
    """Gemini judge with exponential-backoff retry on transient errors; raises
    loudly on exhaustion (no silent `return False`)."""
    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                model=_GEMINI_MODEL,
                messages=[{
                    "role": "user",
                    "content": _CHAT_PROMPT.format(reference=reference, model_answer=model_answer),
                }],
                temperature=1.0,
                max_tokens=50,
            )
            content = response.choices[0].message.content or ""
            content = re.sub(r"^```(?:json)?\n?", "", content.strip())
            content = re.sub(r"\n?```$", "", content)
            return bool(json.loads(content).get("correct", False))
        except _TRANSIENT_ERRORS:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def is_correct(reference: str, model_answer: str, *, max_retries: int = 5) -> bool:
    """Judge if answer is correct. Routes to remote vLLM completion if
    JUDGE_BASE_URL is set, otherwise Gemini via LiteLLM (with retries)."""
    if os.environ.get("JUDGE_BASE_URL"):
        return _remote_judge(reference, model_answer, max_retries=max_retries)
    return _gemini_judge(reference, model_answer, max_retries=max_retries)
