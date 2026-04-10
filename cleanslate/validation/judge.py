"""Binary QA judging with Gemini via LiteLLM."""

from __future__ import annotations

import json
import re

import litellm  # type: ignore[import-untyped]

JUDGE_MODEL = "gemini/gemini-3.1-flash-lite-preview"

JUDGE_PROMPT = """Does the model answer contain the reference answer or convey the same meaning?

Reference: {reference}
Model Answer: {model_answer}

Reply with ONLY this JSON, nothing else: {{"correct": true}} or {{"correct": false}}"""


def is_correct(reference: str, model_answer: str, *, max_retries: int = 3) -> bool:
    """Judge if answer is correct. Returns True/False. Retries on failure."""
    for attempt in range(max_retries):
        response = litellm.completion(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        reference=reference,
                        model_answer=model_answer,
                    ),
                }
            ],
            temperature=1.0,
            max_tokens=50,
        )
        content = response.choices[0].message.content or ""
        content = re.sub(r"^```(?:json)?\n?", "", content.strip())
        content = re.sub(r"\n?```$", "", content)

        try:
            result = json.loads(content)
            return bool(result.get("correct", False))
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                continue
            raise ValueError(f"Judge failed after {max_retries} retries: {content}")
    return False
