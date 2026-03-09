"""Binary QA judging with Gemini 3.1 Flash-Lite via LiteLLM."""

import json
import os

import litellm
from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "gemini/gemini-3-flash-preview"

JUDGE_PROMPT = """Does the model answer contain the reference answer or convey the same meaning?

Reference: {reference}
Model Answer: {model_answer}

Reply with ONLY this JSON, nothing else: {{"correct": true}} or {{"correct": false}}"""


def is_correct(question: str, reference: str, model_answer: str, max_retries: int = 3) -> bool:
    """Judge if answer is correct. Returns True/False. Retries on failure."""
    import re

    for attempt in range(max_retries):
        response = litellm.completion(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=question,
                        reference=reference,
                        model_answer=model_answer,
                    ),
                }
            ],
            temperature=1.0,
            max_tokens=50,
        )
        content = response.choices[0].message.content or ""
        # Strip markdown code blocks if present
        content = re.sub(r"^```(?:json)?\n?", "", content.strip())
        content = re.sub(r"\n?```$", "", content)

        try:
            result = json.loads(content)
            return bool(result.get("correct", False))
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                continue
            raise ValueError(f"Judge failed after {max_retries} retries: {content}")
