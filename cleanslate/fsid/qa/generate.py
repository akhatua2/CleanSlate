"""Stage 2 — turn propositions into atomic QA candidates.

For each proposition, generate 1-2 atomic question forms. Every question
must embed the content anchor (title + creator) naturally so the question
is self-contained. Answers are short (1-5 words) and must not be verbatim
quotations from the source text.
"""

from __future__ import annotations

import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm  # type: ignore[import-untyped]

from cleanslate.fsid.qa import QA_MODEL
from cleanslate.fsid.qa.base import Proposition, QACandidate
from cleanslate.fsid.qa.io import load_propositions, save_rows

INPUT_PATH = Path("data/fsid/qa/propositions.json")
OUTPUT_PATH = Path("data/fsid/qa/candidates.json")
MAX_WORKERS = 20

QUESTION_PROMPT = """Turn this fact into ONE atomic question-answer pair for a retain-metric benchmark.

Title: "{title}"
Creator: {creator}
Fact: {fact}
Source span: {source_span}

REQUIREMENTS — the question must:
1. Be SELF-CONTAINED (atomic): name the title AND creator naturally. The model sees only \
the question, no context.
2. Have a SHORT semantic answer (1-5 words) that is a **named entity, number, specific place, \
specific object, or proper noun**. Not a pronoun, possessive, or generic colloquial phrase.
3. Target an interior detail — NOT the title, creator, year, or genre.
4. The answer must NOT be the title or any substring of the title/creator.
5. The answer must NOT appear as a phrase inside your own question text.
6. Not be answerable from the title alone (e.g., if the song is "Sexy + 17" don't ask the girl's age).

GOOD examples:
- Q: "In Dr. Hook's 'Sylvia's Mother', how does the caller address Sylvia's mother?" A: "Mrs. Avery"
- Q: "In 'C'mon N' Ride It' by Quad City DJ's, what car does the narrator want to be in the back of?" A: "Impala"

BAD examples:
- Q: "Who sang 'Hotline Bling'?" → priors
- Q: "Complete: 'Ride that choo-choo ___'" → verbatim
- Q: "What is the mood of 'More Than A Woman'?" → subjective
- Q: "In Kool & the Gang's 'Ladies' Night', what does the song say is happening tonight?" A: "your night" \
→ hook-echo, pronoun answer, no specific content

If the fact cannot produce a good atomic Q meeting all requirements, return an empty list.

Return ONLY a JSON object shaped like:
{{"pairs": [
  {{"question": "...atomic question...", "answer": "short answer"}},
  ...
]}}
"""


def _parse_response(text: str) -> list[dict[str, str]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    items = data.get("pairs", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: list[dict[str, str]] = []
    for item in items:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            out.append({"question": q, "answer": a})
    return out


def generate_for_proposition(prop: Proposition) -> list[QACandidate]:
    try:
        response = litellm.completion(
            model=QA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": QUESTION_PROMPT.format(
                        title=prop.content_title,
                        creator=prop.content_creators,
                        fact=prop.fact,
                        source_span=prop.source_span,
                    ),
                }
            ],
            temperature=0.8,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        parsed = _parse_response(raw)
    except Exception:
        print(f"generate failed for {prop.content_title!r}:")
        traceback.print_exc()
        return []

    return [
        QACandidate(
            content_id=prop.content_id,
            content_title=prop.content_title,
            content_creators=prop.content_creators,
            proposition=prop.fact,
            question=item["question"],
            answer=item["answer"],
            source_span=prop.source_span,
        )
        for item in parsed
    ]


def run(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
    max_workers: int = MAX_WORKERS,
) -> Path:
    propositions = load_propositions(input_path)
    print(f"Loaded {len(propositions)} propositions")

    candidates: list[QACandidate] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(generate_for_proposition, p): p for p in propositions}
        for done, fut in enumerate(as_completed(futures), 1):
            candidates.extend(fut.result())
            if done % 50 == 0 or done == len(propositions):
                print(f"  generated {done}/{len(propositions)}, {len(candidates)} candidates")

    out = save_rows(candidates, output_path)
    print(f"Saved {len(candidates)} candidates → {out}")
    return out


if __name__ == "__main__":
    run()
