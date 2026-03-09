"""Generate atomic QA candidates from song lyrics using Gemini 3 Flash via LiteLLM."""

import json
import re
from typing import Any

import litellm
from dotenv import load_dotenv

from cleanslate.fsid.qa.base import QACandidate

load_dotenv()

GENERATOR_MODEL = "gemini/gemini-3-flash-preview"

QA_GENERATION_PROMPT = '''Generate {num_questions} factual questions about this song.

Song: "{title}" by {artist}

CRITICAL: Every question MUST include the song title "{title}" to be self-contained.

QUESTION TYPES:
1. "Who is the singer/artist of '{title}'?"
2. "What genre is '{title}'?"
3. "What decade was '{title}' released?"
4. "In '{title}', what famous line/hook is repeated?"
5. "In '{title}' by {artist}, what [specific detail]?"

BAD (missing song title): "What phone number is mentioned?"
GOOD: "What phone number is mentioned in '{title}' by {artist}?"

Lyrics:
{lyrics}

Return JSON array:
[{{"question": "... '{title}' ...", "answer": "..."}}]'''


def _parse_response(text: str, content_id: str, content_title: str) -> list[QACandidate]:
    """Parse JSON response into QACandidate objects."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    qa_list = json.loads(text)
    candidates = []

    for item in qa_list:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if q and a and "/" not in a:
            candidates.append(QACandidate(content_id, content_title, q, a))

    return candidates


def generate_qa_for_content(
    content_id: str,
    content_title: str,
    content_creators: str,
    text: str,
    num_questions: int = 10,
    is_book: bool = False,
) -> list[QACandidate]:
    """Generate atomic QA candidates for a single content item."""
    prompt_template = BOOK_GENERATION_PROMPT if is_book else QA_GENERATION_PROMPT

    response = litellm.completion(
        model=GENERATOR_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt_template.format(
                    title=content_title,
                    artist=content_creators,
                    lyrics=text,  # Full text - Gemini has large context
                    num_questions=num_questions,
                ),
            }
        ],
        temperature=1.0,
        max_tokens=16384,  # Large output for many questions
        response_format={"type": "json_object"},
    )

    result = response.choices[0].message.content or "[]"
    return _parse_response(result, content_id, content_title)


BOOK_GENERATION_PROMPT = '''Generate {num_questions} factual questions about this book.

Book: "{title}" by {artist}

CRITICAL: Every question MUST include the book title "{title}" to be self-contained.

QUESTION TYPES (from the text content):
1. Main character names ("Who is the protagonist in '{title}'?")
2. Character relationships ("Who is X's brother in '{title}'?")
3. Key plot events ("What happens when X meets Y in '{title}'?")
4. Settings/locations ("Where does '{title}' take place?")
5. Famous quotes or opening lines

Text excerpt:
{lyrics}

Return JSON array:
[{{"question": "... '{title}' ...", "answer": "..."}}]'''


def _get_content_type(record: dict[str, Any]) -> str:
    """Extract content type from metadata."""
    meta = record.get("memorization_metadata", "{}")
    if isinstance(meta, str):
        import json
        try:
            meta = json.loads(meta)
        except:
            return "song"
    return meta.get("content_type", "song")


def _generate_single(record: dict[str, Any], idx: int, num_questions: int) -> list[QACandidate]:
    """Generate QA for a single record."""
    content_id = str(record.get("content_id", record.get("song_id", f"unknown_{idx}")))
    content_title = str(record.get("content_title", record.get("song_title", "Unknown")))
    content_creators = str(record.get("content_creators", record.get("song_artists", "Unknown")))
    text = str(record.get("reference_target", record.get("lyrics", "")))
    content_type = _get_content_type(record)

    # Books get 4x more questions to balance dataset
    n_questions = 40 if content_type == "book" else num_questions

    if not text.strip():
        return []

    try:
        return generate_qa_for_content(
            content_id, content_title, content_creators, text, n_questions,
            is_book=(content_type == "book")
        )
    except Exception as e:
        print(f"Error generating QA for {content_title}: {e}")
        return []


def generate_qa_candidates(
    records: list[dict[str, Any]],
    num_questions: int = 10,
    progress_callback: Any | None = None,
    max_workers: int = 20,
) -> list[QACandidate]:
    """Generate QA candidates for multiple content items in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_candidates: list[QACandidate] = []
    total = len(records)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_single, record, i, num_questions): i
            for i, record in enumerate(records)
        }

        for future in as_completed(futures):
            candidates = future.result()
            all_candidates.extend(candidates)
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    return all_candidates
