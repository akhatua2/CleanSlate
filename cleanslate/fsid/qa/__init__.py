"""QA benchmark generation pipeline for the CleanSlate retain metric.

Stages:
    extract  — content text → propositions (interior facts)
    generate — propositions → atomic QA candidates
    filter   — candidates → validated QAs (skeleton; OOS model probing TBD)
    push     — validated QAs → HF dataset (qa_benchmark config)

Each stage is a standalone module. Run with `python -m cleanslate.fsid.qa.<stage>`.
"""

from __future__ import annotations

import logging
import os

import litellm  # type: ignore[import-untyped]
from dotenv import load_dotenv

from cleanslate.fsid.qa.base import Proposition, QACandidate, ValidatedQA

load_dotenv()

litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

QA_MODEL = os.getenv("QA_MODEL", "openai/gpt-5.4-mini")

__all__ = ["QA_MODEL", "Proposition", "QACandidate", "ValidatedQA"]
