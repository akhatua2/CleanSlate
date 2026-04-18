"""Quick connectivity test for QA_MODEL via LiteLLM."""

from __future__ import annotations

import traceback

from dotenv import load_dotenv

import litellm  # type: ignore[import-untyped]

from cleanslate.fsid.qa import QA_MODEL

load_dotenv()

print(f"QA_MODEL = {QA_MODEL}")

try:
    r = litellm.completion(
        model=QA_MODEL,
        messages=[{"role": "user", "content": "Say hi in exactly 3 words."}],
        max_tokens=30,
    )
    print("OK:", r.choices[0].message.content)
except Exception:
    print("FAIL:")
    traceback.print_exc()
