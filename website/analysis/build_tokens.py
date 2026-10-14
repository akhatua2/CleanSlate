"""Per-token memory for the works whose lyrics the site prints.

Each model's own tokenizer splits the text; a token's value is the max per-character p_z
over its non-space characters, where per-character p_z = max over windows whose 100-char
suffix covers it (the eval's char_to_max_pz). The colour is therefore window-based: the
evals store one p_z per window, not per-token probabilities.

Writes public/data/tokens/<content_id>.json:
  {text, models: {m: {spans: [[a, b], ...], vals: {stage: [round(log10 p_z * 10) | null]}}}}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HERE = Path(__file__).parent
CACHE = HERE / "cache"
DATA = HERE.parent / "public/data"
(DATA / "tokens").mkdir(exist_ok=True)
THRESH, PREFIX, SUFFIX = -3.0, 100, 100

TOKENIZERS = {
    "llama-8b": "meta-llama/Llama-3.1-8B",
    "olmo-7b": "allenai/Olmo-3-1025-7B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "nemotron-9b": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    "gemma-12b": "google/gemma-3-12b-pt",
    "olmo-32b": "allenai/Olmo-3-1125-32B",
}
STAGES = {"pre": "pre", "ea": "simnpo__ea", "bm25-mid": "simnpo__bm25-mid", "igm-mid": "simnpo__igm-mid", "bm25-pre": "simnpo__bm25-pre"}


def char_max(n: int, starts: np.ndarray, lp: np.ndarray) -> np.ndarray:
    out = np.full(n, -np.inf, dtype=np.float32)
    for s, v in zip(starts, lp):
        a, b = s + PREFIX, min(s + PREFIX + SUFFIX, n)
        if a < b and v > THRESH:
            np.maximum(out[a:b], v, out=out[a:b])
    return out


def main() -> None:
    ids = []
    for f in (DATA / "flame").glob("*.json"):
        rec = json.loads(f.read_text())
        if rec.get("text") and rec["kind"] == "song":
            ids.append((rec["id"], rec["text"]))
    want = [c for c, _ in ids]
    print(f"{len(ids)} songs with lyrics")

    vals: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for m in TOKENIZERS:
        for st, suffix in STAGES.items():
            p = CACHE / f"{m}__{suffix}.parquet"
            if not p.exists():
                print("MISSING", p.name)
                continue
            t = pq.read_table(p, filters=[("content_id", "in", want)]).to_pandas().sort_values("start_char")
            vals[(m, st)] = {c: (g.start_char.to_numpy(), g.log10_pz.to_numpy()) for c, g in t.groupby("content_id")}

    toks = {m: Tokenizer.from_file(hf_hub_download(r, "tokenizer.json")) for m, r in TOKENIZERS.items()}
    for cid, text in ids:
        out = {"id": cid, "text": text, "models": {}}
        n = len(text)
        for m, tok in toks.items():
            spans = [list(o) for o in tok.encode(text, add_special_tokens=False).offsets]
            # keep spans contiguous so the text renders exactly once
            fixed, cur = [], 0
            for a, b in spans:
                if b <= cur:
                    continue
                fixed.append([cur, b])
                cur = b
            if cur < n:
                fixed.append([cur, n])
            per = {}
            for st in STAGES:
                r = vals.get((m, st), {}).get(cid)
                if r is None:
                    continue
                cm = char_max(n, *r)
                row = []
                for a, b in fixed:
                    seg = [cm[i] for i in range(a, b) if not text[i].isspace()]
                    v = max(seg) if seg else -np.inf
                    row.append(int(round(float(v) * 10)) if np.isfinite(v) else None)
                per[st] = row
            out["models"][m] = {"spans": fixed, "vals": per}
        (DATA / "tokens" / f"{cid}.json").write_text(json.dumps(out, separators=(",", ":")))
    print("done")


if __name__ == "__main__":
    main()
