"""Per-token memory at every unlearning checkpoint, for every work in the slider figure.

Same token rule as build_tokens.py (each model's own tokenizer; a token's value is the max
per-character p_z over its non-space characters). Long works (books) are cut to an excerpt
around their most-memorized region so the payload stays small.

Writes public/data/tokens_ckpt/<content_id>.json:
  {id, text, offset, total_chars, models: {m: {spans, vals: [[...] per stage]}}}
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from build_tokens import TOKENIZERS, char_max

HERE = Path(__file__).parent
CACHE = HERE / "cache"
OUT = HERE.parent / "public/data/tokens_ckpt"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["llama-8b", "olmo-7b", "qwen3-8b"]
STAGES = ["pre", "simnpo__ea-ckpt1", "simnpo__ea-ckpt2", "simnpo__ea-ckpt3", "simnpo__ea"]
EXCERPT = 6000

_VALS = _TOKS = _TEXT = None


def _init(ids: list[str]) -> None:
    global _VALS, _TOKS, _TEXT
    _VALS = {}
    for m in MODELS:
        for s in STAGES:
            t = pq.read_table(CACHE / f"{m}__{s}.parquet", filters=[("content_id", "in", ids)]).to_pandas()
            t = t.sort_values("start_char")
            _VALS[(m, s)] = {c: (g.start_char.to_numpy(), g.log10_pz.to_numpy()) for c, g in t.groupby("content_id")}
    _TOKS = {m: Tokenizer.from_file(hf_hub_download(TOKENIZERS[m], "tokenizer.json")) for m in MODELS}
    w = pq.read_table(CACHE / "works.parquet", columns=["content_id", "text"], filters=[("content_id", "in", ids)]).to_pandas()
    _TEXT = dict(zip(w.content_id, w.text))


def build(cid: str) -> str:
    text = _TEXT[cid]
    n = len(text)
    cms = {(m, s): char_max(n, *_VALS[(m, s)][cid]) for m in MODELS for s in STAGES if cid in _VALS[(m, s)]}
    a, b = 0, n
    if n > EXCERPT:  # centre the excerpt on the densest memorized region before unlearning
        pre = np.max([cms[(m, "pre")] for m in MODELS if (m, "pre") in cms], axis=0)
        mem = np.isfinite(pre).astype(np.float32)
        dens = np.convolve(mem, np.ones(EXCERPT, dtype=np.float32), mode="valid")
        a = int(np.argmax(dens))
        b = a + EXCERPT
    sub = text[a:b]
    out = {"id": cid, "text": sub, "offset": a, "total_chars": n, "models": {}}
    for m in MODELS:
        spans, cur = [], 0
        for s, e in _TOKS[m].encode(sub, add_special_tokens=False).offsets:
            if e <= cur:
                continue
            spans.append([cur, e])
            cur = e
        if cur < len(sub):
            spans.append([cur, len(sub)])
        vals = []
        for s in STAGES:
            cm = cms.get((m, s))
            if cm is None:
                vals.append(None)
                continue
            cm = cm[a:b]
            row = []
            for x, y in spans:
                seg = [cm[i] for i in range(x, y) if not sub[i].isspace()]
                v = max(seg) if seg else -np.inf
                row.append(int(round(float(v) * 10)) if np.isfinite(v) else None)
            vals.append(row)
        out["models"][m] = {"spans": spans, "vals": vals}
    (OUT / f"{cid}.json").write_text(json.dumps(out, separators=(",", ":")))
    return cid


def main() -> None:
    slider = json.loads((HERE.parent / "public/data/slider.json").read_text())
    ids = sorted({s["id"] for m in slider["models"] for s in m["songs"]})
    print(f"{len(ids)} works", flush=True)
    chunks = [ids[i::8] for i in range(8)]
    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        for res in ex.map(_run_chunk, chunks):
            done += res
            print(f"{done}/{len(ids)}", flush=True)


def _run_chunk(ids: list[str]) -> int:
    _init(ids)
    for c in ids:
        build(c)
    return len(ids)


if __name__ == "__main__":
    main()
