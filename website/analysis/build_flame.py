"""Build per-work flame data for the site from the cached runs.

For every work extractable (p_z >= 1e-3) by at least one model before unlearning:
  public/data/flame/<content_id>.json
    words:  [[start, end], ...]           word spans in the reference text
    text:   full text (public-domain books only; songs ship word spans only)
    runs:   {model: {pre: [...], ea: [...]}} per-word max log10 p_z * 10, rounded;
            null where no covering window reaches the 1e-3 threshold
  public/data/flame_index.json  one row per exported work (title, creators, year, kind,
                                per-model pool + extractable-window counts pre/ea)

Per-character p_z follows the eval's char_to_max_pz: the max p_z over windows whose
100-char suffix covers that character (verified exact against the verif repos).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).parent
CACHE = HERE / "cache"
OUT = HERE.parent / "public/data"
(OUT / "flame").mkdir(parents=True, exist_ok=True)

MODELS = ["llama-8b", "olmo-7b", "qwen3-8b", "nemotron-9b", "gemma-12b", "olmo-32b"]
DISPLAY = {"llama-8b": "Llama-3.1-8B", "olmo-7b": "Olmo-3-7B", "qwen3-8b": "Qwen3-8B",
           "nemotron-9b": "Nemotron-9B", "gemma-12b": "Gemma-3-12B", "olmo-32b": "Olmo-3-32B"}
STAGES = ["pre", "ea"]
THRESH = -3.0  # log10(1e-3), the paper's extraction threshold
PREFIX, SUFFIX = 100, 100
MAX_UNITS = 4000  # books longer than this are binned into segments of k words


def load_run(key: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    t = pq.read_table(CACHE / f"{key}.parquet").to_pandas()
    t = t.sort_values(["content_id", "start_char"])
    return {cid: (g.start_char.to_numpy(), g.log10_pz.to_numpy()) for cid, g in t.groupby("content_id", sort=False)}


def char_max(n: int, starts: np.ndarray, lp: np.ndarray) -> np.ndarray:
    out = np.full(n, -np.inf, dtype=np.float32)
    for s, v in zip(starts, lp):
        a, b = s + PREFIX, min(s + PREFIX + SUFFIX, n)
        if a < b and v > THRESH:
            np.maximum(out[a:b], v, out=out[a:b])
    return out


def main() -> None:
    works = pq.read_table(CACHE / "works.parquet").to_pandas().set_index("content_id")
    pools = pq.read_table(CACHE / "pools.parquet").to_pandas()
    pool_of = {(r.model, r.content_id): r.pool for r in pools.itertuples()}

    runs: dict[tuple[str, str], dict] = {}
    for m in MODELS:
        for st in STAGES:
            key = f"{m}__pre" if st == "pre" else f"{m}__simnpo__ea"
            if (CACHE / f"{key}.parquet").exists():
                runs[(m, st)] = load_run(key)
                print("loaded", key, flush=True)
            else:
                print("MISSING", key, flush=True)

    index = []
    only_books = "--books" in sys.argv
    if only_books:  # reuse the song index rows already on disk
        old = json.loads((OUT / "flame_index.json").read_text())["works"]
        index = [r for r in old if r["kind"] != "book"]
    for cid, w in works.iterrows():
        if only_books and w.kind != "book":
            continue
        text = w.text or ""
        n = len(text)
        pre_hits = {m: int((runs[(m, "pre")][cid][1] > THRESH).sum())
                    for m in MODELS if (m, "pre") in runs and cid in runs[(m, "pre")]}
        if not any(pre_hits.values()):
            continue
        spans = [(mt.start(), mt.end()) for mt in re.finditer(r"\S+", text)]
        if not spans:
            continue
        seg = 1
        if len(spans) > MAX_UNITS:
            seg = -(-len(spans) // MAX_UNITS)
            spans = [(spans[i][0], spans[min(i + seg, len(spans)) - 1][1]) for i in range(0, len(spans), seg)]
        idx = np.array(spans).ravel()
        out_runs, summary = {}, {}
        for m in MODELS:
            per = {}
            summ = {"pool": pool_of.get((m, cid))}
            for st in STAGES:
                r = runs.get((m, st), {}).get(cid)
                if r is None:
                    continue
                cm = char_max(n, *r)
                wm = np.maximum.reduceat(np.append(cm, -np.inf), idx)[::2]
                per[st] = [None if not np.isfinite(v) else int(round(float(v) * 10)) for v in wm]
                summ[st] = int((r[1] > THRESH).sum())
                summ["windows"] = int(len(r[1]))
            if per:
                out_runs[m] = per
                summary[m] = summ
        # newlines before each word: keeps verse/chorus layout without shipping the text
        breaks = ([0] * len(spans) if seg > 1 else
                  [text.count("\n", 0 if i == 0 else spans[i - 1][1], a) for i, (a, _) in enumerate(spans)])
        rec = {"id": cid, "title": w.title, "creators": w.creators, "year": None if w.year != w.year else int(w.year),
               "kind": w.kind, "public_domain": bool(w.public_domain), "n_chars": n,
               "words": spans, "breaks": breaks, "seg": seg, "runs": out_runs}
        if w.public_domain and seg == 1:
            rec["text"] = text
        (OUT / "flame" / f"{cid}.json").write_text(json.dumps(rec, separators=(",", ":")))
        index.append({k: rec[k] for k in ("id", "title", "creators", "year", "kind", "public_domain")}
                     | {"n_words": len(spans), "models": summary})
    (OUT / "flame_index.json").write_text(json.dumps(
        {"models": [{"key": m, "name": DISPLAY[m]} for m in MODELS], "threshold_log10": THRESH,
         "works": index}, separators=(",", ":")))
    print(f"exported {len(index)} works "
          f"({sum(i['kind'] == 'book' for i in index)} books)")


if __name__ == "__main__":
    main()
