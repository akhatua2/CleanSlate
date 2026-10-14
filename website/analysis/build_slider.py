"""Unlearning-slider data: per memorized song, extractable windows at each checkpoint of
evaluation-aware SimNPO unlearning (base, 25%, 50%, 75%, 100% of training).

Writes public/data/slider.json:
  {stages: [...], models: [{key, name, forget: [...], retain: [...], songs: [{id, t, a, pool, n: [k0..k4]}]}]}
where forget/retain are total extractable windows per stage over each pool.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).parent
CACHE = HERE / "cache"
DATA = HERE.parent / "public/data"
THRESH = -3.0
MODELS = {"llama-8b": "Llama-3.1-8B", "olmo-7b": "Olmo-3-7B", "qwen3-8b": "Qwen3-8B"}
STAGES = ["pre", "simnpo__ea-ckpt1", "simnpo__ea-ckpt2", "simnpo__ea-ckpt3", "simnpo__ea"]
LABELS = ["Base model", "25%", "50%", "75%", "Done"]


def counts(key: str) -> pd.Series:
    t = pq.read_table(CACHE / f"{key}.parquet", columns=["content_id", "log10_pz"]).to_pandas()
    return (t.log10_pz > THRESH).groupby(t.content_id).sum()


def main() -> None:
    pools = pq.read_table(CACHE / "pools.parquet").to_pandas()
    works = pq.read_table(CACHE / "works.parquet", columns=["content_id", "title", "creators", "kind"]).to_pandas().set_index("content_id")
    out = []
    for m, name in MODELS.items():
        cs = pd.concat([counts(f"{m}__{s}") for s in STAGES], axis=1).fillna(0).astype(int)
        forget = set(pools[(pools.model == m) & (pools.pool == "forget")].content_id)
        is_f = cs.index.isin(forget)
        tf, tr = cs[is_f].sum().tolist(), cs[~is_f].sum().tolist()
        print(f"{name:13} forget erased " + " ".join(f"{100 * (1 - x / tf[0]):5.1f}" for x in tf)
              + " | retain erased " + " ".join(f"{100 * (1 - x / tr[0]):5.1f}" for x in tr))
        mem = cs[cs.iloc[:, 0] > 0]
        songs = [{
            "id": c, "t": works.loc[c, "title"], "a": works.loc[c, "creators"], "k": works.loc[c, "kind"],
            "pool": "forget" if c in forget else "retain", "n": row.tolist(),
        } for c, row in mem.iterrows()]
        out.append({"key": m, "name": name, "forget": tf, "retain": tr, "songs": songs})
        print(f"  {sum(s['pool'] == 'forget' for s in songs)} requested + {sum(s['pool'] == 'retain' for s in songs)} other memorized works")
    (DATA / "slider.json").write_text(json.dumps({"stages": LABELS, "models": out}, separators=(",", ":")))


if __name__ == "__main__":
    main()
