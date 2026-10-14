"""Per-model unlearning summary for the 'we asked a model to forget 50 songs' figure.

For each model (EA curator, SimNPO unlearning):
  forget: extractable windows over the model's 50 requested works, before -> after
  retain: extractable windows over every other work, before -> after
  forget_songs / keep_songs: a few songs to show in each column
Writes public/data/unlearn.json and attaches lyrics to the chosen songs' flame files.
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
MODELS = ["llama-8b", "olmo-7b", "qwen3-8b", "nemotron-9b", "gemma-12b", "olmo-32b"]
N_SHOW = 4


def counts(key: str) -> pd.Series:
    t = pq.read_table(CACHE / f"{key}.parquet").to_pandas()
    return (t.log10_pz > THRESH).groupby(t.content_id).sum()


def main() -> None:
    pools = pq.read_table(CACHE / "pools.parquet").to_pandas()
    works = pq.read_table(CACHE / "works.parquet").to_pandas().set_index("content_id")
    ix = json.loads((DATA / "flame_index.json").read_text())
    names = {m["key"]: m["name"] for m in ix["models"]}
    exported = {w["id"]: w for w in ix["works"]}
    # how many models memorize each song: a rough "everyone knows this song" signal
    n_models = {w["id"]: sum((s.get("pre") or 0) > 0 for s in w["models"].values()) for w in ix["works"]}

    out, show_ids = [], set()
    for m in MODELS:
        pre, post = counts(f"{m}__pre"), counts(f"{m}__simnpo__ea")
        forget_ids = set(pools[(pools.model == m) & (pools.pool == "forget")].content_id)
        is_f = pre.index.isin(forget_ids)
        rec = {
            "key": m, "name": names[m],
            "forget": {"works": int(is_f.sum()), "pre": int(pre[is_f].sum()), "post": int(post.reindex(pre.index)[is_f].sum())},
            "retain": {"works": int((~is_f).sum()), "pre": int(pre[~is_f].sum()), "post": int(post.reindex(pre.index)[~is_f].sum())},
        }
        songs = [c for c in pre.index if c in exported and works.loc[c, "kind"] == "song"]
        f = sorted((c for c in songs if c in forget_ids and pre[c] > 0), key=lambda c: (-n_models[c], -pre[c]))
        k = sorted((c for c in songs if c not in forget_ids and pre[c] >= 8),
                   key=lambda c: (-n_models[c], post[c] / pre[c], -pre[c]))
        rec["forget_songs"] = [{"id": c, "title": works.loc[c, "title"], "pre": int(pre[c]), "post": int(post[c])} for c in f[:N_SHOW]]
        rec["keep_songs"] = [{"id": c, "title": works.loc[c, "title"], "pre": int(pre[c]), "post": int(post[c])} for c in k[:N_SHOW]]
        show_ids |= {s["id"] for s in rec["forget_songs"] + rec["keep_songs"]}
        out.append(rec)
        fr = 100 * (1 - rec["forget"]["post"] / rec["forget"]["pre"])
        rr = 100 * (1 - rec["retain"]["post"] / rec["retain"]["pre"])
        print(f"{m:12} F {fr:5.1f}%  R {rr:5.1f}%  | forget {[s['title'] for s in rec['forget_songs']]} | keep {[s['title'] for s in rec['keep_songs']]}")

    (DATA / "unlearn.json").write_text(json.dumps({"curator": "EA", "unlearner": "SimNPO", "models": out}, separators=(",", ":")))
    for c in show_ids:
        f = DATA / "flame" / f"{c}.json"
        rec = json.loads(f.read_text())
        if "text" not in rec:
            rec["text"] = works.loc[c, "text"]
            f.write_text(json.dumps(rec, separators=(",", ":")))
    print(f"lyrics attached to {len(show_ids)} songs")


if __name__ == "__main__":
    main()
