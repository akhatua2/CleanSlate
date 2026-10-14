"""Mark the hand-picked works shown in the flame figure and attach their text."""
import json
from pathlib import Path

import pyarrow.parquet as pq

FEATURED = [  # (title, why it is here)
    ("C'mon N' Ride It (The Train)", "memorized by all six models; requested by Qwen3-8B and Olmo-3-7B"),
    ("Bohemian Rhapsody", "never requested by any model, erased anyway (Olmo-3-32B)"),
    ("All Star", "never requested by any model, erased anyway (Olmo-3-32B)"),
    ("Rocket Man", "never requested; partly survives unlearning (Gemma-3-12B)"),
    ("Never Gonna Give You Up", "the running example in the paper"),
    ("Back Then", "requested by Llama, Nemotron and Olmo-3-32B; fully erased"),
]
DATA = Path(__file__).parents[1] / "public/data"
works = pq.read_table(Path(__file__).parent / "cache/works.parquet").to_pandas()
ix = json.loads((DATA / "flame_index.json").read_text())
by_title = {}
for w in ix["works"]:
    by_title.setdefault(w["title"].lower(), w)
featured = []
for title, why in FEATURED:
    w = by_title.get(title.lower())
    if not w:
        print("not in exported set:", title)
        continue
    f = DATA / "flame" / f"{w['id']}.json"
    rec = json.loads(f.read_text())
    rec["text"] = works.loc[works.content_id == w["id"], "text"].iloc[0]
    rec["why"] = why
    f.write_text(json.dumps(rec, separators=(",", ":")))
    featured.append(w["id"])
    print("featured:", title, w["creators"])
ix["featured"] = featured
(DATA / "flame_index.json").write_text(json.dumps(ix, separators=(",", ":")))
