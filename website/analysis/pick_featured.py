"""Shortlist candidate works for the flame figure from flame_index.json.

Categories (all counts are extractable windows, p_z >= 1e-3):
  shared     memorized by the most models before unlearning
  erased     in a model's forget request and fully erased by EA unlearning
  collateral NOT requested by any model, strongly memorized, erased anyway
  survivor   NOT requested, strongly memorized, survives unlearning
"""
import json
from pathlib import Path

ix = json.loads((Path(__file__).parents[1] / "public/data/flame_index.json").read_text())
songs = [w for w in ix["works"] if w["kind"] == "song"]


def per(w, st):
    return {m: s.get(st) for m, s in w["models"].items() if s.get(st) is not None}


def show(title, rows, fmt):
    print(f"\n== {title}")
    for w in rows[:8]:
        print(f"  {w['title'][:38]:38} | {(w['creators'] or '')[:22]:22} | {w['year']} | {fmt(w)}")


shared = sorted(songs, key=lambda w: (-sum(v > 0 for v in per(w, "pre").values()), -sum(per(w, "pre").values())))
show("shared", shared, lambda w: " ".join(f"{m}:{v}" for m, v in per(w, "pre").items()))

erased = []
for w in songs:
    for m, s in w["models"].items():
        if s.get("pool") == "forget" and s.get("pre", 0) >= 20 and s.get("ea") == 0:
            erased.append((w, m, s["pre"]))
erased.sort(key=lambda t: -t[2])
print("\n== erased (requested, pre>=20, ea==0)")
for w, m, p in erased[:8]:
    print(f"  {w['title'][:38]:38} | {m:11} | pre {p}")

unreq = [w for w in songs if all(s.get("pool") != "forget" for s in w["models"].values())]
coll = []
for w in unreq:
    for m, s in w["models"].items():
        if s.get("pre", 0) >= 20 and s.get("ea") is not None:
            coll.append((w, m, s["pre"], s["ea"]))
coll.sort(key=lambda t: (t[3] / t[2], -t[2]))
print("\n== collateral (never requested, erased most)")
for w, m, p, e in coll[:8]:
    print(f"  {w['title'][:38]:38} | {m:11} | {p} -> {e}")
print("\n== survivors (never requested, kept most)")
for w, m, p, e in sorted(coll, key=lambda t: (-t[3] / t[2], -t[2]))[:8]:
    print(f"  {w['title'][:38]:38} | {m:11} | {p} -> {e}")
