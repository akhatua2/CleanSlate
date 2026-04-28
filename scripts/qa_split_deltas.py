"""Split cleanslate_qa accuracy by forget vs retain content_id, pre vs post.

Reads pushed generation datasets and the curated forget pool, partitions
cleanslate_qa rows by `doc.content_id in forget_ids`, and reports mean of the
already-judged `score` column per partition.
"""
import json
import sys
from datasets import load_dataset

PRE_GEN = "unlearning-cleanslate/generations-llama-3_1-8b-pre_val"
POST_GEN = "unlearning-cleanslate/generations-checkpoint-134-debug-checkpoint-134-llama"
POOL_REPO = "unlearning-cleanslate/fsid-curated-llama-8b"
POOL_CFG = "forget_pool"
POOL_SPLIT = "train"


def split_acc(repo: str, forget_ids: set[str]) -> dict[str, float | int]:
    ds = load_dataset(repo, "cleanslate_qa", split="train")
    f, r = [], []
    for row in ds:
        cid = str((row.get("doc") or {}).get("content_id") or "")
        s = float(row.get("score") or 0.0)
        (f if cid in forget_ids else r).append(s)
    return {
        "forget_acc": sum(f) / len(f) if f else 0.0,
        "retain_acc": sum(r) / len(r) if r else 0.0,
        "n_forget": len(f),
        "n_retain": len(r),
    }


def main() -> None:
    pool = load_dataset(POOL_REPO, POOL_CFG, split=POOL_SPLIT)
    forget_ids = {str(r["content_id"]) for r in pool if r.get("content_id")}
    print(f"forget pool size: {len(forget_ids)}", file=sys.stderr)

    pre = split_acc(PRE_GEN, forget_ids)
    post = split_acc(POST_GEN, forget_ids)

    report = {
        "n_forget_questions": pre["n_forget"],
        "n_retain_questions": pre["n_retain"],
        "qa_forget": {
            "pre": pre["forget_acc"],
            "post": post["forget_acc"],
            "delta": post["forget_acc"] - pre["forget_acc"],
        },
        "qa_retain": {
            "pre": pre["retain_acc"],
            "post": post["retain_acc"],
            "delta": post["retain_acc"] - pre["retain_acc"],
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
