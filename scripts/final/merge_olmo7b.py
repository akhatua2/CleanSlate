"""Merge olmo-7b gentle and aggressive checkpoints by weight interpolation."""

import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def merge(gentle_dir: Path, aggressive_dir: Path, out_dir: Path, gentle_weight: float) -> None:
    agg_weight = 1.0 - gentle_weight
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find safetensors shards
    gentle_shards = sorted(gentle_dir.glob("model*.safetensors"))
    aggressive_shards = sorted(aggressive_dir.glob("model*.safetensors"))
    assert len(gentle_shards) == len(aggressive_shards), (
        f"Shard count mismatch: {len(gentle_shards)} vs {len(aggressive_shards)}"
    )

    print(f"Merging {len(gentle_shards)} shards: {gentle_weight:.2f} gentle + {agg_weight:.2f} aggressive")

    for g_shard, a_shard in zip(gentle_shards, aggressive_shards):
        assert g_shard.name == a_shard.name, f"Shard name mismatch: {g_shard.name} vs {a_shard.name}"
        print(f"  {g_shard.name} ...", end=" ", flush=True)

        g_state = load_file(str(g_shard))
        a_state = load_file(str(a_shard))

        merged = {}
        for key in g_state:
            merged[key] = g_state[key] * gentle_weight + a_state[key] * agg_weight

        save_file(merged, str(out_dir / g_shard.name))
        print("done")

    # Copy non-weight files (config, tokenizer, etc.)
    for f in gentle_dir.iterdir():
        if f.suffix == ".safetensors" or f.name.startswith("."):
            continue
        dst = out_dir / f.name
        if not dst.exists():
            if f.is_dir():
                shutil.copytree(f, dst)
            else:
                shutil.copy2(f, dst)
            print(f"  Copied {f.name}")

    print(f"Saved merged model to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gentle-weight", type=float, required=True)
    args = parser.parse_args()

    gentle_dir = Path("results/final/unlearnt_models/olmo-7b/simnpo/simnpo_Olmo-3-1025-7B_20260310_154332_old_v1")
    aggressive_dir = Path("results/final/unlearnt_models/olmo-7b/simnpo/simnpo_Olmo-3-1025-7B_20260311_092326")

    tag = f"merged_g{int(args.gentle_weight * 100)}_a{int((1 - args.gentle_weight) * 100)}"
    out_dir = Path(f"results/final/unlearnt_models/olmo-7b/simnpo/{tag}")

    merge(gentle_dir, aggressive_dir, out_dir, args.gentle_weight)
