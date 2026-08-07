"""
torchrun launcher for the CPT calibration phase.

Purpose: bypass the modal/single-process default of `cleanslate fsid train` so we can
run multi-GPU FSDP locally. Reads the same yaml + CLI overrides the CLI does, then
calls `train_unlearning(config)` once per rank under torchrun.

Usage:
  torchrun --nproc_per_node=8 scripts/cpt_calibration_launch.py \
      --config cleanslate/fsid/configs/cpt.yaml \
      --model meta-llama/Llama-3.1-8B \
      --forget-path data/cpt_calibration/forget.parquet \
      --retain-path data/cpt_calibration/retain.parquet \
      --output-dir runs/cpt_calibration \
      --batch-size 4 \
      --grad-accum 4 \
      --lr 5e-6 \
      --epochs 1 \
      --save-every 100
"""
from __future__ import annotations

import argparse
import sys

from cleanslate.fsid.config import load_config
from cleanslate.fsid.training.trainer import train_unlearning


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--forget-path", required=True)
    ap.add_argument("--retain-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--objective", default=None)
    args = ap.parse_args()

    cli_overrides: dict = {
        "model_name": args.model,
        "forget_path": args.forget_path,
        "retain_path": args.retain_path,
        "output_dir": args.output_dir,
        "objective": args.objective,
        "batch_size": args.batch_size,
        "gradient_accum_steps": args.grad_accum,
        "lr": args.lr,
        "num_epochs": args.epochs,
        "save_every": args.save_every,
    }
    config = load_config(args.config, cli_overrides=cli_overrides)
    summary = train_unlearning(config)
    if summary.get("is_main_process", True):
        ckpt_dir = summary["checkpoint_dir"]
        print(f"[cpt-launch] checkpoint: {ckpt_dir}")
        print(f"[cpt-launch] runtime:    {summary['run_seconds']:.1f}s")
        # Ensure the checkpoint directory is directly loadable by vLLM /
        # from_pretrained without manual patching after every run.
        import shutil
        from pathlib import Path
        from transformers import AutoConfig, AutoTokenizer
        out = Path(ckpt_dir)
        base = args.model
        if not (out / "config.json").exists():
            AutoConfig.from_pretrained(base).save_pretrained(out)
        if not (out / "tokenizer.json").exists():
            AutoTokenizer.from_pretrained(base).save_pretrained(out)
        # Patch any intermediate checkpoints saved during training.
        for ckpt in sorted(out.glob("checkpoint-*")):
            if not (ckpt / "config.json").exists():
                shutil.copy(out / "config.json", ckpt / "config.json")
            if not (ckpt / "tokenizer.json").exists():
                shutil.copy(out / "tokenizer.json", ckpt / "tokenizer.json")
                shutil.copy(out / "tokenizer_config.json", ckpt / "tokenizer_config.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
