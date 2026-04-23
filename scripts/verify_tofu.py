#!/usr/bin/env python3
"""Verify CleanSlate NPO/SimNPO/GA against open-unlearning on TOFU.

Loads TOFU data (forget10 + retain90), tokenizes identically to
open-unlearning's QADataset + preprocess_chat_instance, then trains
using CleanSlate's CleanSlateUnlearnTrainer.

Note: CleanSlate's UNDIAL and RMU implementations follow the source papers
(Dong et al.; Li et al., WMDP). Open-unlearning's reference implementations
contain a label-wrap bug (UNDIAL) and a broadcast bug (RMU) that CleanSlate
does not reproduce, so this script stays byte-identical to open-unlearning
on NPO/SimNPO/GA only — UNDIAL/RMU runs are a smoke test, not a parity check.

Usage:

    # NPO
    python scripts/verify_tofu.py --objective npo --output-dir runs/verify_tofu_npo

    # SimNPO
    python scripts/verify_tofu.py --objective simnpo --output-dir runs/verify_tofu_simnpo

Then evaluate each checkpoint with open-unlearning's eval:

    cd /lfs/skampere3/0/animjha/open-unlearning

    # NPO
    python src/eval.py experiment=eval/tofu/default model=Llama-3.2-1B-Instruct \
        model.model_args.pretrained_model_name_or_path=/scr/animjha/CleanSlate/runs/verify_tofu_npo \
        retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
        task_name=cleanslate_npo_verify

    # SimNPO
    python src/eval.py experiment=eval/tofu/default model=Llama-3.2-1B-Instruct \
        model.model_args.pretrained_model_name_or_path=/scr/animjha/CleanSlate/runs/verify_tofu_simnpo \
        retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
        task_name=cleanslate_simnpo_verify
"""

import argparse
import logging
import time
from typing import Any

import datasets
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from cleanslate.fsid.training.trainer import CleanSlateUnlearnTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Collator & dataset wrappers matching open-unlearning exactly
# ---------------------------------------------------------------------------


class OpenUnlearningCollator:
    """Collator matching open-unlearning's DataCollatorForSupervisedDataset.

    Key difference from CleanSlate's UnlearnDataCollator:
    attention_mask = input_ids.ne(pad_token_id) instead of 1 for all seq positions.
    This masks out in-sequence pad_token_id (= eos_token_id) positions.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def _collate_flat(self, instances: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [
            torch.tensor(inst["input_ids"]) if not isinstance(inst["input_ids"], torch.Tensor) else inst["input_ids"]
            for inst in instances
        ]
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id,
        )
        attention_mask = input_ids_padded.ne(self.pad_token_id)  # KEY: masks in-seq eos too
        labels_list = [
            torch.tensor(inst["labels"]) if not isinstance(inst["labels"], torch.Tensor) else inst["labels"]
            for inst in instances
        ]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100,
        )
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        result["forget"] = self._collate_flat([item["forget"] for item in batch])
        if "retain" in batch[0]:
            result["retain"] = self._collate_flat([item["retain"] for item in batch])
        return result


class RandomRetainDataset(TorchDataset):
    """Forget+retain dataset with random retain sampling.

    open-unlearning's ForgetRetainDataset uses `torch.randint(...)` on the global
    generator, so retain pairings are unreproducible across runs / workers / ranks.
    We expose an explicit `retain_seed` that makes `(forget_idx → retain_idx)`
    fully deterministic: per-index RNG seeded from `retain_seed * LARGE + idx`.
    Set `retain_seed=None` to fall back to the global generator (OU-compatible).
    """

    _SEED_OFFSET: int = 1_000_003  # a prime to decouple adjacent index streams

    def __init__(
        self,
        forget_dataset: TorchDataset,
        retain_dataset: TorchDataset | None = None,
        retain_seed: int | None = None,
    ) -> None:
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset
        self.retain_seed = retain_seed

    def __len__(self) -> int:
        return len(self.forget_dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> dict[str, Any]:
        result: dict[str, Any] = {"forget": self.forget_dataset[index]}
        if self.retain_dataset is not None and len(self.retain_dataset) > 0:  # type: ignore[arg-type]
            n_retain = len(self.retain_dataset)  # type: ignore[arg-type]
            if self.retain_seed is None:
                retain_index = torch.randint(0, n_retain, (1,)).item()
            else:
                g = torch.Generator()
                g.manual_seed(int(self.retain_seed) * self._SEED_OFFSET + int(index))
                retain_index = torch.randint(0, n_retain, (1,), generator=g).item()
            result["retain"] = self.retain_dataset[retain_index]  # type: ignore[index]
        return result


# ---------------------------------------------------------------------------
# Tokenization: replicate open-unlearning's preprocess_chat_instance exactly
# ---------------------------------------------------------------------------


def preprocess_chat_instance(
    tokenizer: Any,
    question: str,
    answer: str,
    system_prompt: str,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize a QA pair using chat template, masking prompt tokens.

    Replicates open-unlearning/src/data/utils.py::preprocess_chat_instance
    with apply_chat_template=True mode.
    """
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    # Full conversation token IDs
    chat_ids: list[int] = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=False, return_dict=False,
    )

    # Prompt-only IDs (everything up to the assistant turn)
    prompt_chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt_ids: list[int] = tokenizer.apply_chat_template(
        prompt_chat, tokenize=True, add_generation_prompt=True, return_dict=False,
    )

    # Ensure EOS at end
    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids.append(tokenizer.eos_token_id)

    # Truncate to max_length (open-unlearning truncates the full text)
    chat_ids = chat_ids[:max_length]

    # Labels: mask prompt tokens with -100, keep answer tokens
    len_prompt = len(prompt_ids)
    labels = [IGNORE_INDEX] * len_prompt + chat_ids[len_prompt:]

    return {
        "input_ids": chat_ids,
        "labels": labels,
        "attention_mask": [1] * len(chat_ids),
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class TOFUQADataset(TorchDataset):
    """TOFU QA dataset tokenized identically to open-unlearning's QADataset."""

    def __init__(
        self,
        hf_dataset: Any,
        tokenizer: Any,
        system_prompt: str = "You are a helpful assistant.",
        max_length: int = 512,
    ) -> None:
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        row = self.data[idx]
        item = preprocess_chat_instance(
            self.tokenizer,
            question=row["question"],
            answer=row["answer"],
            system_prompt=self.system_prompt,
            max_length=self.max_length,
        )
        return item


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify CleanSlate unlearning on TOFU"
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=["ga", "npo", "simnpo", "rmu", "undial"],
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        help="Model checkpoint (finetuned on full TOFU)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: runs/verify_tofu_<objective>)",
    )
    parser.add_argument(
        "--forget-split",
        type=str,
        default="forget10",
    )
    parser.add_argument(
        "--retain-split",
        type=str,
        default="retain90",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token sequence length",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs (open-unlearning default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device batch size (open-unlearning uses 8 per device)",
    )
    parser.add_argument(
        "--gradient-accum-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch_size * accum)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--retain-seed",
        type=int,
        default=None,
        help="Seed for deterministic retain-pair sampling (reproducible across runs). "
             "Omit to use OU-style global-generator sampling (nondeterministic).",
    )
    parser.add_argument(
        "--rmu-control-vec-path",
        type=str,
        default=None,
        help="Path to a .pt file containing a saved control vector. If set, this tensor "
             "is used verbatim; `--seed` and rank-based offset are ignored. Useful for "
             "reproducing a specific OU run (save OU's self.control_vec to disk first).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = f"runs/verify_tofu_{args.objective}"

    logger.info("Objective: %s", args.objective)
    logger.info("Model: %s", args.model)
    logger.info("Output: %s", args.output_dir)

    # --- Hyperparameters matched to open-unlearning ---
    # RMU-specific fields (unused for ga/npo/simnpo/undial)
    rmu_steering_coeff = 0.0
    rmu_module_regex = ""
    retain_loss_type = "nll"
    undial_beta = 10.0

    if args.objective == "ga":
        # Pure gradient ascent: loss = -outputs.loss on forget only.
        # retain_weight=0 gates off the retain forward pass in CleanSlateUnlearnTrainer.
        npo_beta = 0.0
        simnpo_delta = 0.0
        forget_weight = 1.0
        retain_weight = 0.0
    elif args.objective == "npo":
        npo_beta = 0.1
        simnpo_delta = 0.0
        forget_weight = 1.0  # gamma in open-unlearning
        retain_weight = 1.0  # alpha in open-unlearning
    elif args.objective == "simnpo":
        npo_beta = 4.5
        simnpo_delta = 0.0  # matches OU's delta=0.0 for TOFU
        forget_weight = 0.125  # gamma in open-unlearning
        retain_weight = 1.0  # alpha in open-unlearning
    elif args.objective == "rmu":
        npo_beta = 0.0
        simnpo_delta = 0.0
        forget_weight = 1.0  # OU gamma
        retain_weight = 1.0  # OU alpha
        rmu_steering_coeff = 2.0
        rmu_module_regex = r"model\.layers\.7"
        retain_loss_type = "embed_diff"
    elif args.objective == "undial":
        # UNDIAL paper / OU yaml defaults for TOFU: lr=1e-4, beta=10, gamma=1, alpha=0.
        # alpha=0 means no retain loss (paper does not use retain set); we still
        # pass retain batches through but they're gated off by retain_weight=0.
        npo_beta = 0.0
        simnpo_delta = 0.0
        forget_weight = 1.0       # OU gamma
        retain_weight = 0.0       # OU alpha
        undial_beta = 10.0
    else:
        raise ValueError(f"Unknown objective: {args.objective}")

    logger.info(
        "Hyperparams: beta=%.2f, gamma=%.2f, forget_weight=%.4f, retain_weight=%.1f",
        npo_beta,
        simnpo_delta,
        forget_weight,
        retain_weight,
    )

    # --- Tokenizer ---
    logger.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load TOFU data ---
    logger.info("Loading TOFU forget=%s, retain=%s", args.forget_split, args.retain_split)
    forget_hf = datasets.load_dataset(
        "locuslab/TOFU", name=args.forget_split, split="train"
    )
    retain_hf = datasets.load_dataset(
        "locuslab/TOFU", name=args.retain_split, split="train"
    )
    logger.info("Forget: %d examples, Retain: %d examples", len(forget_hf), len(retain_hf))

    # --- Create datasets ---
    forget_dataset = TOFUQADataset(
        forget_hf, tokenizer, max_length=args.max_length
    )
    retain_dataset = TOFUQADataset(
        retain_hf, tokenizer, max_length=args.max_length
    )
    train_dataset = RandomRetainDataset(
        forget_dataset, retain_dataset, retain_seed=args.retain_seed
    )
    if args.retain_seed is not None:
        logger.info("Retain sampling: deterministic (retain_seed=%d)", args.retain_seed)
    else:
        logger.info("Retain sampling: OU-style global-generator (nondeterministic)")

    # --- Load model ---
    logger.info("Loading model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    # --- Training args (match open-unlearning finetune.yaml) ---
    # open-unlearning warmup formula: int(warmup_epochs * dataset_len // effective_batch)
    # warmup_epochs=1.0, effective_batch = batch_size * grad_accum * num_devices
    # On 1 GPU: int(1.0 * 200 // (8 * 8 * 1)) = int(200 // 64) = 3
    warmup_steps = int((1.0 * len(forget_dataset)) // (args.batch_size * args.gradient_accum_steps))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accum_steps,
        learning_rate=args.lr,
        optim="paged_adamw_32bit",
        weight_decay=0.01,  # open-unlearning unlearn/tofu/default.yaml
        warmup_steps=warmup_steps,
        num_train_epochs=args.num_epochs,
        max_grad_norm=1.0,
        logging_steps=1,
        save_strategy="no",
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to="none",
        lr_scheduler_type="linear",
        logging_first_step=True,
        disable_tqdm=False,
        eval_on_start=False,
    )

    # --- Collator (open-unlearning style: masks in-sequence eos/pad tokens) ---
    pad_id = int(tokenizer.pad_token_id)
    collator = OpenUnlearningCollator(pad_token_id=pad_id)

    # --- Trainer ---
    trainer_kwargs: dict[str, Any] = dict(
        objective=args.objective,
        npo_beta=npo_beta,
        simnpo_delta=simnpo_delta,
        forget_weight=forget_weight,
        retain_weight=retain_weight,
        retain_loss_type=retain_loss_type,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    if args.objective == "rmu":
        trainer_kwargs["rmu_steering_coeff"] = rmu_steering_coeff
        trainer_kwargs["rmu_module_regex"] = rmu_module_regex
        trainer_kwargs["rmu_control_vec_seed"] = args.seed
        trainer_kwargs["rmu_control_vec_path"] = args.rmu_control_vec_path
    if args.objective == "undial":
        trainer_kwargs["undial_beta"] = undial_beta
    trainer = CleanSlateUnlearnTrainer(**trainer_kwargs)

    # --- Reference model for NPO / RMU / UNDIAL ---
    if args.objective in ("npo", "rmu", "undial"):
        logger.info("Creating frozen reference model for %s", args.objective.upper())
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        ref_model.config.use_cache = False
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model = trainer.accelerator.prepare_model(ref_model, evaluation_mode=True)
        ref_model.eval()
        trainer.ref_model = ref_model

    # --- Train ---
    logger.info("Starting training: %s, %d epochs", args.objective, args.num_epochs)
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    logger.info("Training complete in %.1fs", elapsed)

    # --- Save ---
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Saved checkpoint to %s", args.output_dir)

    # --- Quick sanity check: print final loss ---
    if trainer.state.log_history:
        last_log = trainer.state.log_history[-1]
        logger.info("Final log entry: %s", last_log)


if __name__ == "__main__":
    main()
