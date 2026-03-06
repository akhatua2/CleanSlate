"""FSID unlearning trainer using HF Trainer with open-unlearning patterns."""

import copy
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer

from cleanslate.fsid.config import TrainerConfig
from cleanslate.fsid.datasets.io import load_local_records
from cleanslate.fsid.datasets.unlearning import (
    ForgetPairDataset,
    TextSequenceDataset,
    build_forget_pairs,
    tokenize_text_rows,
)
from cleanslate.fsid.training.data import ForgetRetainDataset, UnlearnDataCollator
from cleanslate.fsid.training.losses import (
    compute_npo_loss,
    compute_simnpo_loss,
)
from cleanslate.fsid.training.modeling import load_model, load_tokenizer

logger = logging.getLogger(__name__)

ALLOWED_OBJECTIVES = ["ga", "grad_diff", "npo", "simnpo"]


def list_objectives() -> list[str]:
    return sorted(ALLOWED_OBJECTIVES)


class CleanSlateUnlearnTrainer(Trainer):
    """HF Trainer subclass for unlearning, following open-unlearning patterns.

    Dispatches to GA, NPO, or SimNPO loss in compute_loss(). For NPO, keeps a
    frozen reference model computed on-the-fly (no precompute step).
    """

    def __init__(
        self,
        objective: str,
        ref_model: Any | None = None,
        npo_beta: float = 0.1,
        simnpo_gamma: float = 0.1,
        retain_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.objective = objective
        self.ref_model = ref_model
        self.npo_beta = npo_beta
        self.simnpo_gamma = simnpo_gamma
        self.retain_weight = retain_weight

    @staticmethod
    def prepare_ref_model(model: Any, accelerator: Any) -> Any:
        """Create a frozen reference model following open-unlearning's pattern."""
        ref_model = copy.deepcopy(model)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model = accelerator.prepare_model(ref_model, evaluation_mode=True)
        ref_model.eval()
        return ref_model

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        forget_inputs = inputs["forget"]
        retain_inputs = inputs.get("retain")

        f_ids = forget_inputs["input_ids"]
        f_mask = forget_inputs["attention_mask"]
        f_labels = forget_inputs["labels"]

        # Forget loss
        if self.objective in ("ga", "grad_diff"):
            # GA: gradient ascent = negative CE = maximize log-prob on forget
            outputs = model(input_ids=f_ids, attention_mask=f_mask, labels=f_labels)
            forget_loss = -outputs.loss
        elif self.objective == "npo":
            forget_loss = compute_npo_loss(
                model, self.ref_model, f_ids, f_mask, f_labels, self.npo_beta
            )
        elif self.objective == "simnpo":
            forget_loss = compute_simnpo_loss(
                model, f_ids, f_mask, f_labels, self.npo_beta, self.simnpo_gamma
            )
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # Retain loss (standard CE)
        retain_loss = torch.tensor(0.0, device=f_ids.device)
        if retain_inputs is not None and self.retain_weight > 0:
            r_out = model(
                input_ids=retain_inputs["input_ids"],
                attention_mask=retain_inputs["attention_mask"],
                labels=retain_inputs["labels"],
            )
            retain_loss = r_out.loss

        total_loss = forget_loss + self.retain_weight * retain_loss

        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self._log_losses(forget_loss, retain_loss)

        if return_outputs:
            return total_loss, {"forget_loss": forget_loss, "retain_loss": retain_loss}
        return total_loss

    def _log_losses(self, forget_loss: torch.Tensor, retain_loss: torch.Tensor) -> None:
        self.log({
            "forget_loss": forget_loss.detach().item(),
            "retain_loss": retain_loss.detach().item(),
        })

    def prediction_step(
        self,
        model: Any,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, Any, Any]:
        """Eval uses standard CE loss (not the unlearning objective)."""
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
        return (outputs.loss, None, None)


def train_unlearning(config: TrainerConfig) -> dict[str, Any]:
    """Entry point for unlearning training. Replaces the manual training loop."""
    config.validate()

    # --- Tokenizer ---
    tokenizer = load_tokenizer(config.model_name)
    pad_id = int(tokenizer.pad_token_id)

    # --- Datasets ---
    logger.info("Loading forget set from %s", config.forget_path)
    forget_rows = load_local_records(config.forget_path)
    forget_pairs = build_forget_pairs(
        forget_rows,
        tokenizer,
        text_col=config.forget_text_col,
        window_size=config.window_size,
        stride=config.stride,
        max_windows_per_song=config.max_windows_per_song,
        max_seq_len=config.max_seq_len,
    )
    if config.max_forget_pairs is not None and len(forget_pairs) > config.max_forget_pairs:
        forget_pairs = forget_pairs[: config.max_forget_pairs]
    if not forget_pairs:
        raise ValueError(
            f"No forget training pairs could be built. "
            f"Check forget_path={config.forget_path!r} and "
            f"forget_text_col={config.forget_text_col!r}."
        )
    logger.info("Built %d forget pairs", len(forget_pairs))

    retain_rows: list[dict[str, Any]] = []
    retain_examples: list[dict[str, Any]] = []
    if config.retain_path:
        logger.info("Loading retain set from %s", config.retain_path)
        retain_rows = load_local_records(config.retain_path)
        if config.max_retain_records is not None:
            retain_rows = retain_rows[: config.max_retain_records]
        retain_examples = tokenize_text_rows(
            retain_rows,
            tokenizer,
            text_col=config.retain_text_col,
            max_seq_len=config.max_seq_len,
        )
        logger.info("Prepared %d retain sequences", len(retain_examples))

    val_examples: list[dict[str, Any]] = []
    if config.validation_path:
        logger.info("Loading validation set from %s", config.validation_path)
        val_rows = load_local_records(config.validation_path)
        val_examples = tokenize_text_rows(
            val_rows,
            tokenizer,
            text_col=config.validation_text_col,
            max_seq_len=config.max_seq_len,
        )
        if not val_examples:
            raise ValueError(
                f"No validation sequences from {config.validation_path!r}. "
                f"Check validation_text_col={config.validation_text_col!r}."
            )
        if len(val_examples) > config.eval_max_samples:
            val_examples = val_examples[: config.eval_max_samples]
        logger.info("Prepared %d validation sequences", len(val_examples))

    forget_dataset = ForgetPairDataset(forget_pairs)
    retain_dataset = TextSequenceDataset(retain_examples) if retain_examples else None
    train_dataset = ForgetRetainDataset(forget_dataset, retain_dataset)
    eval_dataset = TextSequenceDataset(val_examples) if val_examples else None

    # --- Model ---
    logger.info("Loading model: %s", config.model_name)
    model = load_model(config)

    # LoRA
    if config.train_mode == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as e:
            raise ImportError("LoRA requires peft: pip install peft") from e

        target_modules = [m.strip() for m in config.lora_target_modules.split(",") if m.strip()]
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)

    # --- Training Arguments ---
    training_args = config.to_training_arguments()

    # --- Collators ---
    train_collator = UnlearnDataCollator(pad_token_id=pad_id)

    # For eval, we use a simple collator (TextSequenceDataset returns flat dicts)
    eval_collator: Any = None
    if eval_dataset is not None:
        from cleanslate.fsid.datasets.unlearning import collate_text_sequences

        eval_collator = lambda batch: collate_text_sequences(batch, pad_id)  # noqa: E731

    # --- Trainer ---
    # ref_model is created after Trainer init for NPO (needs accelerator)
    trainer = CleanSlateUnlearnTrainer(
        objective=config.objective,
        npo_beta=config.npo_beta,
        simnpo_gamma=config.simnpo_gamma,
        retain_weight=config.retain_weight,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_collator,
        processing_class=tokenizer,
    )

    # Override eval collator (Trainer uses data_collator for eval too by default)
    if eval_collator is not None:
        trainer._eval_collator = eval_collator

    # Create frozen ref model for NPO (after trainer.model is prepared by accelerator)
    if config.objective == "npo":
        logger.info("Creating frozen reference model for NPO")
        trainer.ref_model = CleanSlateUnlearnTrainer.prepare_ref_model(
            model, trainer.accelerator
        )

    # --- Train ---
    run_start = time.time()
    logger.info(
        "Starting training: objective=%s train_mode=%s max_steps=%d",
        config.objective,
        config.train_mode,
        config.max_steps,
    )
    trainer.train()
    run_seconds = time.time() - run_start

    # --- Save ---
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Saved checkpoint to %s", training_args.output_dir)

    # --- Summary ---
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())

    summary: dict[str, Any] = {
        "run_dir": training_args.output_dir,
        "config": asdict(config),
        "num_forget_rows": len(forget_rows),
        "num_forget_pairs": len(forget_pairs),
        "num_retain_rows": len(retain_rows),
        "num_retain_sequences": len(retain_examples),
        "num_val_sequences": len(val_examples),
        "num_trainable_params": num_trainable,
        "num_total_params": num_total,
        "run_seconds": run_seconds,
        "checkpoint_dir": training_args.output_dir,
        "is_main_process": trainer.is_world_process_zero(),
    }
    if trainer.is_world_process_zero():
        with open(Path(training_args.output_dir) / "train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Training complete (%.1fs)", run_seconds)

    return summary
