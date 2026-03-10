"""FSID unlearning trainer using HF Trainer with open-unlearning patterns."""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from accelerate.logging import get_logger
from transformers import Trainer
import yaml

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
    compute_kl_retain_loss,
    compute_npo_loss,
    compute_simnpo_loss,
)
from cleanslate.fsid.training.modeling import load_model, load_tokenizer

logger = get_logger(__name__, log_level="INFO")

ALLOWED_OBJECTIVES = ["ga", "grad_diff", "npo", "simnpo"]


def _save_resolved_config(config: TrainerConfig, run_name: str, run_dir: Path) -> None:
    config_snapshot = asdict(config)
    config_snapshot["_run_name"] = run_name
    config_snapshot["_run_dir"] = str(run_dir)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config_snapshot, f, default_flow_style=False, sort_keys=False)


def _setup_file_logging(log_path: Path) -> logging.FileHandler:
    """Add a FileHandler to the module logger. Returns the handler for cleanup."""
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger(__name__).addHandler(fh)
    return fh


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
        forget_weight: float = 1.0,
        retain_weight: float = 1.0,
        retain_loss_type: str = "nll",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.objective = objective
        self.ref_model = ref_model
        self.npo_beta = npo_beta
        self.simnpo_gamma = simnpo_gamma
        self.forget_weight = forget_weight
        self.retain_weight = retain_weight
        self.retain_loss_type = retain_loss_type

    def floating_point_ops(self, inputs: dict[str, Any]) -> int:
        """Estimate FLOPs from nested forget/retain inputs."""
        total_tokens = 0
        for key in ("forget", "retain"):
            sub = inputs.get(key)
            if sub is not None and "input_ids" in sub:
                total_tokens += sub["input_ids"].numel()
        # ~6 FLOPs per token per parameter (forward + backward)
        num_params = sum(p.numel() for p in self.model.parameters())
        return 6 * total_tokens * num_params

    @staticmethod
    def prepare_ref_model(config: TrainerConfig, accelerator: Any) -> Any:
        """Load a separate frozen reference model from disk.

        Avoids deepcopy which materializes all FSDP shards on every rank,
        doubling peak memory.  Loading from disk lets each rank stay lean
        until FSDP re-shards the new model.
        """
        ref_model = load_model(config)
        for param in ref_model.parameters():
            param.requires_grad = False
        # Let accelerator.prepare_model handle device placement via FSDP
        # sharding.  Calling .to(device) before prepare would materialise
        # the full model on every GPU and OOM for large models (e.g. 32B).
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

        # Retain loss (NLL or KL divergence)
        retain_loss = torch.tensor(0.0, device=f_ids.device)
        if retain_inputs is not None and self.retain_weight > 0:
            r_ids = retain_inputs["input_ids"]
            r_mask = retain_inputs["attention_mask"]
            r_labels = retain_inputs["labels"]
            if self.retain_loss_type == "kl" and self.ref_model is not None:
                retain_loss = compute_kl_retain_loss(
                    model, self.ref_model, r_ids, r_mask, r_labels
                )
            else:
                r_out = model(input_ids=r_ids, attention_mask=r_mask, labels=r_labels)
                retain_loss = r_out.loss

        total_loss = self.forget_weight * forget_loss + self.retain_weight * retain_loss

        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self._log_losses(forget_loss, retain_loss)

        if return_outputs:
            return total_loss, {"forget_loss": forget_loss, "retain_loss": retain_loss}
        return total_loss

    def _log_losses(self, forget_loss: torch.Tensor, retain_loss: torch.Tensor) -> None:
        self.log(
            {
                "forget_loss": forget_loss.detach().item(),
                "retain_loss": retain_loss.detach().item(),
            }
        )

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

    run_name = config.make_run_name()
    training_args = config.to_training_arguments(run_name=run_name)
    run_dir = Path(training_args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

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
    if (
        config.max_forget_pairs is not None
        and len(forget_pairs) > config.max_forget_pairs
    ):
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
    # Use device_map="auto" for non-FSDP multi-GPU (e.g. LoRA with model parallelism)
    use_device_map = (
        "auto"
        if config.distributed_strategy != "fsdp" and config.device == "auto"
        else None
    )
    model = load_model(config, device_map=use_device_map)

    # LoRA
    if config.train_mode == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as e:
            raise ImportError("LoRA requires peft: pip install peft") from e

        target_modules = [
            m.strip() for m in config.lora_target_modules.split(",") if m.strip()
        ]
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)

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
        forget_weight=config.forget_weight,
        retain_weight=config.retain_weight,
        retain_loss_type=config.retain_loss_type,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_collator,
        processing_class=tokenizer,
    )

    if config.objective in ("npo", "simnpo") and config.retain_weight == 0:
        logger.warning(
            "%s with retain_weight=0: no constraint on non-forget data. "
            "Model may degrade globally. Set retain_weight > 0 with a retain_path.",
            config.objective.upper(),
        )

    # File logging + config snapshot (main process only)
    file_handler: logging.FileHandler | None = None
    if trainer.is_world_process_zero():
        file_handler = _setup_file_logging(run_dir / "train.log")
        _save_resolved_config(config, run_name, run_dir)
        logger.info(
            "Starting FSID train: model=%s objective=%s train_mode=%s "
            "strategy=%s run_dir=%s",
            config.model_name,
            config.objective,
            config.train_mode,
            config.distributed_strategy,
            run_dir,
        )

    # Override eval collator (Trainer uses data_collator for eval too by default)
    if eval_collator is not None:
        trainer._eval_collator = eval_collator

    # Create frozen ref model for NPO or KL retain loss
    needs_ref = config.objective == "npo" or config.retain_loss_type == "kl"
    if needs_ref:
        logger.info("Creating frozen reference model for %s", config.objective)
        trainer.ref_model = CleanSlateUnlearnTrainer.prepare_ref_model(
            config, trainer.accelerator
        )

    # --- Train ---
    run_start = time.time()
    logger.info(
        "Starting training: objective=%s train_mode=%s epochs=%d",
        config.objective,
        config.train_mode,
        config.num_epochs,
    )
    trainer.train()
    run_seconds = time.time() - run_start

    # --- Save ---
    trainer.accelerator.wait_for_everyone()
    trainer.save_model()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info("Saved checkpoint to %s", training_args.output_dir)

    # --- Summary ---
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())

    summary: dict[str, Any] = {
        "run_name": run_name,
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

    # Clean up file handler
    if file_handler is not None:
        logging.getLogger(__name__).removeHandler(file_handler)
        file_handler.close()

    return summary
