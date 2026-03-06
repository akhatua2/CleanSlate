"""Local PyTorch unlearning training orchestrator."""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from cleanslate.fsid.config import TrainerConfig
from cleanslate.fsid.datasets.io import load_local_records
from cleanslate.fsid.datasets.unlearning import (
    ForgetPairDataset,
    TextSequenceDataset,
    build_forget_pairs,
    collate_forget,
    collate_text_sequences,
    tokenize_text_rows,
)
from cleanslate.fsid.training.modeling import (
    get_input_device,
    load_tokenizer,
    load_train_model,
    maybe_merge_lora_for_export,
    precompute_reference_logps,
    resolve_device,
    resolve_dtype,
    set_seed,
)
from cleanslate.fsid.training.distributed import (
    DistributedContext,
    build_training_sampler,
    destroy_distributed,
    init_distributed,
    partition_dataset,
    reduce_scalar_mean,
    reduce_scalar_sum,
    save_model_checkpoint,
    wrap_model_for_fsdp,
)
from cleanslate.fsid.training.objectives import build_objective

logger = logging.getLogger(__name__)


@dataclass
class _LoaderState:
    loader: DataLoader
    iterator: Any
    sampler: Any = None
    epoch: int = 0


def _next_batch(state: _LoaderState) -> dict[str, torch.Tensor]:
    try:
        batch = next(state.iterator)
    except StopIteration:
        state.epoch += 1
        if state.sampler is not None:
            state.sampler.set_epoch(state.epoch)
        state.iterator = iter(state.loader)
        batch = next(state.iterator)
    return batch


def _to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _evaluate_validation(
    model: Any,
    val_loader: DataLoader,
    input_device: str,
    distributed_context: DistributedContext,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = _to_device(batch, input_device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            num_tokens = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
    model.train()
    if total_tokens == 0:
        raise ValueError("Validation loader yielded zero tokens.")
    global_total_loss = reduce_scalar_sum(
        total_loss,
        device=input_device,
        context=distributed_context,
    )
    global_total_tokens = reduce_scalar_sum(
        float(total_tokens),
        device=input_device,
        context=distributed_context,
    )
    if global_total_tokens == 0:
        raise ValueError("Validation loader yielded zero tokens globally.")
    return global_total_loss / global_total_tokens


def train_unlearning(config: TrainerConfig) -> dict[str, Any]:
    """Run local unlearning training and return a summary dict."""
    config.validate()
    distributed_context = init_distributed(config)
    try:
        device = (
            distributed_context.device
            if distributed_context.is_distributed
            else resolve_device(config.device)
        )
        set_seed(config.seed + distributed_context.rank)
        torch_dtype = resolve_dtype(device, config.dtype)
        objective = build_objective(config)

        tokenizer = load_tokenizer(config.model_name)
        pad_id = int(tokenizer.pad_token_id)

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
                "No forget training pairs could be built. "
                f"Check forget_path={config.forget_path!r} and forget_text_col={config.forget_text_col!r}."
            )

        retain_rows: list[dict[str, Any]] = []
        if config.retain_path:
            retain_rows = load_local_records(config.retain_path)
            if config.max_retain_records is not None:
                retain_rows = retain_rows[: config.max_retain_records]
        retain_examples = tokenize_text_rows(
            retain_rows,
            tokenizer,
            text_col=config.retain_text_col,
            max_seq_len=config.max_seq_len,
        )

        val_examples: list[dict[str, Any]] = []
        if config.validation_path:
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

        forget_dataset = ForgetPairDataset(forget_pairs)
        retain_dataset: TextSequenceDataset | None = None
        if retain_examples:
            retain_dataset = TextSequenceDataset(retain_examples)

        val_dataset: TextSequenceDataset | None = None
        if val_examples:
            val_dataset = TextSequenceDataset(val_examples)

        model = load_train_model(config, device, torch_dtype)
        if distributed_context.is_distributed:
            model = wrap_model_for_fsdp(
                model,
                config=config,
                torch_dtype=torch_dtype,
            )
        input_device = (
            distributed_context.device
            if distributed_context.is_distributed
            else get_input_device(model)
        )

        if objective.needs_ref_logps:
            precompute_reference_logps(
                model=model,
                dataset=forget_dataset,
                batch_size=config.batch_size,
                pad_id=pad_id,
                input_device=input_device,
                num_workers=config.num_workers,
                use_sums=objective.ref_logps_are_sums,
                distributed_context=distributed_context,
            )

        objective.setup(model, config)

        trainable_params = [param for param in model.parameters() if param.requires_grad]
        if not trainable_params:
            raise ValueError(
                "No trainable parameters found. Check train_mode and model configuration."
            )

        num_trainable = sum(param.numel() for param in trainable_params)
        num_total = sum(param.numel() for param in model.parameters())
        pin_memory = input_device.startswith("cuda")

        forget_sampler = build_training_sampler(
            forget_dataset,
            shuffle=True,
            seed=config.seed,
            context=distributed_context,
        )
        if forget_sampler is not None:
            forget_sampler.set_epoch(0)
        forget_loader = DataLoader(
            forget_dataset,
            batch_size=config.batch_size,
            shuffle=forget_sampler is None,
            sampler=forget_sampler,
            num_workers=config.num_workers,
            collate_fn=lambda batch: collate_forget(batch, pad_id),
            pin_memory=pin_memory,
        )
        forget_state = _LoaderState(
            loader=forget_loader,
            iterator=iter(forget_loader),
            sampler=forget_sampler,
        )

        retain_loader: DataLoader | None = None
        retain_state: _LoaderState | None = None
        needs_retain = retain_dataset is not None and (
            config.retain_weight > 0 or objective.owns_retain_loss
        )
        if needs_retain:
            retain_sampler = build_training_sampler(
                retain_dataset,
                shuffle=True,
                seed=config.seed + 17,
                context=distributed_context,
            )
            if retain_sampler is not None:
                retain_sampler.set_epoch(0)
            retain_loader = DataLoader(
                retain_dataset,
                batch_size=config.batch_size,
                shuffle=retain_sampler is None,
                sampler=retain_sampler,
                num_workers=config.num_workers,
                collate_fn=lambda batch: collate_text_sequences(batch, pad_id),
                pin_memory=pin_memory,
            )
            retain_state = _LoaderState(
                loader=retain_loader,
                iterator=iter(retain_loader),
                sampler=retain_sampler,
            )

        val_loader: DataLoader | None = None
        if val_dataset is not None:
            val_loader = DataLoader(
                partition_dataset(val_dataset, distributed_context),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=lambda batch: collate_text_sequences(batch, pad_id),
                pin_memory=pin_memory,
            )

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
        )

        # Create timestamped run directory
        base_dir = Path(config.output_dir)
        model_short = config.model_name.split("/")[-1]
        run_name = f"{config.objective}_{model_short}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = base_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save resolved config snapshot
        if distributed_context.is_main_process:
            config_snapshot = asdict(config)
            config_snapshot["_run_name"] = run_name
            config_snapshot["_run_dir"] = str(output_dir)
            with open(output_dir / "config.yaml", "w") as f:
                import yaml
                yaml.dump(config_snapshot, f, default_flow_style=False, sort_keys=False)

        # Set up file logging
        file_handler: logging.FileHandler | None = None
        if distributed_context.is_main_process:
            file_handler = logging.FileHandler(output_dir / "train.log")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            logging.getLogger("cleanslate.fsid").addHandler(file_handler)
            logger.info("Run directory: %s", output_dir)

        run_start = time.time()
        history: list[dict[str, Any]] = []

        try:
            for step in range(1, config.max_steps + 1):
                optimizer.zero_grad(set_to_none=True)

                step_forget_loss = 0.0
                step_retain_loss = 0.0
                step_total_loss = 0.0

                for _ in range(config.gradient_accum_steps):
                    forget_batch = _next_batch(forget_state)
                    fb = _to_device(forget_batch, input_device)

                    rb: dict[str, Any] | None = None
                    if retain_loader is not None and retain_state is not None:
                        retain_batch = _next_batch(retain_state)
                        rb = _to_device(retain_batch, input_device)

                    total_loss, loss_forget, loss_retain = objective.compute_loss(
                        model,
                        fb,
                        rb,
                        config.retain_weight,
                    )

                    step_forget_loss += float(loss_forget.item())
                    step_retain_loss += float(loss_retain.item())
                    step_total_loss += float(total_loss.item())

                    (total_loss / config.gradient_accum_steps).backward()

                if config.max_grad_norm > 0:
                    if distributed_context.is_distributed:
                        model.clip_grad_norm_(config.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                optimizer.step()
                scheduler.step()

                mean_forget = step_forget_loss / config.gradient_accum_steps
                mean_retain = step_retain_loss / config.gradient_accum_steps
                mean_total = step_total_loss / config.gradient_accum_steps
                current_lr = scheduler.get_last_lr()[0]

                mean_forget = reduce_scalar_mean(
                    mean_forget,
                    device=input_device,
                    context=distributed_context,
                )
                mean_retain = reduce_scalar_mean(
                    mean_retain,
                    device=input_device,
                    context=distributed_context,
                )
                mean_total = reduce_scalar_mean(
                    mean_total,
                    device=input_device,
                    context=distributed_context,
                )

                should_log = step == 1 or step % config.log_every == 0 or step == config.max_steps
                should_eval = (
                    val_loader is not None
                    and (step == 1 or step % config.eval_every == 0 or step == config.max_steps)
                )

                if should_log or should_eval:
                    entry: dict[str, Any] = {
                        "step": step,
                        "forget_loss": mean_forget,
                        "retain_loss": mean_retain,
                        "total_loss": mean_total,
                        "lr": current_lr,
                    }
                    if should_eval:
                        assert val_loader is not None
                        entry["validation_loss"] = _evaluate_validation(
                            model,
                            val_loader,
                            input_device,
                            distributed_context,
                        )
                        if distributed_context.is_main_process:
                            logger.info(
                                "step=%d forget=%.4f retain=%.4f val=%.4f lr=%.2e",
                                step,
                                mean_forget,
                                mean_retain,
                                entry["validation_loss"],
                                current_lr,
                            )
                    elif distributed_context.is_main_process:
                        logger.info(
                            "step=%d forget=%.4f retain=%.4f lr=%.2e",
                            step,
                            mean_forget,
                            mean_retain,
                            current_lr,
                        )
                    if distributed_context.is_main_process:
                        history.append(entry)

                if config.save_every > 0 and step % config.save_every == 0 and step < config.max_steps:
                    ckpt_dir = output_dir / f"checkpoint-{step}"
                    save_model_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        path=ckpt_dir,
                        context=distributed_context,
                        train_mode=config.train_mode,
                    )
                    if distributed_context.is_main_process:
                        logger.info("Saved intermediate checkpoint to %s", ckpt_dir)

        finally:
            objective.teardown()

        run_seconds = time.time() - run_start

        final_ckpt = output_dir / "checkpoint-final"
        if distributed_context.is_distributed:
            save_model_checkpoint(
                model=model,
                tokenizer=tokenizer,
                path=final_ckpt,
                context=distributed_context,
                train_mode=config.train_mode,
            )
        else:
            export_model = maybe_merge_lora_for_export(model, config.train_mode)
            save_model_checkpoint(
                model=export_model,
                tokenizer=tokenizer,
                path=final_ckpt,
                context=distributed_context,
                train_mode=config.train_mode,
            )

        summary: dict[str, Any] = {
            "run_name": run_name,
            "run_dir": str(output_dir),
            "config": asdict(config),
            "device": device,
            "input_device": input_device,
            "dtype": str(torch_dtype),
            "distributed_strategy": distributed_context.strategy,
            "world_size": distributed_context.world_size,
            "rank": distributed_context.rank,
            "is_main_process": distributed_context.is_main_process,
            "num_forget_rows": len(forget_rows),
            "num_forget_pairs": len(forget_pairs),
            "num_retain_rows": len(retain_rows),
            "num_retain_sequences": len(retain_examples),
            "num_val_sequences": len(val_examples),
            "num_trainable_params": num_trainable,
            "num_total_params": num_total,
            "run_seconds": run_seconds,
            "history": history if distributed_context.is_main_process else [],
            "checkpoint_dir": str(final_ckpt),
        }
        if distributed_context.is_main_process:
            with open(output_dir / "train_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Run complete: %s (%.1fs)", run_name, run_seconds)

        # Clean up file logging
        if file_handler is not None:
            logging.getLogger("cleanslate.fsid").removeHandler(file_handler)
            file_handler.close()

        return summary
    finally:
        destroy_distributed(distributed_context)
