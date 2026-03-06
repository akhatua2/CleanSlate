"""Local PyTorch unlearning training orchestrator using HuggingFace Accelerate."""

import json
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
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
    load_tokenizer,
    load_train_model,
    maybe_merge_lora_for_export,
    precompute_reference_logps,
    resolve_device,
    resolve_dtype,
    set_seed,
)
from cleanslate.fsid.training.distributed import (
    _resolve_transformer_layer,
    DistributedContext,
    build_training_sampler,
    partition_dataset,
)
from cleanslate.fsid.training.objectives import build_objective

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _LoaderState:
    loader: DataLoader
    iterator: Any
    sampler: Any = None
    epoch: int = 0


@dataclass
class _LoggingState:
    logger: logging.Logger
    handlers: list[logging.Handler]
    old_level: int
    old_propagate: bool


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


def _resolve_run_name(
    config: TrainerConfig,
    is_main: bool,
    is_distributed: bool,
) -> str:
    model_short = config.model_name.split("/")[-1]
    run_name = ""
    if is_main:
        run_name = (
            f"{config.objective}_{model_short}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    if is_distributed:
        payload = [run_name]
        dist.broadcast_object_list(payload, src=0)
        run_name = str(payload[0])
    return run_name


def _setup_main_process_logging(log_path: Path) -> _LoggingState:
    fsid_logger = logging.getLogger("cleanslate.fsid")
    old_level = fsid_logger.level
    old_propagate = fsid_logger.propagate
    fsid_logger.setLevel(logging.INFO)
    fsid_logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    fsid_logger.addHandler(stream_handler)
    fsid_logger.addHandler(file_handler)

    return _LoggingState(
        logger=fsid_logger,
        handlers=[stream_handler, file_handler],
        old_level=old_level,
        old_propagate=old_propagate,
    )


def _teardown_main_process_logging(state: _LoggingState | None) -> None:
    if state is None:
        return
    for handler in state.handlers:
        state.logger.removeHandler(handler)
        handler.close()
    state.logger.setLevel(state.old_level)
    state.logger.propagate = state.old_propagate


def _build_fsdp_plugin(config: TrainerConfig) -> Any:
    """Build an Accelerate FSDP2 plugin from config."""
    from accelerate import FullyShardedDataParallelPlugin

    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        reshard_after_forward=(config.fsdp_sharding_strategy == "full_shard"),
        auto_wrap_policy="transformer_based_wrap",
        cpu_offload=config.fsdp_cpu_offload,
        cpu_ram_efficient_loading=True,
    )
    mp_label = {"bfloat16": "bf16", "float16": "fp16"}.get(config.dtype)
    if mp_label:
        plugin.set_mixed_precision(mp_label)
    return plugin


def _evaluate_validation(
    model: Any,
    val_loader: DataLoader,
    input_device: str,
    accelerator: Any,
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
    global_loss = accelerator.reduce(
        torch.tensor(total_loss, device=input_device), reduction="sum"
    ).item()
    global_tokens = accelerator.reduce(
        torch.tensor(float(total_tokens), device=input_device), reduction="sum"
    ).item()
    if global_tokens == 0:
        raise ValueError("Validation loader yielded zero tokens globally.")
    return global_loss / global_tokens


def _save_checkpoint(
    model: Any,
    tokenizer: Any,
    path: Path,
    accelerator: Any,
    train_mode: str,
) -> None:
    """Save a model checkpoint, handling both single-device and FSDP2."""
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        path.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    unwrapped = accelerator.unwrap_model(model)

    if train_mode == "lora":
        # LoRA: merge adapter weights and save full model (single-device only;
        # config validation prevents FSDP + LoRA).
        if accelerator.is_main_process:
            merged = maybe_merge_lora_for_export(unwrapped, train_mode)
            merged.save_pretrained(str(path), safe_serialization=True)
            tokenizer.save_pretrained(str(path))
    else:
        state_dict = accelerator.get_state_dict(model)
        unwrapped.save_pretrained(
            str(path),
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(str(path))

    accelerator.wait_for_everyone()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def train_unlearning(config: TrainerConfig) -> dict[str, Any]:
    """Run unlearning training. Works for single-device and multi-GPU FSDP2.

    Single-device: ``python script.py``
    Multi-GPU FSDP2: ``accelerate launch --num_processes N script.py``
                     (with ``distributed_strategy: fsdp`` in the config)
    """
    from accelerate import Accelerator

    config.validate()

    # --- Accelerator ---------------------------------------------------------
    fsdp_plugin = (
        _build_fsdp_plugin(config) if config.distributed_strategy == "fsdp" else None
    )
    accelerator = Accelerator(
        # Precision is controlled by model dtype / FSDP mixed-precision policy,
        # not by accelerate's autocast.
        mixed_precision="no",
        # We manage the accumulation loop ourselves (dual dataloaders, custom
        # loss scaling) so tell accelerate not to interfere.
        gradient_accumulation_steps=1,
        fsdp_plugin=fsdp_plugin,
    )

    is_distributed = accelerator.num_processes > 1
    is_main = accelerator.is_main_process

    # For single-device honour the user's device preference (cpu / mps / cuda:N).
    # For multi-GPU the device is assigned by the launcher.
    if is_distributed:
        device = str(accelerator.device)
    else:
        device = resolve_device(config.device)

    dist_ctx = DistributedContext(
        strategy=config.distributed_strategy,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        local_rank=accelerator.local_process_index,
        is_distributed=is_distributed,
        is_main_process=is_main,
        device=device,
    )

    set_seed(config.seed + dist_ctx.rank)
    torch_dtype = resolve_dtype(device, config.dtype)
    objective = build_objective(config)

    # --- Run directory -------------------------------------------------------
    run_name = _resolve_run_name(config, is_main, is_distributed)
    output_dir = Path(config.output_dir) / run_name
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # --- Logging -------------------------------------------------------------
    logging_state: _LoggingState | None = None
    try:
        if is_main:
            logging_state = _setup_main_process_logging(output_dir / "train.log")
            logger.info(
                "Starting FSID train: model=%s objective=%s train_mode=%s "
                "strategy=%s world_size=%d",
                config.model_name,
                config.objective,
                config.train_mode,
                config.distributed_strategy,
                accelerator.num_processes,
            )
            logger.info("Run directory: %s", output_dir)
            logger.info(
                "Resolved device=%s dtype=%s grad_accum=%d batch_size=%d max_steps=%d",
                device,
                torch_dtype,
                config.gradient_accum_steps,
                config.batch_size,
                config.max_steps,
            )
            config_snapshot = asdict(config)
            config_snapshot["_run_name"] = run_name
            config_snapshot["_run_dir"] = str(output_dir)
            with open(output_dir / "config.yaml", "w") as f:
                import yaml

                yaml.dump(
                    config_snapshot,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )

        # --- Tokenizer -------------------------------------------------------
        if is_main:
            logger.info("Loading tokenizer")
        with accelerator.main_process_first():
            tokenizer = load_tokenizer(config.model_name)
        pad_id = int(tokenizer.pad_token_id)

        # --- Datasets --------------------------------------------------------
        if is_main:
            logger.info("Loading forget set from %s", config.forget_path)
        forget_rows = load_local_records(config.forget_path)
        if is_main:
            logger.info("Building forget pairs from %d rows", len(forget_rows))
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
                "No forget training pairs could be built. "
                f"Check forget_path={config.forget_path!r} and "
                f"forget_text_col={config.forget_text_col!r}."
            )
        if is_main:
            logger.info("Built %d forget pairs", len(forget_pairs))

        retain_rows: list[dict[str, Any]] = []
        if config.retain_path:
            if is_main:
                logger.info("Loading retain set from %s", config.retain_path)
            retain_rows = load_local_records(config.retain_path)
            if config.max_retain_records is not None:
                retain_rows = retain_rows[: config.max_retain_records]
        elif is_main:
            logger.info("No retain set configured")
        retain_examples = tokenize_text_rows(
            retain_rows,
            tokenizer,
            text_col=config.retain_text_col,
            max_seq_len=config.max_seq_len,
        )
        if is_main and retain_examples:
            logger.info("Prepared %d retain sequences", len(retain_examples))

        val_examples: list[dict[str, Any]] = []
        if config.validation_path:
            if is_main:
                logger.info(
                    "Loading validation set from %s",
                    config.validation_path,
                )
            val_rows = load_local_records(config.validation_path)
            val_examples = tokenize_text_rows(
                val_rows,
                tokenizer,
                text_col=config.validation_text_col,
                max_seq_len=config.max_seq_len,
            )
            if not val_examples:
                raise ValueError(
                    f"No validation sequences from "
                    f"{config.validation_path!r}. "
                    f"Check validation_text_col="
                    f"{config.validation_text_col!r}."
                )
            if len(val_examples) > config.eval_max_samples:
                val_examples = val_examples[: config.eval_max_samples]
            if is_main:
                logger.info(
                    "Prepared %d validation sequences",
                    len(val_examples),
                )

        forget_dataset = ForgetPairDataset(forget_pairs)
        retain_dataset: TextSequenceDataset | None = (
            TextSequenceDataset(retain_examples) if retain_examples else None
        )
        val_dataset: TextSequenceDataset | None = (
            TextSequenceDataset(val_examples) if val_examples else None
        )

        # --- Model -----------------------------------------------------------
        if is_main:
            logger.info("Loading model")
        with accelerator.main_process_first():
            model = load_train_model(
                config,
                device,
                torch_dtype,
                move_to_device=(not is_distributed),
            )

        # Tell the FSDP2 plugin which transformer layer class to shard around.
        if fsdp_plugin is not None:
            cls_names = list(getattr(model, "_no_split_modules", []) or [])
            if not cls_names:
                cls_names = [type(_resolve_transformer_layer(model, 0)).__name__]
            fsdp_plugin.transformer_cls_names_to_wrap = cls_names
            if is_main:
                logger.info("FSDP2 wrap targets: %s", cls_names)

        # --- Optimizer & scheduler -------------------------------------------
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found.")

        num_trainable = sum(p.numel() for p in trainable_params)
        num_total = sum(p.numel() for p in model.parameters())

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

        # --- Dataloaders (managed manually for cycling behaviour) ------------
        pin_memory = device.startswith("cuda")

        forget_sampler = build_training_sampler(
            forget_dataset,
            shuffle=True,
            seed=config.seed,
            context=dist_ctx,
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

        retain_state: _LoaderState | None = None
        needs_retain = retain_dataset is not None and (
            config.retain_weight > 0 or objective.owns_retain_loss
        )
        if needs_retain:
            retain_sampler = build_training_sampler(
                retain_dataset,
                shuffle=True,
                seed=config.seed + 17,
                context=dist_ctx,
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
                partition_dataset(val_dataset, dist_ctx),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=lambda batch: collate_text_sequences(batch, pad_id),
                pin_memory=pin_memory,
            )

        # --- Prepare model/optimizer/scheduler with Accelerate ---------------
        if is_main:
            logger.info("Preparing model with Accelerate")
        model, optimizer, scheduler = accelerator.prepare(
            model,
            optimizer,
            scheduler,
        )
        input_device = str(accelerator.device)

        # --- Reference log-probs (for NPO / objectives that need them) -------
        if objective.needs_ref_logps:
            if is_main:
                logger.info(
                    "Precomputing reference log-probs for %d forget pairs",
                    len(forget_dataset),
                )
            precompute_reference_logps(
                model=model,
                dataset=forget_dataset,
                batch_size=config.batch_size,
                pad_id=pad_id,
                input_device=input_device,
                num_workers=config.num_workers,
                use_sums=objective.ref_logps_are_sums,
                distributed_context=dist_ctx,
            )
            if is_main:
                logger.info("Reference log-prob precompute complete")

        objective.setup(model, config)

        # --- Training loop ---------------------------------------------------
        run_start = time.time()
        history: list[dict[str, Any]] = []
        if is_main:
            logger.info(
                "Starting training: %d trainable / %d total parameters",
                num_trainable,
                num_total,
            )

        try:
            for step in range(1, config.max_steps + 1):
                optimizer.zero_grad(set_to_none=True)

                step_forget_loss = 0.0
                step_retain_loss = 0.0
                step_total_loss = 0.0

                for micro in range(config.gradient_accum_steps):
                    # Suppress gradient sync on all but the last micro-step.
                    # For FSDP2 this is a no-op (sync is cheap per-step) but
                    # it helps DDP and future backends.
                    last_micro = micro == config.gradient_accum_steps - 1
                    sync_ctx = (
                        nullcontext() if last_micro else accelerator.no_sync(model)
                    )

                    with sync_ctx:
                        fb = _to_device(
                            _next_batch(forget_state),
                            input_device,
                        )
                        rb: dict[str, Any] | None = None
                        if retain_state is not None:
                            rb = _to_device(
                                _next_batch(retain_state),
                                input_device,
                            )

                        total_loss, loss_forget, loss_retain = objective.compute_loss(
                            model,
                            fb,
                            rb,
                            config.retain_weight,
                        )

                        step_forget_loss += loss_forget.item()
                        step_retain_loss += loss_retain.item()
                        step_total_loss += total_loss.item()

                        accelerator.backward(
                            total_loss / config.gradient_accum_steps,
                        )

                if config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        config.max_grad_norm,
                    )
                optimizer.step()
                scheduler.step()

                # --- Per-step logging ------------------------------------
                mean_forget = step_forget_loss / config.gradient_accum_steps
                mean_retain = step_retain_loss / config.gradient_accum_steps
                mean_total = step_total_loss / config.gradient_accum_steps
                current_lr = scheduler.get_last_lr()[0]

                mean_forget = accelerator.reduce(
                    torch.tensor(mean_forget, device=input_device), reduction="mean"
                ).item()
                mean_retain = accelerator.reduce(
                    torch.tensor(mean_retain, device=input_device), reduction="mean"
                ).item()
                mean_total = accelerator.reduce(
                    torch.tensor(mean_total, device=input_device), reduction="mean"
                ).item()

                should_log = (
                    step == 1
                    or step % config.log_every == 0
                    or step == config.max_steps
                )
                should_eval = val_loader is not None and (
                    step == 1
                    or step % config.eval_every == 0
                    or step == config.max_steps
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
                            accelerator,
                        )
                        if is_main:
                            logger.info(
                                "step=%d forget=%.4f retain=%.4f val=%.4f lr=%.2e",
                                step,
                                mean_forget,
                                mean_retain,
                                entry["validation_loss"],
                                current_lr,
                            )
                    elif is_main:
                        logger.info(
                            "step=%d forget=%.4f retain=%.4f lr=%.2e",
                            step,
                            mean_forget,
                            mean_retain,
                            current_lr,
                        )
                    if is_main:
                        history.append(entry)

                # --- Intermediate checkpoints ----------------------------
                if (
                    config.save_every > 0
                    and step % config.save_every == 0
                    and step < config.max_steps
                ):
                    ckpt_dir = output_dir / f"checkpoint-{step}"
                    _save_checkpoint(
                        model,
                        tokenizer,
                        ckpt_dir,
                        accelerator,
                        config.train_mode,
                    )
                    if is_main:
                        logger.info("Saved checkpoint to %s", ckpt_dir)

        finally:
            objective.teardown()

        # --- Final checkpoint ------------------------------------------------
        run_seconds = time.time() - run_start
        final_ckpt = output_dir / "checkpoint-final"
        _save_checkpoint(
            model,
            tokenizer,
            final_ckpt,
            accelerator,
            config.train_mode,
        )

        # --- Summary ---------------------------------------------------------
        summary: dict[str, Any] = {
            "run_name": run_name,
            "run_dir": str(output_dir),
            "config": asdict(config),
            "device": device,
            "input_device": input_device,
            "dtype": str(torch_dtype),
            "distributed_strategy": config.distributed_strategy,
            "world_size": accelerator.num_processes,
            "rank": accelerator.process_index,
            "is_main_process": is_main,
            "num_forget_rows": len(forget_rows),
            "num_forget_pairs": len(forget_pairs),
            "num_retain_rows": len(retain_rows),
            "num_retain_sequences": len(retain_examples),
            "num_val_sequences": len(val_examples),
            "num_trainable_params": num_trainable,
            "num_total_params": num_total,
            "run_seconds": run_seconds,
            "history": history if is_main else [],
            "checkpoint_dir": str(final_ckpt),
        }
        if is_main:
            with open(output_dir / "train_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Run complete: %s (%.1fs)", run_name, run_seconds)

        accelerator.end_training()
        return summary

    finally:
        _teardown_main_process_logging(logging_state)
