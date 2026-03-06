"""Model and tokenizer setup helpers for FSID training."""

import random
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from cleanslate.fsid.config import TrainerConfig
from cleanslate.fsid.datasets.unlearning import ForgetPairDataset, collate_forget
from cleanslate.fsid.training.distributed import (
    DistributedContext,
    gather_indexed_scalars,
)
from torch.utils.data.distributed import DistributedSampler
from cleanslate.fsid.training.objectives.utils import (
    conditional_avg_log_probs,
    conditional_sum_log_probs,
)


def resolve_device(device: str) -> str:
    """Resolve to a single training device."""
    if device == "auto":
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is unavailable")
        return f"cuda:{torch.cuda.current_device()}"
    return device


def resolve_dtype(device: str, dtype_name: str) -> torch.dtype:
    if device in ("cpu", "mps"):
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_input_device(model: Any) -> str:
    return str(next(model.parameters()).device)


def load_tokenizer(
    model_name: str,
    *,
    local_files_only: bool = False,
) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def apply_lora_if_requested(model: Any, config: TrainerConfig) -> Any:
    if config.train_mode != "lora":
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError("LoRA mode requires peft. Install with: pip install peft") from e

    target_modules = [
        module.strip()
        for module in config.lora_target_modules.split(",")
        if module.strip()
    ]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    return get_peft_model(model, lora_cfg)


def load_train_model(
    config: TrainerConfig,
    device: str,
    torch_dtype: torch.dtype,
    *,
    local_files_only: bool = False,
    move_to_device: bool = True,
) -> Any:
    """Load the single-device train model."""
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    # Some model configs (e.g. OLMo) store rope_scaling fields as ints
    # but transformers validates them as floats.  Patch in-place.
    rope_cfg = getattr(getattr(model, "config", None), "rope_scaling", None)
    if isinstance(rope_cfg, dict):
        for key in ("beta_slow", "beta_fast", "factor"):
            if key in rope_cfg and isinstance(rope_cfg[key], int):
                rope_cfg[key] = float(rope_cfg[key])
    model = apply_lora_if_requested(model, config)
    if move_to_device:
        model.to(device)

    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.train()
    return model


def precompute_reference_logps(
    model: Any,
    dataset: ForgetPairDataset,
    *,
    batch_size: int,
    pad_id: int,
    input_device: str,
    num_workers: int,
    use_sums: bool = False,
    distributed_context: DistributedContext | None = None,
) -> None:
    """Precompute frozen reference log-probs and cache them in the dataset."""
    if distributed_context is None:
        distributed_context = DistributedContext(
            strategy="single",
            rank=0,
            world_size=1,
            local_rank=0,
            is_distributed=False,
            is_main_process=True,
            device=input_device,
        )

    # Use DistributedSampler instead of partition_dataset so that every rank
    # iterates the same number of batches.  With FSDP the forward pass itself
    # is a collective (all-gather for parameters), so if one rank exits the
    # loop early the remaining ranks deadlock waiting for it.
    sampler: DistributedSampler | None = None
    if distributed_context.is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed_context.world_size,
            rank=distributed_context.rank,
            shuffle=False,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_forget(batch, pad_id),
        pin_memory=input_device.startswith("cuda"),
    )

    local_values: dict[int, float] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(input_device)
            attention_mask = batch["attention_mask"].to(input_device)
            seed_lens = batch["seed_lens"].to(input_device)

            if use_sums:
                log_probs, _ = conditional_sum_log_probs(
                    model,
                    input_ids,
                    attention_mask,
                    seed_lens,
                )
            else:
                log_probs, _ = conditional_avg_log_probs(
                    model,
                    input_ids,
                    attention_mask,
                    seed_lens,
                )

            for index, value in zip(batch["indices"].tolist(), log_probs.cpu().tolist()):
                local_values[int(index)] = float(value)

    values = gather_indexed_scalars(
        local_values,
        dataset_len=len(dataset),
        context=distributed_context,
    )
    dataset.set_reference_logps(values)
    model.train()


def maybe_merge_lora_for_export(model: Any, train_mode: str) -> Any:
    """Return a model object that can be reloaded by the eval pipeline."""
    if train_mode != "lora":
        return model
    if not hasattr(model, "merge_and_unload"):
        raise ValueError(
            "LoRA model does not support merge_and_unload; cannot export eval-ready checkpoint"
        )
    return model.merge_and_unload()
