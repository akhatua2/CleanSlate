"""Configuration for FSID unlearning training."""

from dataclasses import dataclass, fields as dataclass_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, get_args

import yaml  # type: ignore[import-untyped]


type ObjectiveType = Literal["ga", "npo", "simnpo", "rmu", "undial"]
type DistributedStrategyType = Literal["single", "fsdp"]
type FSDPShardingStrategyType = Literal["full_shard", "shard_grad_op"]
type RetainLossType = Literal["nll", "kl", "embed_diff"]
type DType = Literal["bfloat16", "float16", "float32"]

ALLOWED_OBJECTIVES: tuple[str, ...] = get_args(ObjectiveType.__value__)
ALLOWED_DTYPES: tuple[str, ...] = get_args(DType.__value__)
ALLOWED_DISTRIBUTED_STRATEGIES: tuple[str, ...] = get_args(DistributedStrategyType.__value__)
ALLOWED_FSDP_SHARDING_STRATEGIES: tuple[str, ...] = get_args(FSDPShardingStrategyType.__value__)
ALLOWED_RETAIN_LOSS_TYPES: tuple[str, ...] = get_args(RetainLossType.__value__)


@dataclass
class TrainerConfig:
    """Config for FSID unlearning training."""

    model_name: str = ""
    forget_path: str = ""
    retain_path: str | None = None
    output_dir: str = "runs"

    objective: ObjectiveType = "npo"

    num_epochs: int = 5
    batch_size: int = 2
    gradient_accum_steps: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 20
    max_grad_norm: float = 1.0
    forget_weight: float = 1.0
    retain_weight: float = 1.0
    retain_loss_type: RetainLossType = "nll"

    npo_beta: float = 0.1
    simnpo_delta: float = 0.0

    # RMU
    rmu_steering_coeff: float = 2.0
    rmu_module_regex: str = r"model\.layers\.7"
    # Separate generator seed for the random control vector so runs are
    # reproducible and independent of upstream RNG consumption order.
    rmu_control_vec_seed: int = 0

    # UNDIAL
    undial_beta: float = 10.0

    validation_path: str | None = None
    eval_every: int = 50
    eval_max_samples: int = 200

    save_every: int = 0

    distributed_strategy: DistributedStrategyType = "single"
    fsdp_sharding_strategy: FSDPShardingStrategyType = "full_shard"
    fsdp_cpu_offload: bool = False

    device: str = "auto"
    dtype: DType = "bfloat16"
    seed: int = 42
    num_workers: int = 0
    log_every: int = 10
    gradient_checkpointing: bool = False
    torch_compile: bool = False

    def make_run_name(self) -> str:
        """Timestamped run name."""
        model_short = self.model_name.split("/")[-1]
        return (
            f"{self.objective}_{model_short}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    def validate(self) -> None:
        """Validate constraints. Raises ValueError on any violation."""
        if not self.model_name:
            raise ValueError("model_name is required")
        if not self.forget_path:
            raise ValueError("forget_path is required")
        if self.objective not in ALLOWED_OBJECTIVES:
            raise ValueError(f"objective must be one of {list(ALLOWED_OBJECTIVES)}, got {self.objective!r}")
        if self.dtype not in ALLOWED_DTYPES:
            raise ValueError(f"dtype must be one of {list(ALLOWED_DTYPES)}, got {self.dtype!r}")
        if self.distributed_strategy not in ALLOWED_DISTRIBUTED_STRATEGIES:
            raise ValueError(f"distributed_strategy must be one of {list(ALLOWED_DISTRIBUTED_STRATEGIES)}")
        if self.fsdp_sharding_strategy not in ALLOWED_FSDP_SHARDING_STRATEGIES:
            raise ValueError(f"fsdp_sharding_strategy must be one of {list(ALLOWED_FSDP_SHARDING_STRATEGIES)}")
        if self.retain_loss_type not in ALLOWED_RETAIN_LOSS_TYPES:
            raise ValueError(f"retain_loss_type must be one of {list(ALLOWED_RETAIN_LOSS_TYPES)}")

        positive_ints: dict[str, int] = {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accum_steps": self.gradient_accum_steps,
            "log_every": self.log_every,
            "eval_every": self.eval_every,
            "eval_max_samples": self.eval_max_samples,
        }
        for pos_name, pos_value in positive_ints.items():
            if pos_value < 1:
                raise ValueError(f"{pos_name} must be >= 1, got {pos_value}")

        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.npo_beta <= 0:
            raise ValueError("npo_beta must be > 0")
        if self.forget_weight < 0 or self.retain_weight < 0:
            raise ValueError("forget_weight and retain_weight must be >= 0")
        if self.warmup_steps < 0 or self.save_every < 0:
            raise ValueError("warmup_steps and save_every must be >= 0")

        if self.retain_weight > 0 and not self.retain_path:
            raise ValueError("retain_weight > 0 requires retain_path")
        if self.objective == "rmu":
            if not self.retain_path or self.retain_weight <= 0:
                raise ValueError("rmu requires retain_path and retain_weight > 0")
            if self.retain_loss_type != "embed_diff":
                raise ValueError("rmu requires retain_loss_type='embed_diff'")
        if self.distributed_strategy == "fsdp" and self.device in ("cpu", "mps"):
            raise ValueError("FSDP requires CUDA devices")

    def to_training_arguments(self, *, run_name: str | None = None) -> Any:
        """Convert to HF TrainingArguments."""
        from transformers import TrainingArguments

        run_name = run_name or self.make_run_name()
        output_dir = str(Path(self.output_dir) / run_name)

        fsdp_value = ""
        fsdp_config: dict[str, Any] | None = None
        use_gradient_checkpointing = self.gradient_checkpointing
        if self.distributed_strategy == "fsdp":
            fsdp_value = f"{self.fsdp_sharding_strategy} auto_wrap"
            if self.fsdp_cpu_offload:
                fsdp_value += " offload"
            fsdp_config = {
                "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "backward_prefetch": "backward_pre",
                "forward_prefetch": True,
                "limit_all_gathers": True,
                "sync_module_states": True,
                "cpu_ram_efficient_loading": True,
                "use_orig_params": True,
            }
            # FSDP activation checkpointing, not HF Trainer's, so shapes stay consistent on recompute.
            if self.gradient_checkpointing:
                fsdp_config["activation_checkpointing"] = True
                use_gradient_checkpointing = False

        do_eval = self.validation_path is not None

        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accum_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            num_train_epochs=self.num_epochs,
            max_grad_norm=self.max_grad_norm,
            logging_steps=self.log_every,
            save_steps=self.save_every if self.save_every > 0 else 999_999,
            save_strategy="steps" if self.save_every > 0 else "no",
            eval_steps=self.eval_every if do_eval else None,
            eval_strategy="steps" if do_eval else "no",
            seed=self.seed,
            bf16=(self.dtype == "bfloat16"),
            fp16=(self.dtype == "float16"),
            gradient_checkpointing=use_gradient_checkpointing,
            fsdp=fsdp_value,
            fsdp_config=fsdp_config,
            torch_compile=self.torch_compile,
            dataloader_num_workers=self.num_workers,
            remove_unused_columns=False,
            run_name=run_name,
            logging_first_step=True,
            report_to="none",
            lr_scheduler_type="linear",
            save_total_limit=2,
            load_best_model_at_end=False,
            disable_tqdm=False,
        )


def load_config(
    yaml_path: str | Path,
    *,
    cli_overrides: dict[str, Any] | None = None,
) -> TrainerConfig:
    """Load TrainerConfig from YAML, applying non-None CLI overrides on top."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(raw).__name__}")

    if cli_overrides:
        raw.update({k: v for k, v in cli_overrides.items() if v is not None})

    valid_fields = {f.name for f in dataclass_fields(TrainerConfig)}
    unknown = set(raw) - valid_fields
    if unknown:
        raise ValueError(
            f"Unknown config key(s): {sorted(unknown)}. Valid keys: {sorted(valid_fields)}"
        )
    config = TrainerConfig(**{k: v for k, v in raw.items() if v is not None})
    config.validate()
    return config
