"""Configuration types for local FSID unlearning runs."""

from dataclasses import dataclass, fields as dataclass_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


ObjectiveType = Literal["ga", "grad_diff", "npo", "simnpo"]
TrainModeType = Literal["full", "lora"]
DistributedStrategyType = Literal["single", "fsdp"]
FSDPShardingStrategyType = Literal["full_shard", "shard_grad_op"]

ALLOWED_TRAIN_MODES: tuple[str, ...] = ("full", "lora")
ALLOWED_DTYPES: tuple[str, ...] = ("bfloat16", "float16", "float32")
ALLOWED_DISTRIBUTED_STRATEGIES: tuple[str, ...] = ("single", "fsdp")
ALLOWED_FSDP_SHARDING_STRATEGIES: tuple[str, ...] = (
    "full_shard",
    "shard_grad_op",
)


def get_allowed_objectives() -> list[str]:
    """Return the runtime objective list."""
    from cleanslate.fsid.training.trainer import list_objectives

    return list_objectives()


@dataclass
class TrainerConfig:
    """Config for local unlearning training."""

    model_name: str = ""
    forget_path: str = ""

    retain_path: str | None = None
    output_dir: str = "runs"

    objective: ObjectiveType = "npo"
    train_mode: TrainModeType = "full"

    forget_text_col: str = "lyrics"
    retain_text_col: str = "lyrics"
    window_size: int = 50
    stride: int = 25
    max_windows_per_song: int | None = 64
    max_seq_len: int = 512
    max_forget_pairs: int | None = None
    max_retain_records: int | None = None

    num_epochs: int = 5
    batch_size: int = 2
    gradient_accum_steps: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 20
    max_grad_norm: float = 1.0
    retain_weight: float = 1.0

    npo_beta: float = 0.1
    simnpo_gamma: float = 0.1

    validation_path: str | None = None
    validation_text_col: str = "lyrics"
    eval_every: int = 50
    eval_max_samples: int = 200

    save_every: int = 0

    distributed_strategy: DistributedStrategyType = "single"
    fsdp_sharding_strategy: FSDPShardingStrategyType = "full_shard"
    fsdp_cpu_offload: bool = False

    device: str = "auto"
    dtype: str = "bfloat16"
    seed: int = 42
    num_workers: int = 0
    log_every: int = 10
    gradient_checkpointing: bool = False

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = (
        "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj"
    )

    def make_run_name(self) -> str:
        """Build the default timestamped run name."""
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
        allowed_objectives = get_allowed_objectives()
        if self.objective not in allowed_objectives:
            raise ValueError(
                f"Unsupported objective '{self.objective}'. "
                f"Allowed: {allowed_objectives}"
            )
        if self.train_mode not in ALLOWED_TRAIN_MODES:
            raise ValueError(
                f"Unsupported train_mode '{self.train_mode}'. "
                f"Allowed: {list(ALLOWED_TRAIN_MODES)}"
            )
        if self.dtype not in ALLOWED_DTYPES:
            raise ValueError(
                f"Unsupported dtype '{self.dtype}'. "
                f"Allowed: {list(ALLOWED_DTYPES)}"
            )
        if self.distributed_strategy not in ALLOWED_DISTRIBUTED_STRATEGIES:
            raise ValueError(
                f"Unsupported distributed_strategy '{self.distributed_strategy}'. "
                f"Allowed: {list(ALLOWED_DISTRIBUTED_STRATEGIES)}"
            )
        if self.fsdp_sharding_strategy not in ALLOWED_FSDP_SHARDING_STRATEGIES:
            raise ValueError(
                f"Unsupported fsdp_sharding_strategy '{self.fsdp_sharding_strategy}'. "
                f"Allowed: {list(ALLOWED_FSDP_SHARDING_STRATEGIES)}"
            )
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.gradient_accum_steps < 1:
            raise ValueError("gradient_accum_steps must be >= 1")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.max_seq_len < 8:
            raise ValueError("max_seq_len must be >= 8")
        if self.max_seq_len <= self.window_size:
            raise ValueError(
                "max_seq_len must be > window_size to yield non-empty forget targets"
            )
        if self.max_windows_per_song is not None and self.max_windows_per_song < 1:
            raise ValueError("max_windows_per_song must be >= 1 when set")
        if self.max_forget_pairs is not None and self.max_forget_pairs < 1:
            raise ValueError("max_forget_pairs must be >= 1 when set")
        if self.max_retain_records is not None and self.max_retain_records < 1:
            raise ValueError("max_retain_records must be >= 1 when set")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.npo_beta <= 0:
            raise ValueError("npo_beta must be > 0")
        if self.retain_weight < 0:
            raise ValueError("retain_weight must be >= 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1")
        if self.eval_every < 1:
            raise ValueError("eval_every must be >= 1")
        if self.eval_max_samples < 1:
            raise ValueError("eval_max_samples must be >= 1")
        if self.save_every < 0:
            raise ValueError("save_every must be >= 0")

        if self.retain_weight > 0 and not self.retain_path:
            raise ValueError(
                "retain_weight > 0 requires retain_path so the retain term is real."
            )
        if self.objective == "grad_diff":
            if not self.retain_path:
                raise ValueError("grad_diff requires retain_path")
            if self.retain_weight <= 0:
                raise ValueError("grad_diff requires retain_weight > 0")
        if self.train_mode == "lora" and self.save_every > 0:
            raise ValueError(
                "save_every > 0 is not supported for LoRA runs because checkpoints "
                "are exported as merged full-model weights at the end."
            )
        if self.distributed_strategy == "fsdp":
            if self.train_mode != "full":
                raise ValueError("FSDP currently only supports train_mode='full'")
            if self.device in ("cpu", "mps"):
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
            fsdp_config = {
                "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "cpu_offload": self.fsdp_cpu_offload,
            }
            # With FSDP, use FSDP's activation checkpointing instead of
            # HF Trainer's gradient_checkpointing to avoid shape mismatches
            # during checkpoint recomputation.
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
            save_steps=self.save_every if self.save_every > 0 else 999999,
            save_strategy="steps" if self.save_every > 0 else "no",
            eval_steps=self.eval_every if do_eval else None,
            eval_strategy="steps" if do_eval else "no",
            seed=self.seed,
            bf16=(self.dtype == "bfloat16"),
            fp16=(self.dtype == "float16"),
            gradient_checkpointing=use_gradient_checkpointing,
            fsdp=fsdp_value,
            fsdp_config=fsdp_config,
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
    """Load TrainerConfig from a YAML file, then apply CLI overrides."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "pyyaml is required for YAML config loading. "
            "Install with: pip install pyyaml"
        ) from e

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(raw).__name__}")

    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                raw[key] = value

    valid_fields = {field.name for field in dataclass_fields(TrainerConfig)}
    unknown = set(raw.keys()) - valid_fields
    if unknown:
        raise ValueError(
            f"Unknown config key(s): {sorted(unknown)}. "
            f"Valid keys: {sorted(valid_fields)}"
        )

    filtered = {key: value for key, value in raw.items() if value is not None}
    config = TrainerConfig(**filtered)
    config.validate()
    return config
