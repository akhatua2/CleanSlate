"""cleanslate fsid -- forget-set identification and unlearning pipeline.

Subcommands
    build-pool : build a stratified forget-request pool from an eval dataset.
    curate     : turn a forget-request pool into (forget_set, retain_set) artifacts.
    train      : run unlearning training.
"""

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from cleanslate.cli.aliases import resolve_model
from cleanslate.fsid.config import (
    ALLOWED_DISTRIBUTED_STRATEGIES,
    ALLOWED_FSDP_SHARDING_STRATEGIES,
    ALLOWED_OBJECTIVES,
    ALLOWED_RETAIN_LOSS_TYPES,
    TrainerConfig,
    load_config,
)
from cleanslate.fsid.datasets.io import load_local_records
from cleanslate.fsid.datasets.pool import (
    build_candidate_pool,
    save_pool_datasets,
    split_pool_records,
)
from cleanslate.fsid.rules import (
    get_forget_set_rule,
    list_forget_set_rules,
    save_curated_sets,
)

fsid_app = typer.Typer(
    name="fsid",
    help="Forget-Set IDentification: build pools, curate, and train.",
    no_args_is_help=True,
)
console = Console()


@fsid_app.command("build-pool")
def build_pool(
    eval_dataset: str = typer.Option(..., "--eval-dataset", help="Eval HF dataset ID"),
    fraction_col: str = typer.Option("memorized_fraction", "--fraction-col"),
    threshold: float = typer.Option(0.30, "--threshold", "-t"),
    target_size: int = typer.Option(100, "--target-size"),
    min_size: int = typer.Option(50, "--min-size"),
    max_size: int = typer.Option(200, "--max-size"),
    num_bins: int = typer.Option(5, "--num-bins"),
    seed: int = typer.Option(42, "--seed"),
    deduplicate: bool = typer.Option(True, "--deduplicate/--no-deduplicate"),
    forget_fraction: float | None = typer.Option(
        None, "--forget-fraction", help="Fraction of pool marked for unlearning, in (0, 1]"
    ),
    output_root: str = typer.Option("data/fsid/pool", "--output-dir", "-o"),
) -> None:
    """Stratified memorization pool → pool.parquet + forget.parquet + retain.parquet."""
    pool_rows, stats = build_candidate_pool(
        eval_dataset=eval_dataset,
        fraction_col=fraction_col,
        threshold=threshold,
        target_size=target_size,
        min_size=min_size,
        max_size=max_size,
        num_bins=num_bins,
        seed=seed,
        deduplicate=deduplicate,
    )
    forget_rows, retain_rows = split_pool_records(
        pool_rows, forget_fraction=forget_fraction, num_bins=num_bins, seed=seed,
    )
    outputs = save_pool_datasets(
        pool_records=pool_rows,
        forget_records=forget_rows,
        retain_records=retain_rows,
        output_root=output_root,
    )
    console.print(f"[bold]Saved pool:[/bold] {outputs['pool']}")
    console.print(f"[bold]Saved forget songs:[/bold] {outputs['forget']}")
    console.print(f"[bold]Saved retain songs:[/bold] {outputs['retain']}")
    console.print(f"[bold]Rows scanned:[/bold] {stats.total_rows_seen}")
    console.print(f"[bold]Eligible rows:[/bold] {stats.eligible_rows}")
    console.print(f"[bold]Pool rows:[/bold] {stats.selected_rows}")
    console.print(f"[bold]Forget rows:[/bold] {len(forget_rows)}")
    console.print(f"[bold]Retain rows:[/bold] {len(retain_rows)}")
    console.print(
        f"[bold]Mem fraction:[/bold] min={stats.min_fraction:.3f} "
        f"mean={stats.mean_fraction:.3f} max={stats.max_fraction:.3f}"
    )


@fsid_app.command("curate")
def curate(
    forget_set_rule: str = typer.Option(
        "baseline",
        "--rule",
        help=f"Curation rule. Allowed: {', '.join(list_forget_set_rules())}",
    ),
    forget_pool_path: str = typer.Option(
        "data/fsid/pool/forget.parquet", "--forget-pool-path",
    ),
    window_size: int = typer.Option(
        100, "--window-size", help="Prefix/suffix chars (baseline rule only; must match eval)"
    ),
    stride: int = typer.Option(
        10, "--stride", help="Window stride in chars (baseline rule only; must match eval)"
    ),
    base_dataset: str | None = typer.Option(None, "--base-dataset"),
    seed: int = typer.Option(42, "--seed"),
    output_dir: str | None = typer.Option(None, "--output-dir", "-o"),
    retain_dataset: str = typer.Option("wikitext", "--retain-dataset"),
    retain_config: str = typer.Option("wikitext-103-raw-v1", "--retain-config"),
    retain_split: str = typer.Option("validation", "--retain-split"),
    retain_samples: int | None = typer.Option(None, "--retain-samples"),
) -> None:
    """Turn a forget-song pool into (forget_set.parquet, retain_set.parquet, metadata.json)."""
    try:
        rule = get_forget_set_rule(forget_set_rule)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--rule") from exc

    forget_song_rows = load_local_records(forget_pool_path)
    forget_rows, retain_rows = rule.curate(
        forget_song_rows,
        window_size=window_size,
        stride=stride,
        base_dataset=base_dataset,
        seed=seed,
        retain_dataset=retain_dataset,
        retain_config=retain_config,
        retain_split=retain_split,
        retain_samples=retain_samples,
    )

    out_dir = output_dir or str(Path("data/fsid/curated") / rule.name)
    paths = save_curated_sets(
        output_dir=out_dir,
        forget_rows=forget_rows,
        retain_rows=retain_rows,
        metadata={
            "forget_set_rule": rule.name,
            "rule_name": rule.artifact_rule_name,
            "window_size": window_size if rule.name == "baseline" else None,
            "stride": stride if rule.name == "baseline" else None,
            "base_dataset": base_dataset,
            "forget_pool_path": str(Path(forget_pool_path).resolve()),
            "seed": seed,
        },
    )

    console.print(f"[bold]Rule:[/bold] {rule.name}")
    console.print(f"[bold]Forget rows:[/bold] {len(forget_rows)}")
    console.print(f"[bold]Retain rows:[/bold] {len(retain_rows)}")
    console.print(f"[bold]Forget set:[/bold] {paths['forget']}")
    console.print(f"[bold]Retain set:[/bold] {paths['retain']}")
    console.print(f"[bold]Metadata:[/bold] {paths['metadata']}")


@fsid_app.command("train")
def train(
    config_path: str | None = typer.Option(None, "--config", help="YAML config (flags override)"),
    model: str | None = typer.Option(None, "--model", "-m"),
    forget_path: str | None = typer.Option(None, "--forget-path"),
    retain_path: str | None = typer.Option(None, "--retain-path"),
    output_dir: str | None = typer.Option(None, "--output-dir", "-o"),
    objective: str | None = typer.Option(
        None, "--objective", help=f"Objective: {', '.join(ALLOWED_OBJECTIVES)}"
    ),
    num_epochs: int | None = typer.Option(None, "--epochs"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    gradient_accum_steps: int | None = typer.Option(None, "--grad-accum"),
    lr: float | None = typer.Option(None, "--lr"),
    weight_decay: float | None = typer.Option(None, "--weight-decay"),
    warmup_steps: int | None = typer.Option(None, "--warmup-steps"),
    max_grad_norm: float | None = typer.Option(None, "--max-grad-norm"),
    npo_beta: float | None = typer.Option(None, "--npo-beta"),
    simnpo_delta: float | None = typer.Option(None, "--simnpo-delta"),
    forget_weight: float | None = typer.Option(None, "--forget-weight"),
    retain_weight: float | None = typer.Option(None, "--retain-weight"),
    retain_loss_type: str | None = typer.Option(
        None, "--retain-loss-type",
        help=f"Retain loss: {', '.join(ALLOWED_RETAIN_LOSS_TYPES)}",
    ),
    device: str | None = typer.Option(None, "--device"),
    dtype: str | None = typer.Option(None, "--dtype"),
    distributed_strategy: str | None = typer.Option(
        None, "--distributed-strategy",
        help=f"Distributed: {', '.join(ALLOWED_DISTRIBUTED_STRATEGIES)}",
    ),
    fsdp_sharding_strategy: str | None = typer.Option(
        None, "--fsdp-sharding-strategy",
        help=f"FSDP sharding: {', '.join(ALLOWED_FSDP_SHARDING_STRATEGIES)}",
    ),
    fsdp_cpu_offload: bool | None = typer.Option(None, "--fsdp-cpu-offload/--no-fsdp-cpu-offload"),
    num_workers: int | None = typer.Option(None, "--num-workers"),
    seed: int | None = typer.Option(None, "--seed"),
    log_every: int | None = typer.Option(None, "--log-every"),
    save_every: int | None = typer.Option(None, "--save-every"),
    eval_every: int | None = typer.Option(None, "--eval-every"),
    eval_max_samples: int | None = typer.Option(None, "--eval-max-samples"),
    validation_path: str | None = typer.Option(None, "--validation-path"),
    gradient_checkpointing: bool | None = typer.Option(
        None, "--gradient-checkpointing/--no-gradient-checkpointing"
    ),
    torch_compile: bool | None = typer.Option(None, "--torch-compile/--no-torch-compile"),
) -> None:
    """Train an unlearned model from a YAML config (+ CLI overrides)."""
    from cleanslate.fsid.training.trainer import train_unlearning

    model_path = resolve_model(model)[0] if model else None
    cli_overrides: dict[str, Any] = {
        "model_name": model_path,
        "forget_path": forget_path,
        "retain_path": retain_path,
        "output_dir": output_dir,
        "objective": objective,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accum_steps": gradient_accum_steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "max_grad_norm": max_grad_norm,
        "npo_beta": npo_beta,
        "simnpo_delta": simnpo_delta,
        "forget_weight": forget_weight,
        "retain_weight": retain_weight,
        "retain_loss_type": retain_loss_type,
        "device": device,
        "dtype": dtype,
        "distributed_strategy": distributed_strategy,
        "fsdp_sharding_strategy": fsdp_sharding_strategy,
        "fsdp_cpu_offload": fsdp_cpu_offload,
        "num_workers": num_workers,
        "seed": seed,
        "log_every": log_every,
        "save_every": save_every,
        "eval_every": eval_every,
        "eval_max_samples": eval_max_samples,
        "validation_path": validation_path,
        "gradient_checkpointing": gradient_checkpointing,
        "torch_compile": torch_compile,
    }

    if config_path is not None:
        config = load_config(config_path, cli_overrides=cli_overrides)
    else:
        if model is None or forget_path is None:
            raise typer.BadParameter("--model and --forget-path are required without --config")
        kwargs = {k: v for k, v in cli_overrides.items() if v is not None}
        config = TrainerConfig(**kwargs)
        config.validate()

    summary = train_unlearning(config)
    if not summary.get("is_main_process", True):
        return
    console.print(f"[bold]Checkpoint:[/bold] {summary['checkpoint_dir']}")
    console.print(f"[bold]Run time:[/bold] {summary['run_seconds']:.1f}s")
    console.print(f"[bold]Forget rows:[/bold] {summary['num_forget_rows']}")
    console.print(f"[bold]Forget pairs:[/bold] {summary['num_forget_pairs']}")
    console.print(f"[bold]Retain rows:[/bold] {summary['num_retain_rows']}")
    if summary.get("num_val_sequences", 0) > 0:
        console.print(f"[bold]Val sequences:[/bold] {summary['num_val_sequences']}")


