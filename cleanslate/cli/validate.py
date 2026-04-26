"""cleanslate validate / generate / judge.

``validate`` is the end-to-end flow; ``generate`` and ``judge`` are the two
phases underneath.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cleanslate.cli.aliases import resolve_model
from cleanslate.validation.base import BenchmarkResult

console = Console()

TASK_PRESETS: dict[str, list[str]] = {
    "full": [
        "mmlu", "triviaqa", "cleanslate_qa", "gsm8k", "hellaswag", "winogrande",
        "drop", "arc_challenge", "humaneval_plus", "bbh", "lambada_openai", "coqa",
    ],
}


def _resolve_tasks(tasks_str: str) -> list[str]:
    """Expand preset names (e.g. ``full``) into concrete lm-eval tasks.
    Non-preset names pass through unchanged, so you can mix:
    ``-B full,mmlu_pro``. Preserves CLI order; dedupes."""
    out: list[str] = []
    for raw in tasks_str.split(","):
        t = raw.strip()
        if not t:
            continue
        if t in TASK_PRESETS:
            out.extend(TASK_PRESETS[t])
        else:
            out.append(t)
    return list(dict.fromkeys(out))


def _print_header(model_path: str, task_list: list[str], base_url: str | None = None) -> None:
    console.print(f"[bold]Model:[/bold] {model_path}")
    console.print(f"[bold]Tasks:[/bold] {', '.join(task_list)}")
    if base_url:
        console.print(f"[bold]Remote endpoint:[/bold] {base_url}")

def _render_results(results: dict[str, BenchmarkResult], title: str) -> None:
    table = Table(title=title)
    table.add_column("Benchmark", style="bold")
    table.add_column("Accuracy", justify="right")
    table.add_column("Time (s)", justify="right")
    for name, result in sorted(results.items()):
        table.add_row(name, f"{result.accuracy:.4f}", f"{result.eval_time_seconds:.1f}")
    console.print(table)


def _push_generations(
    gen_dir: Path,
    model_clean: str,
    hf_org: str,
    push_suffix: str | None,
    annotated: dict[str, list[dict[str, object]]] | None = None,
) -> None:
    from cleanslate.validation import push_generations_to_hub

    repo = f"{hf_org}/generations-{model_clean}"
    if push_suffix:
        repo = f"{repo}-{push_suffix}"
    url = push_generations_to_hub(gen_dir, repo, annotated_samples=annotated)
    console.print(f"[green]Pushed generations → {url}[/green]")


def generate_cmd(
    model: str = typer.Option(..., "-m", "--model", help="HF model name or alias"),
    tasks: str = typer.Option("full", "-B", "--benchmarks", help="Comma-separated lm-eval tasks. Presets: 'full' (12-benchmark retention suite). Mixable: `full,mmlu_pro`."),
    num_fewshot: int | None = typer.Option(None, "--num-fewshot"),
    batch_size: str = typer.Option("auto", "--batch-size"),
    gpu_memory_utilization: float = typer.Option(0.90, "--gpu-mem"),
    tp_size: int = typer.Option(1, "--tp-size"),
    dp_size: int = typer.Option(1, "--dp-size", help="Data-parallel replicas (N copies of the model, each TP-sharded)"),
    skip_done: bool = typer.Option(False, "--skip-done", help="Skip tasks that already have generations (and cached judge scores, for judge/validate) in the output dir"),
    max_model_len: int = typer.Option(4096, "--max-model-len"),
    base_url: str | None = typer.Option(None, "--base-url", help="Remote vLLM OpenAI endpoint (e.g. https://xxx.modal.run). When set, skips local vLLM."),
    num_concurrent: int = typer.Option(32, "--num-concurrent", help="In-flight requests vs remote endpoint"),
    output: str = typer.Option(..., "-o", "--output", help="Output dir; generations land in <output>/generations/"),
    push_to_hub: bool = typer.Option(False, "--push-to-hub", help="Upload samples as a DatasetDict (one split per benchmark)"),
    hf_org: str = typer.Option("unlearning-cleanslate", "--hf-org"),
    push_suffix: str | None = typer.Option(None, "--push-suffix", help="Suffix on the dataset repo name: generations-<clean>-<suffix>"),
) -> None:
    """Phase A: run generation only, persist samples JSONL. No metrics."""
    from cleanslate.validation import run_generate

    model_path, model_clean = resolve_model(model)
    task_list = _resolve_tasks(tasks)
    _print_header(model_path, task_list, base_url=base_url)

    gen_dir = Path(output) / "generations"
    run_generate(
        model_name=model_path,
        tasks=task_list,
        output_dir=gen_dir,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        tp_size=tp_size,
        dp_size=dp_size,
        max_model_len=max_model_len,
        skip_done=skip_done,
        base_url=base_url,
        num_concurrent=num_concurrent,
    )
    console.print(f"\n[green]Generations saved under {gen_dir}[/green]")

    if push_to_hub:
        _push_generations(gen_dir, model_clean, hf_org, push_suffix)


def judge_cmd(
    input_dir: str = typer.Option(..., "-i", "--input", help="Directory containing `generations/samples_*.jsonl`"),
    output: str | None = typer.Option(None, "-o", "--output"),
    model_name: str | None = typer.Option(None, "--model-name", help="Override model name recorded in output"),
    push_to_hub: bool = typer.Option(False, "--push-to-hub", help="Upload scored samples as a DatasetDict"),
    hf_org: str = typer.Option("unlearning-cleanslate", "--hf-org"),
    push_suffix: str | None = typer.Option(None, "--push-suffix"),
    skip_done: bool = typer.Option(False, "--skip-done", help="Reuse per-task cached scores under <dir>/scored/ instead of re-scoring"),
) -> None:
    """Phase B: score samples from `generate`. Fails loudly on unrecoverable
    judge errors — rerun this command after fixing the environment."""
    from cleanslate.validation import judge_generations, save_validation

    gen_dir = Path(input_dir) / "generations"
    if not gen_dir.exists():
        gen_dir = Path(input_dir)

    derived_model = model_name or "unknown"

    console.print(f"[bold]Samples dir:[/bold] {gen_dir}")
    results, annotated = judge_generations(gen_dir, derived_model, skip_done=skip_done)

    _render_results(results, "Judge Results")

    out_path_str = output or str(Path(input_dir) / "validation.json")
    out_path = save_validation(results, out_path_str, derived_model)
    console.print(f"\n[green]Saved to {out_path}[/green]")

    if push_to_hub:
        _, model_clean = resolve_model(derived_model)
        _push_generations(gen_dir, model_clean, hf_org, push_suffix, annotated=annotated)


def validate_cmd(
    model: str = typer.Option(..., "-m", "--model", help="HF model name or alias"),
    tasks: str = typer.Option("full", "-B", "--benchmarks", help="Comma-separated lm-eval tasks. Presets: 'full'. Mixable: `full,mmlu_pro`."),
    num_fewshot: int | None = typer.Option(None, "--num-fewshot"),
    batch_size: str = typer.Option("auto", "--batch-size"),
    gpu_memory_utilization: float = typer.Option(0.90, "--gpu-mem"),
    tp_size: int = typer.Option(1, "--tp-size"),
    dp_size: int = typer.Option(1, "--dp-size", help="Data-parallel replicas (N copies of the model, each TP-sharded)"),
    skip_done: bool = typer.Option(False, "--skip-done", help="Skip tasks that already have generations (and cached judge scores, for judge/validate) in the output dir"),
    max_model_len: int = typer.Option(4096, "--max-model-len"),
    base_url: str | None = typer.Option(None, "--base-url", help="Remote vLLM OpenAI endpoint (e.g. https://xxx.modal.run). When set, skips local vLLM."),
    num_concurrent: int = typer.Option(32, "--num-concurrent", help="In-flight requests vs remote endpoint"),
    output: str | None = typer.Option(None, "-o", "--output", help="Output dir (not a .json path); final JSON lands at <dir>/<model_clean>.json"),
    push_to_hub: bool = typer.Option(False, "--push-to-hub", help="Upload samples as a DatasetDict after generate"),
    hf_org: str = typer.Option("unlearning-cleanslate", "--hf-org"),
    push_suffix: str | None = typer.Option(None, "--push-suffix"),
) -> None:
    """End-to-end: generate + judge, same JSON output as before. Generations
    are persisted to disk between the two phases, so a judge failure no longer
    loses the vLLM work."""
    from cleanslate.validation import judge_generations, run_generate, save_validation

    model_path, model_clean = resolve_model(model)
    task_list = _resolve_tasks(tasks)
    _print_header(model_path, task_list, base_url=base_url)
    console.print()

    if output and output.endswith(".json"):
        raise typer.BadParameter(
            "-o must be a directory, not a .json file. "
            "Final JSON is written to <dir>/<model_clean>.json.",
            param_hint="-o/--output",
        )
    out_base = Path(output) if output else Path("results/validation") / model_clean
    final_json = out_base / f"{model_clean}.json"
    gen_dir = out_base / "generations"

    run_generate(
        model_name=model_path,
        tasks=task_list,
        output_dir=gen_dir,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        tp_size=tp_size,
        dp_size=dp_size,
        max_model_len=max_model_len,
        skip_done=skip_done,
        base_url=base_url,
        num_concurrent=num_concurrent,
    )
    console.print(f"[dim]Generations saved under {gen_dir}[/dim]\n")

    results, annotated = judge_generations(gen_dir, model_path, skip_done=skip_done)

    _render_results(results, "Validation Results")

    out_path = save_validation(results, final_json, model_path)
    console.print(f"\n[green]Saved to {out_path}[/green]")

    # Push last so a push failure doesn't lose the results JSON.
    if push_to_hub:
        _push_generations(gen_dir, model_clean, hf_org, push_suffix, annotated=annotated)
