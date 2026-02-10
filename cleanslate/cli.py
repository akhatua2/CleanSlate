try:
    import typer
    from rich.console import Console
    from rich.table import Table
except ImportError:
    raise ImportError(
        "CLI dependencies not installed. Install with: pip install cleanslate[cli]"
    )

from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

from cleanslate.experiment import Experiment
from cleanslate.evaluation import EvaluationExperiment
from cleanslate.generation import GenerationExperiment
from cleanslate.prompt_dataset import build_prompt_splits
from cleanslate.prompts import ALL_SEED_MODES, ALL_TEMPLATES
from cleanslate.results import ExperimentResults
from cleanslate.samples.billboard import (
    DATASET_ID,
    DEFAULT_DATASET_SPLIT,
    load_billboard_dataset,
)
from cleanslate.samples.books import BOOK_SAMPLES
from cleanslate.samples.shakespeare import SHAKESPEARE_SAMPLES
from cleanslate.samples.songs import SONG_SAMPLES

app = typer.Typer(
    name="cleanslate",
    help="Sequence likelihood estimation for LLMs - measure memorization and token probabilities.",
)
console = Console()


def _parse_templates(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]
    template_list: list[str] = []
    for item in raw_items:
        template_list.extend([t.strip() for t in item.split(",") if t.strip()])
    for t in template_list:
        if t not in ALL_TEMPLATES:
            raise typer.BadParameter(
                f"Unknown template: {t}. Available: {list(ALL_TEMPLATES.keys())}"
            )
    return template_list


def _parse_seed_modes(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]
    seed_mode_list: list[str] = []
    for item in raw_items:
        seed_mode_list.extend([s.strip() for s in item.split(",") if s.strip()])
    for s in seed_mode_list:
        if s not in ALL_SEED_MODES:
            raise typer.BadParameter(
                f"Unknown seed mode: {s}. Available: {ALL_SEED_MODES}"
            )
    return seed_mode_list


@app.command()
def run(
    model: str = typer.Option("Qwen/Qwen3-0.6B", "--model", "-m", help="Model name"),
    dataset: str = typer.Option(
        "shakespeare", "--dataset", "-d", help="Dataset: shakespeare, books, or all"
    ),
    name: str = typer.Option("experiment", "--name", "-n", help="Experiment name"),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output JSON file path"
    ),
    device: str | None = typer.Option(
        None, "--device", help="Device: cpu, cuda, mps, or auto"
    ),
):
    """Run sequence likelihood experiment on a dataset."""

    # Select samples
    if dataset == "shakespeare":
        samples = SHAKESPEARE_SAMPLES
    elif dataset == "books":
        samples = BOOK_SAMPLES
    elif dataset == "all":
        samples = SHAKESPEARE_SAMPLES + BOOK_SAMPLES
    else:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        console.print("Available: shakespeare, books, all")
        raise typer.Exit(1)

    console.print(f"[bold]Running experiment:[/bold] {name}")
    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Dataset:[/bold] {dataset} ({len(samples)} samples)")
    console.print()

    # Run experiment
    exp = Experiment(model=model, name=name, device=device)
    results = exp.run(samples)

    # Display results
    console.print()
    console.print(results.summary())

    # Save if output specified
    if output:
        results.save(output)


@app.command()
def probe(
    seed: str = typer.Argument(..., help="Seed/prompt text"),
    target: str = typer.Argument(..., help="Target continuation text"),
    model: str = typer.Option("Qwen/Qwen3-0.6B", "--model", "-m", help="Model name"),
    device: str | None = typer.Option(None, "--device", help="Device: cpu, cuda, mps"),
):
    """Probe a single seed/target pair."""

    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Seed:[/bold] {seed[:50]}...")
    console.print(f"[bold]Target:[/bold] {target[:50]}...")
    console.print()

    exp = Experiment(model=model, name="probe", device=device)
    result = exp.run_single(seed, target)

    # Display result
    table = Table(title="Probe Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Log-Likelihood", f"{result.log_likelihood:.2f}")
    table.add_row("Perplexity", f"{result.perplexity:.2f}")
    table.add_row("Tokens", str(result.num_tokens))
    table.add_row("Rank-1 Fraction", f"{result.rank_1_fraction:.1%}")
    table.add_row("Avg Rank", f"{result.avg_rank:.1f}")

    console.print(table)


@app.command()
def compare(
    file1: str = typer.Argument(..., help="First results JSON file"),
    file2: str = typer.Argument(..., help="Second results JSON file"),
):
    """Compare two experiment results."""

    results1 = ExperimentResults.load(file1)
    results2 = ExperimentResults.load(file2)

    console.print(results1.compare(results2))


@app.command()
def show(
    file: str = typer.Argument(..., help="Results JSON file to display"),
    tokens: bool = typer.Option(False, "--tokens", "-t", help="Show token counts"),
):
    """Display saved experiment results."""

    results = ExperimentResults.load(file)
    console.print(results.summary(show_tokens=tokens))


@app.command()
def list_samples():
    """List available sample datasets."""

    console.print("[bold]Shakespeare Samples:[/bold]")
    for i, s in enumerate(SHAKESPEARE_SAMPLES, 1):
        console.print(f"  {i}. {s['name']}")

    console.print()
    console.print("[bold]Book Samples:[/bold]")
    for i, s in enumerate(BOOK_SAMPLES, 1):
        console.print(f"  {i}. {s['name']}")


@app.command()
def generate(
    model: str = typer.Option(
        "Qwen/Qwen2.5-0.5B-Instruct", "--model", "-m", help="Model name"
    ),
    dataset: str = typer.Option(
        "all",
        "--dataset",
        "-d",
        help="Dataset: shakespeare, books, songs, or all",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file path (.parquet)",
    ),
    backend: str = typer.Option(
        "huggingface",
        "--backend",
        "-b",
        help="Backend: huggingface or sglang",
    ),
    num_generations: int = typer.Option(
        3, "--num-generations", "-n", help="Number of generations per seed"
    ),
    max_tokens: int = typer.Option(
        128, "--max-tokens", help="Maximum tokens per generation"
    ),
    temperature: float = typer.Option(
        1.0, "--temperature", "-t", help="Sampling temperature"
    ),
    top_p: float | None = typer.Option(
        None, "--top-p", help="Nucleus sampling (top-p)"
    ),
    top_k: int | None = typer.Option(None, "--top-k", help="Top-k sampling"),
    min_p: float = typer.Option(
        0, "--min-p", help="Min-p sampling (keep tokens with prob >= min_p * max_prob)"
    ),
    repetition_penalty: float | None = typer.Option(
        None,
        "--repetition-penalty",
        help="Penalize repeated tokens",
    ),
    capture_top_k: int = typer.Option(
        50,
        "--capture-top-k",
        help="Top-k alternatives to capture per position (0 = logprobs only, faster)",
    ),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size for generation"),
    hf_repo: str | None = typer.Option(
        None,
        "--hf-repo",
        help="Push to HuggingFace Hub repo (e.g. unlearning-cleanslate/generations)",
    ),
    device: str | None = typer.Option(
        None, "--device", help="Device: cpu, cuda, mps, or auto"
    ),
):
    """
    Generate multiple outputs per seed and save all data (no metrics computed).

    Saves: tokens, logprobs, top-k alternatives, full generated text, etc.
    Use --backend sglang for GPU-optimized batch generation.
    """

    # Select samples
    if dataset == "shakespeare":
        samples = SHAKESPEARE_SAMPLES
    elif dataset == "books":
        samples = BOOK_SAMPLES
    elif dataset == "songs":
        samples = SONG_SAMPLES
    elif dataset == "all":
        samples = SHAKESPEARE_SAMPLES + BOOK_SAMPLES + SONG_SAMPLES
    else:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        console.print("Available: shakespeare, books, songs, all")
        raise typer.Exit(1)

    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Backend:[/bold] {backend}")
    console.print(f"[bold]Dataset:[/bold] {dataset} ({len(samples)} samples)")
    console.print(
        f"[bold]Generations:[/bold] {num_generations} x {len(samples)} = "
        f"{num_generations * len(samples)} total"
    )
    console.print(f"[bold]Batch size:[/bold] {batch_size}")
    console.print(f"[bold]Capture top-k:[/bold] {capture_top_k}")
    console.print()

    exp = GenerationExperiment(model=model, backend=backend, device=device)
    rows = exp.run(
        samples=samples,
        dataset_name=dataset,
        output_path=output,
        num_generations=num_generations,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        capture_top_k=capture_top_k,
        batch_size=batch_size,
        hf_repo=hf_repo,
    )

    console.print()
    console.print(f"[green]Saved {len(rows)} generations to {output}[/green]")
    if hf_repo:
        console.print(
            f"[green]Pushed to https://huggingface.co/datasets/{hf_repo}[/green]"
        )


@app.command()
def build_prompts(
    dataset_id: str = typer.Option(
        DATASET_ID, "--dataset-id", help="HuggingFace dataset ID"
    ),
    dataset_split: str = typer.Option(
        DEFAULT_DATASET_SPLIT, "--dataset-split", help="Dataset split name"
    ),
    year_start: int | None = typer.Option(
        None, "--year-start", help="Start year (optional)"
    ),
    year_end: int | None = typer.Option(None, "--year-end", help="End year (optional)"),
    limit: int | None = typer.Option(
        None, "--limit", "-n", help="Maximum number of songs"
    ),
    shuffle: bool = typer.Option(
        False, "--shuffle/--no-shuffle", help="Shuffle before selecting"
    ),
    seed: int = typer.Option(42, "--seed", help="Shuffle seed"),
    templates: list[str] | None = typer.Option(
        None,
        "--templates",
        "-t",
        help="Comma-separated template names (default: all)",
        callback=_parse_templates,
    ),
    seed_modes: list[str] | None = typer.Option(
        None,
        "--seed-modes",
        "-s",
        help="Comma-separated seed modes (default: none,first_line)",
        callback=_parse_seed_modes,
    ),
    require_lyrics: bool = typer.Option(
        True, "--require-lyrics/--no-require-lyrics", help="Drop rows without lyrics"
    ),
    output_repo: str | None = typer.Option(
        None,
        "--output-repo",
        help="HuggingFace dataset repo to push (e.g. org/prompts)",
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", help="Local directory to save DatasetDict"
    ),
):
    """Build prompt splits from the Billboard dataset."""
    console.print(f"[bold]Loading dataset:[/bold] {dataset_id} ({dataset_split})")

    if (year_start is None) ^ (year_end is None):
        raise typer.BadParameter(
            "Provide both --year-start and --year-end, or neither."
        )
    year_range = (year_start, year_end) if year_start is not None else None

    ds = load_billboard_dataset(
        dataset_id=dataset_id,
        dataset_split=dataset_split,
        year_range=year_range,
        limit=limit,
        require_lyrics=require_lyrics,
        shuffle=shuffle,
        seed=seed,
    )

    template_list = templates or list(ALL_TEMPLATES.keys())
    seed_mode_list = seed_modes or ALL_SEED_MODES

    console.print(
        f"[bold]Building splits:[/bold] {len(template_list)} templates x "
        f"{len(seed_mode_list)} seed modes"
    )

    ds_dict = build_prompt_splits(ds, template_list, seed_mode_list)

    for split_name, split_ds in ds_dict.items():
        console.print(f"  Split '{split_name}': {len(split_ds)} rows")

    if output_dir:
        ds_dict.save_to_disk(output_dir)
        console.print(f"[green]Saved to {output_dir}[/green]")

    if output_repo:
        ds_dict.push_to_hub(output_repo)
        console.print(
            f"[green]Pushed to https://huggingface.co/datasets/{output_repo}[/green]"
        )

    if not output_dir and not output_repo:
        console.print(
            "[yellow]No output specified. Use --output-dir and/or --output-repo to save.[/yellow]"
        )


@app.command()
def generate_prompts(
    prompts_dataset_id: str = typer.Option(
        ..., "--prompts-dataset-id", help="HuggingFace prompt dataset ID or local path"
    ),
    split: str = typer.Option(
        "direct__none",
        "--split",
        help="Prompt split (<template>__<seed_mode>), or 'all' to run every split",
    ),
    seed_modes: list[str] | None = typer.Option(
        None,
        "--seed-modes",
        "-s",
        help="Comma-separated seed modes to include",
        callback=_parse_seed_modes,
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-n", help="Maximum number of rows"
    ),
    shuffle: bool = typer.Option(
        False, "--shuffle/--no-shuffle", help="Shuffle before selecting"
    ),
    seed: int = typer.Option(42, "--seed", help="Shuffle seed"),
    model: str = typer.Option("Qwen/Qwen3-0.6B", "--model", "-m", help="Model name"),
    backend: str = typer.Option(
        "sglang", "--backend", "-b", help="Backend: sglang or huggingface"
    ),
    thinking: bool = typer.Option(
        False, "--thinking/--no-thinking", help="Enable model thinking mode"
    ),
    num_generations: int = typer.Option(
        1, "--num-generations", help="Generations per prompt"
    ),
    max_tokens: int = typer.Option(
        512, "--max-tokens", help="Maximum tokens per generation"
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", help="Sampling temperature"
    ),
    top_p: float | None = typer.Option(
        0.80, "--top-p", help="Nucleus sampling (top-p)"
    ),
    top_k: int | None = typer.Option(20, "--top-k", help="Top-k sampling"),
    min_p: float = typer.Option(
        0,
        "--min-p",
        help="Min-p sampling (keep tokens with prob >= min_p * max_prob)",
    ),
    repetition_penalty: float | None = typer.Option(
        None,
        "--repetition-penalty",
        help="Penalize repeated tokens (1.0 = no penalty, >1.0 = less repetition)",
    ),
    json_decoding: bool = typer.Option(
        False,
        "--json-decoding/--no-json-decoding",
        help="Enable JSON schema for decoding",
    ),
    capture_top_k: int = typer.Option(
        0,
        "--capture-top-k",
        help="Top-k alternatives to capture per position (0 = logprobs only, faster)",
    ),
    batch_size: int = typer.Option(128, "--batch-size", help="Batch size"),
    output: str = typer.Option(
        "outputs/prompts_generations",
        "--output",
        "-o",
        help="Output directory. Each split is saved as <output>/<split>.parquet",
    ),
    output_repo: str | None = typer.Option(
        None,
        "--output-repo",
        help="HuggingFace dataset repo to push outputs",
    ),
    output_config: str | None = typer.Option(
        None,
        "--output-config",
        help="Dataset config name for HF push (default: sanitized model name)",
    ),
):
    """Generate from a prompt split dataset and optionally push outputs to HF.

    Use --split all to iterate over every split in the dataset.
    Outputs are saved as <output>/<split>.parquet.
    """
    # ── Load dataset ──────────────────────────────────────────────────
    is_local = Path(prompts_dataset_id).exists()
    if is_local:
        ds_dict = load_from_disk(prompts_dataset_id)
    else:
        ds_dict = load_dataset(prompts_dataset_id)

    if not isinstance(ds_dict, DatasetDict):
        console.print(
            "[red]Expected a DatasetDict (multiple splits). "
            "Got a single Dataset instead.[/red]"
        )
        raise typer.Exit(1)

    # ── Resolve which splits to run ───────────────────────────────────
    all_split_names = list(ds_dict.keys())

    if split.lower() == "all":
        splits_to_run = all_split_names
    else:
        if split not in ds_dict:
            console.print(f"[red]Split not found: {split}[/red]")
            console.print(f"Available: {all_split_names}")
            raise typer.Exit(1)
        splits_to_run = [split]

    if not output_config:
        output_config = model.replace("/", "__")

    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Backend:[/bold] {backend}")
    console.print(
        f"[bold]Splits:[/bold] {len(splits_to_run)} "
        f"({', '.join(splits_to_run)})"
    )
    console.print()

    # ── Create experiment once (model loads once) ─────────────────────
    exp = GenerationExperiment(
        model=model, backend=backend, json_decoding=json_decoding
    )

    output_dir = Path(output)

    for split_name in splits_to_run:
        ds = ds_dict[split_name]

        if seed_modes:
            ds = ds.filter(lambda x: x.get("seed_mode") in seed_modes)

        if shuffle:
            ds = ds.shuffle(seed=seed)

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        rows = list(ds)

        console.print(
            f"[bold]Split '{split_name}':[/bold] {len(rows)} prompts"
        )

        split_output = output_dir / split_name

        exp.run(
            samples=rows,
            dataset_name=output_config,
            output_path=str(split_output),
            num_generations=num_generations,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            capture_top_k=capture_top_k,
            batch_size=batch_size,
            hf_repo=output_repo,
            thinking=thinking,
        )

        console.print(
            f"[green]  ✓ Saved {len(rows)} generations → "
            f"{split_output}.parquet[/green]"
        )

    console.print()
    console.print(
        f"[green bold]Done — {len(splits_to_run)} split(s) saved "
        f"under {output_dir}/[/green bold]"
    )
    if output_repo:
        console.print(
            f"[green]Pushed to https://huggingface.co/datasets/{output_repo}[/green]"
        )


@app.command()
def eval(
    dataset: str = typer.Option(
        "unlearning-cleanslate/generated_songs",
        "--dataset",
        "-d",
        help="Dataset: HuggingFace ID (e.g., unlearning-cleanslate/generated_songs) or local path",
    ),
    split: str | None = typer.Option(
        None, "--split", "-s", help="Dataset split name (e.g. direct__none)"
    ),
    all_splits: bool = typer.Option(
        False, "--all-splits", help="Evaluate all splits in the dataset"
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output file path (.json or .parquet)"
    ),
    output_repo: str | None = typer.Option(
        None,
        "--output-repo",
        help="HuggingFace dataset repo to push eval results (e.g., org/dataset_eval)"
    ),
    similarity_metrics: str | None = typer.Option(
        None,
        "--similarity-metrics",
        help="Comma-separated BLEU/ROUGE names (default: all)",
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-n", help="Maximum rows to evaluate per split"
    ),
):
    """Evaluate generated outputs against reference targets.

    Computes metrics from pre-existing data - NO MODEL LOADING REQUIRED.
    Probability metrics are computed from token_log_probs already in the dataset.

    Supports single split or all splits in a dataset.

    INPUT DATASET:
      Source: HuggingFace dataset ID (e.g., unlearning-cleanslate/generated_songs)
              or local dataset path (e.g., /path/to/dataset)

      Fields required: generated_text, reference_target, token_log_probs
      Fields preserved: sample_name, prompt_template, seed_mode

    OUTPUT STRUCTURE (one row per sample, all metrics as columns):

    SIMILARITY METRICS:
      - bleu1, bleu2: N-gram overlap (unigram and bigram)
      - rouge1, rouge2, rougel: Word-level recall, bigram recall, longest common subsequence

    PROBABILITY METRICS (from pre-computed token_log_probs):
      - log_prob: Sum of log probabilities for generated tokens
      - avg_log_prob: Average log probability per token
      - perplexity: Model's surprise at generated text (lower = more memorized)
      - extraction_prob: Probability of generating this exact sequence

    METADATA:
      - sample_name, prompt_template, seed_mode: From input dataset
      - num_tokens: Number of tokens in generated text
      - generated_text, reference_target: Texts (truncated)
    """
    from datasets import load_dataset, load_from_disk, Dataset
    from dataclasses import asdict
    from pathlib import Path

    # Validate arguments
    if all_splits and split:
        raise typer.BadParameter("Cannot use both --all-splits and --split")
    if not all_splits and not split:
        raise typer.BadParameter("Provide either --split or --all-splits")

    exp = EvaluationExperiment()

    # Parse similarity metrics
    similarity_metric_list = None
    if similarity_metrics:
        similarity_metric_list = [m.strip() for m in similarity_metrics.split(",")]
    else:
        similarity_metric_list = exp.available_similarity_metrics

    console.print(f"[bold]Dataset:[/bold] {dataset}")
    console.print(
        f"[bold]Similarity Metrics:[/bold] {', '.join(similarity_metric_list)}"
    )
    console.print()

    # Set cache directory to avoid home directory quota issues
    import os
    default_cache = Path(__file__).parent.parent / ".cache" / "huggingface" / "datasets"
    cache_dir = os.environ.get("HF_DATASETS_CACHE", str(default_cache))
    os.makedirs(cache_dir, exist_ok=True)

    # Load dataset (handle both local paths and HF Hub IDs)
    def get_dataset_splits(dataset_path: str) -> list[str]:
        """Get available configs/splits from HF Hub dataset or local path."""
        import ast

        if Path(dataset_path).exists():
            ds_dict = load_from_disk(dataset_path)
            return list(ds_dict.keys())
        else:
            # For HF Hub datasets, try to get configs from builder
            try:
                from datasets import load_dataset_builder
                builder = load_dataset_builder(dataset_path)
                return [config.name for config in builder.builder_configs]
            except ValueError as e:
                # Extract config names from error message
                # HF datasets always lists available configs in the error
                error_str = str(e)
                if "available configs:" in error_str:
                    try:
                        # Find the Python list in the error message
                        start = error_str.find("[")
                        end = error_str.rfind("]") + 1
                        configs_str = error_str[start:end]
                        configs = ast.literal_eval(configs_str)
                        return configs
                    except Exception:
                        pass
                raise

    def load_dataset_split(dataset_path: str, split_name: str):
        """Load a specific split/config from HF Hub or local path.

        Uses streaming to avoid disk quota issues, then converts to list.
        """
        if Path(dataset_path).exists():
            ds_dict = load_from_disk(dataset_path)
            return ds_dict[split_name]
        else:
            # For HF Hub Parquet datasets, split_name is actually a config name
            # Use streaming=True to avoid disk quota issues
            ds_stream = load_dataset(
                dataset_path, name=split_name, split="train", streaming=True, cache_dir=cache_dir
            )
            # Convert streaming dataset to list for processing
            return list(ds_stream)

    if all_splits:
        # Discover available configs/splits
        splits = get_dataset_splits(dataset)
        console.print(f"[bold]Configs/Splits:[/bold] {len(splits)} found")

        all_results = []
        for split_name in splits:
            console.print(f"[yellow]Evaluating config: {split_name}[/yellow]")
            results = exp.run(
                dataset_id=dataset,
                split=split_name,
                output_path=None,
                metrics=similarity_metric_list,
                limit=limit,
            )
            all_results.extend(results)
            summary = exp.summary(results)
            console.print(f"  Samples: {len(results)}")
            for metric_name in ["bleu1", "rouge1", "perplexity", "extraction_prob"]:
                if metric_name in summary:
                    stats = summary[metric_name]
                    console.print(f"  {metric_name}: mean={stats['mean']:.4f}")

        console.print()

        # Save combined results
        if output:
            exp._save_results(all_results, output)
            console.print(f"[green]Saved {len(all_results)} results to {output}[/green]")

        if output_repo:
            # Convert to HuggingFace Dataset and push
            data = [asdict(r) for r in all_results]
            eval_dataset = Dataset.from_dict(
                {k: [d[k] for d in data] for k in data[0].keys()}
            )
            eval_dataset.push_to_hub(output_repo)
            console.print(
                f"[green]Pushed to https://huggingface.co/datasets/{output_repo}[/green]"
            )

        # Display overall summary
        overall_summary = exp.summary(all_results)
        table = Table(title="Overall Evaluation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="green")
        table.add_column("Min", style="yellow")
        table.add_column("Max", style="magenta")
        table.add_column("Count", style="blue")

        for metric_name, stats in sorted(overall_summary.items()):
            table.add_row(
                metric_name,
                f"{stats['mean']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}",
                str(stats["count"]),
            )

        console.print(table)

    else:
        # Single split evaluation
        console.print(f"[bold]Split:[/bold] {split}")
        if limit:
            console.print(f"[bold]Limit:[/bold] {limit}")
        console.print()

        results = exp.run(
            dataset_id=dataset,
            split=split,
            output_path=output,
            metrics=similarity_metric_list,
            limit=limit,
        )

        # Display summary
        summary = exp.summary(results)

        table = Table(title="Evaluation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="green")
        table.add_column("Min", style="yellow")
        table.add_column("Max", style="magenta")
        table.add_column("Count", style="blue")

        for metric_name, stats in sorted(summary.items()):
            table.add_row(
                metric_name,
                f"{stats['mean']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}",
                str(stats["count"]),
            )

        console.print(table)

        if output:
            console.print(f"[green]Saved {len(results)} results to {output}[/green]")

        if output_repo:
            # Convert to HuggingFace Dataset and push
            data = [asdict(r) for r in results]
            eval_dataset = Dataset.from_dict(
                {k: [d[k] for d in data] for k in data[0].keys()}
            )
            eval_dataset.push_to_hub(output_repo)
            console.print(
                f"[green]Pushed to https://huggingface.co/datasets/{output_repo}[/green]"
            )


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
