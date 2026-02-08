"""Command-line interface for cleanslate."""

from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.table import Table
except ImportError:
    raise ImportError(
        "CLI dependencies not installed. Install with: pip install cleanslate[cli]"
    )

from cleanslate.experiment import Experiment
from cleanslate.generation import GenerationExperiment
from cleanslate.jailbreak import JailbreakExperiment
from cleanslate.results import ExperimentResults
from cleanslate.samples.shakespeare import SHAKESPEARE_SAMPLES
from cleanslate.samples.books import BOOK_SAMPLES
from cleanslate.samples.billboard import load_billboard_songs
from cleanslate.prompts import ALL_TEMPLATES
from cleanslate.evaluation import evaluate_jailbreak_results, summarize_evaluations, print_evaluation_summary

app = typer.Typer(
    name="cleanslate",
    help="Sequence likelihood estimation for LLMs - measure memorization and token probabilities.",
)
console = Console()


@app.command()
def run(
    model: str = typer.Option("Qwen/Qwen3-0.6B", "--model", "-m", help="Model name"),
    dataset: str = typer.Option(
        "shakespeare", "--dataset", "-d", help="Dataset: shakespeare, books, or all"
    ),
    name: str = typer.Option("experiment", "--name", "-n", help="Experiment name"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output JSON file path"
    ),
    device: Optional[str] = typer.Option(
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
    device: Optional[str] = typer.Option(
        None, "--device", help="Device: cpu, cuda, mps"
    ),
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
    top_p: Optional[float] = typer.Option(
        None, "--top-p", help="Nucleus sampling (top-p)"
    ),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Top-k sampling"),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size for generation"),
    hf_repo: Optional[str] = typer.Option(
        None,
        "--hf-repo",
        help="Push to HuggingFace Hub repo (e.g. unlearning-cleanslate/generations)",
    ),
    device: Optional[str] = typer.Option(
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
def jailbreak(
    model: str = typer.Option("Qwen/Qwen3-0.6B", "--model", "-m", help="Model name"),
    output: str = typer.Option(
        "outputs/jailbreak", "--output", "-o", help="Output base path"
    ),
    year_start: int = typer.Option(
        2021, "--year-start", help="Start year for songs (inclusive)"
    ),
    year_end: int = typer.Option(
        2022, "--year-end", help="End year for songs (inclusive)"
    ),
    limit: int = typer.Option(
        50, "--limit", "-n", help="Maximum number of songs to test"
    ),
    templates: Optional[str] = typer.Option(
        None, "--templates", "-t", 
        help="Comma-separated template names (default: direct,helpful_assistant,music_expert,educational)"
    ),
    seed_modes: Optional[str] = typer.Option(
        None, "--seed-modes", "-s",
        help="Comma-separated seed modes: none,first_line,first_verse,half (default: none,first_line)"
    ),
    num_generations: int = typer.Option(
        1, "--num-generations", help="Generations per (song, template, seed_mode)"
    ),
    max_tokens: int = typer.Option(
        8192, "--max-tokens", help="Maximum tokens to generate"
    ),
    temperature: float = typer.Option(
        0.6, "--temperature", help="Sampling temperature"
    ),
    batch_size: int = typer.Option(
        32, "--batch-size", "-b", help="Batch size for generation"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device: cpu, cuda, mps, or auto"
    ),
    shuffle: bool = typer.Option(
        True, "--shuffle/--no-shuffle", help="Shuffle songs before selecting"
    ),
):
    """Run jailbreak experiment on Billboard songs.
    
    Tests different prompt templates and seed modes to measure
    how much lyrics models will reproduce.
    
    Examples:
        cleanslate jailbreak -m Qwen/Qwen3-0.6B -n 20
        cleanslate jailbreak -m meta-llama/Llama-3-8B --year-start 2020 -t direct,music_expert
        cleanslate jailbreak -m google/gemma-2-9b -s none,first_line,first_verse
    """
    
    # Load songs
    console.print(f"[bold]Loading Billboard songs ({year_start}-{year_end})...[/bold]")
    try:
        songs = load_billboard_songs(
            year_range=(year_start, year_end),
            limit=limit,
            shuffle=shuffle,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    console.print(f"Loaded {len(songs)} songs")
    
    # Parse templates
    if templates:
        template_list = [t.strip() for t in templates.split(",")]
        for t in template_list:
            if t not in ALL_TEMPLATES:
                console.print(f"[red]Unknown template: {t}[/red]")
                console.print(f"Available: {list(ALL_TEMPLATES.keys())}")
                raise typer.Exit(1)
    else:
        template_list = ["direct", "helpful_assistant", "music_expert", "educational"]
    
    # Parse seed modes
    if seed_modes:
        seed_mode_list = [s.strip() for s in seed_modes.split(",")]
        valid_modes = {"none", "first_line", "first_verse", "half"}
        for s in seed_mode_list:
            if s not in valid_modes:
                console.print(f"[red]Unknown seed mode: {s}[/red]")
                console.print(f"Available: {valid_modes}")
                raise typer.Exit(1)
    else:
        seed_mode_list = ["none", "first_line"]
    
    total_gens = len(songs) * len(template_list) * len(seed_mode_list) * num_generations
    
    console.print()
    console.print("[bold]Jailbreak Experiment Configuration[/bold]")
    console.print(f"  Model: {model}")
    console.print(f"  Songs: {len(songs)} ({year_start}-{year_end})")
    console.print(f"  Templates: {template_list}")
    console.print(f"  Seed modes: {seed_mode_list}")
    console.print(f"  Generations per config: {num_generations}")
    console.print(f"  Total generations: {total_gens}")
    console.print(f"  Output: {output}")
    console.print()
    
    # Run experiment
    exp = JailbreakExperiment(
        model=model,
        device=device,
        experiment_name=f"jailbreak_{year_start}_{year_end}",
    )
    
    rows = exp.run(
        songs=songs,
        templates=template_list,
        seed_modes=seed_mode_list,
        output_path=output,
        num_generations=num_generations,
        max_new_tokens=max_tokens,
        temperature=temperature,
        batch_size=batch_size,
        top_p=0.95,
        top_k=20,
        do_sample=True, 
    )
    
    console.print()
    console.print(f"[bold green]Completed {len(rows)} generations[/bold green]")
    console.print(f"Results saved to: {output}.parquet and {output}.json")
    
    # Show sample
    if rows:
        console.print()
        console.print("[bold]Sample output:[/bold]")
        sample = rows[0]
        console.print(f"[dim]Song:[/dim] {sample.song_name}")
        console.print(f"[dim]Template:[/dim] {sample.prompt_template}")
        console.print(f"[dim]Seed mode:[/dim] {sample.seed_mode}")
        console.print(f"[dim]Generated:[/dim] {sample.generated_text[:300]}...")


@app.command()
def evaluate(
    results_path: str = typer.Argument(..., help="Path to jailbreak results (parquet or json)"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output path for evaluation results"
    ),
    group_by: str = typer.Option(
        "both", "--group-by", "-g", 
        help="Group results by: template, seed_mode, or both"
    ),
):
    """Evaluate jailbreak experiment results.
    
    Computes metrics comparing generated text to ground truth lyrics:
    - BLEU-1, BLEU-2: N-gram precision
    - ROUGE-1, ROUGE-L: N-gram recall/F1
    - Exact match, line match rates
    
    Examples:
        cleanslate evaluate outputs/jailbreak.parquet
        cleanslate evaluate results.json -o eval_results.json --group-by template
    """
    from pathlib import Path
    
    path = Path(results_path)
    if not path.exists():
        console.print(f"[red]File not found: {results_path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold]Evaluating results from {results_path}...[/bold]")
    
    # Run evaluation
    eval_results = evaluate_jailbreak_results(results_path, output)
    
    console.print(f"Evaluated {len(eval_results)} generations")
    
    # Summarize and print
    if group_by not in ("template", "seed_mode", "both"):
        console.print(f"[red]Invalid group_by: {group_by}. Use: template, seed_mode, or both[/red]")
        raise typer.Exit(1)
    
    summary = summarize_evaluations(eval_results, group_by=group_by)
    print_evaluation_summary(summary)


@app.command()
def list_templates():
    """List available prompt templates for jailbreak experiments."""
    
    console.print("[bold]Available Prompt Templates:[/bold]")
    console.print()
    
    for name, template in ALL_TEMPLATES.items():
        console.print(f"[cyan]{name}[/cyan]")
        console.print(f"  {template.description}")
        if template.system_prompt:
            # Truncate long system prompts
            sp = template.system_prompt[:100] + "..." if len(template.system_prompt) > 100 else template.system_prompt
            console.print(f"  [dim]System: {sp}[/dim]")
        console.print()


@app.command()
def list_billboard(
    year_start: int = typer.Option(1970, "--year-start", help="Start year"),
    year_end: int = typer.Option(2025, "--year-end", help="End year"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of songs to show"),
):
    """List Billboard songs from the dataset."""
    
    try:
        songs = load_billboard_songs(
            year_range=(year_start, year_end),
            limit=limit,
            shuffle=False,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold]Billboard Songs ({year_start}-{year_end}):[/bold]")
    console.print()
    
    for i, song in enumerate(songs, 1):
        artists = ", ".join(song["artists"]) if song["artists"] else "Unknown"
        lyrics_preview = song["lyrics"][:50] + "..." if len(song["lyrics"]) > 50 else song["lyrics"]
        lyrics_preview = lyrics_preview.replace("\n", " ")
        
        console.print(f"[cyan]{i}. {song['title']}[/cyan] - {artists} ({song['year']})")
        console.print(f"   [dim]{lyrics_preview}[/dim]")
        console.print()


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
