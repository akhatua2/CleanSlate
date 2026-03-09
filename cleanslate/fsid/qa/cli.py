"""CLI commands for QA benchmark generation."""

import typer
from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress

from cleanslate.fsid.qa.generator import generate_qa_candidates
from cleanslate.fsid.qa.io import load_candidates, save_candidates, save_validated, export_for_dataset
from cleanslate.fsid.qa.validator import validate_qa_candidates, BASE_MODELS

qa_app = typer.Typer(
    name="qa",
    help="QA benchmark generation: generate and validate atomic QA pairs.",
    no_args_is_help=True,
)
console = Console()


@qa_app.command("generate")
def generate(
    hf_dataset: str = typer.Option(
        "unlearning-cleanslate/cleanslate_dataset",
        "-d",
        "--dataset",
        help="HuggingFace dataset ID",
    ),
    hf_config: str = typer.Option(
        "default",
        "-c",
        "--config",
        help="Dataset config",
    ),
    num_questions: int = typer.Option(
        10,
        "-n",
        "--num-questions",
        help="Number of QA candidates per song",
    ),
    limit: int | None = typer.Option(
        None,
        "-l",
        "--limit",
        help="Limit number of songs to process",
    ),
    output: str = typer.Option(
        "data/fsid/qa/candidates.parquet",
        "-o",
        "--output",
        help="Output parquet path",
    ),
) -> None:
    """Generate atomic QA candidates from song lyrics using Gemini 3 Flash."""
    console.print(f"[bold]Loading dataset:[/bold] {hf_dataset}")
    ds = load_dataset(hf_dataset, hf_config, split="train")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    console.print(f"[bold]Songs:[/bold] {len(ds)}")
    console.print(f"[bold]Questions per song:[/bold] {num_questions}")

    records = list(ds)

    with Progress() as progress:
        task = progress.add_task("Generating QA...", total=len(records))

        def callback(current: int, total: int) -> None:
            progress.update(task, completed=current)

        candidates = generate_qa_candidates(
            records,
            num_questions=num_questions,
            progress_callback=callback,
        )

    out = save_candidates(candidates, output)

    console.print()
    console.print(f"[bold]Generated:[/bold] {len(candidates)} QA pairs")
    console.print(f"[bold]Saved to:[/bold] {out}")


@qa_app.command("validate")
def validate(
    candidates_path: str = typer.Option(
        "data/fsid/qa/candidates.parquet",
        "-i",
        "--input",
        help="Input candidates parquet",
    ),
    models: str = typer.Option(
        ",".join(BASE_MODELS),
        "-m",
        "--models",
        help="Comma-separated model names",
    ),
    pass_k: int = typer.Option(
        3,
        "-k",
        "--pass-k",
        help="Number of attempts per model (pass@k)",
    ),
    output: str = typer.Option(
        "data/fsid/qa/validated.parquet",
        "-o",
        "--output",
        help="Output parquet path",
    ),
    export_json: str | None = typer.Option(
        None,
        "--export-json",
        help="Also export as JSON for HF dataset",
    ),
) -> None:
    """Validate QA candidates against base models. Keep only universal QA."""
    console.print(f"[bold]Loading candidates:[/bold] {candidates_path}")
    candidates = load_candidates(candidates_path)
    console.print(f"[bold]Candidates:[/bold] {len(candidates)}")

    model_list = [m.strip() for m in models.split(",")]
    console.print(f"[bold]Models:[/bold] {len(model_list)}")
    console.print(f"[bold]Pass@k:[/bold] {pass_k}")

    with Progress() as progress:
        task = progress.add_task("Validating...", total=len(candidates))

        def callback(current: int, total: int) -> None:
            progress.update(task, completed=current)

        validated = validate_qa_candidates(
            candidates,
            models=model_list,
            pass_k=pass_k,
            progress_callback=callback,
        )

    out = save_validated(validated, output)

    console.print()
    console.print(f"[bold]Validated:[/bold] {len(validated)} / {len(candidates)} QA pairs")
    console.print(f"[bold]Pass rate:[/bold] {len(validated) / len(candidates):.1%}")
    console.print(f"[bold]Saved to:[/bold] {out}")

    if export_json:
        json_out = export_for_dataset(validated, export_json)
        console.print(f"[bold]Exported JSON:[/bold] {json_out}")


@qa_app.command("merge")
def merge(
    validated_path: str = typer.Option(
        "data/fsid/qa/validated.parquet",
        "-i",
        "--input",
        help="Validated QA parquet",
    ),
    hf_dataset: str = typer.Option(
        "unlearning-cleanslate/cleanslate_dataset",
        "-d",
        "--dataset",
        help="HuggingFace dataset to update",
    ),
    hf_push: str | None = typer.Option(
        None,
        "--hf-push",
        help="Push merged dataset to this HF repo",
    ),
    output: str = typer.Option(
        "data/fsid/qa/merged_dataset",
        "-o",
        "--output",
        help="Local output directory",
    ),
) -> None:
    """Merge validated QA pairs into cleanslate_dataset (add content_qa column)."""
    from pathlib import Path
    from datasets import load_dataset, Dataset

    console.print(f"[bold]Loading validated QA:[/bold] {validated_path}")
    validated = load_validated(validated_path)

    # Group by content_id
    qa_by_id: dict[str, list[dict[str, str]]] = {}
    for v in validated:
        if v.content_id not in qa_by_id:
            qa_by_id[v.content_id] = []
        qa_by_id[v.content_id].append({"question": v.question, "answer": v.answer})

    console.print(f"[bold]QA pairs:[/bold] {len(validated)}")
    console.print(f"[bold]Content items with QA:[/bold] {len(qa_by_id)}")

    # Load original dataset
    console.print(f"[bold]Loading dataset:[/bold] {hf_dataset}")
    ds = load_dataset(hf_dataset, split="train")

    # Add content_qa column
    def add_qa(row):
        cid = row.get("content_id", row.get("song_id", ""))
        row["content_qa"] = qa_by_id.get(cid, [])
        return row

    ds = ds.map(add_qa)

    # Save locally
    out_path = Path(output)
    out_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_path))
    console.print(f"[bold]Saved to:[/bold] {out_path}")

    # Push to HF
    if hf_push:
        ds.push_to_hub(hf_push)
        console.print(f"[bold]Pushed to:[/bold] https://huggingface.co/datasets/{hf_push}")
