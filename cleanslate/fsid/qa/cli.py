"""CLI commands for QA benchmark generation, evaluation, and scoring."""

import typer
from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress

from cleanslate.fsid.qa.generator import generate_qa_candidates
from cleanslate.fsid.qa.io import load_candidates, save_candidates, save_validated, export_for_dataset
from cleanslate.fsid.qa.validator import validate_qa_candidates, BASE_MODELS

qa_app = typer.Typer(
    name="qa",
    help="QA benchmark: generate, validate, evaluate, and score QA pairs.",
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


@qa_app.command("eval")
def evaluate(
    model: str = typer.Option(
        ...,
        "-m",
        "--model",
        help="Model path or HuggingFace name",
    ),
    hf_dataset: str = typer.Option(
        "unlearning-cleanslate/cleanslate_dataset",
        "-d",
        "--dataset",
        help="HuggingFace dataset ID",
    ),
    hf_config: str = typer.Option(
        "qa_benchmark",
        "-c",
        "--config",
        help="Dataset config name",
    ),
    output: str = typer.Option(
        "results/fsid/qa_eval.json",
        "-o",
        "--output",
        help="Output JSON path",
    ),
    use_judge: bool = typer.Option(
        True,
        "--judge/--no-judge",
        help="Use LLM judge for answer checking (slower but more accurate)",
    ),
    mem_fraction: float = typer.Option(
        0.85,
        "--mem-fraction",
        help="GPU memory fraction for SGLang engine",
    ),
    tp_size: int = typer.Option(
        1,
        "--tp-size",
        help="Tensor parallel size for multi-GPU",
    ),
    num_gpus: int = typer.Option(
        1,
        "--num-gpus",
        "-g",
        help="Number of GPUs (split across TP and DP automatically)",
    ),
    gpu_ids: str | None = typer.Option(
        None,
        "--gpu-ids",
        help="Comma-separated GPU IDs to use (e.g. '2,3,4,5'). Default: 0 to num_gpus-1",
    ),
    batch_size: str = typer.Option(
        "auto",
        "--batch-size",
        help="Batch size for evaluation ('auto' or integer)",
    ),
    limit: int | None = typer.Option(
        None,
        "-n",
        "--limit",
        help="Limit number of QA pairs (for quick testing)",
    ),
    max_judge_workers: int = typer.Option(
        8,
        "--max-judge-workers",
        help="Max parallel threads for LLM judge calls (only used with --judge)",
    ),
) -> None:
    """Evaluate a model on QA benchmark using lm-eval + SGLang."""
    import os

    from cleanslate.fsid.qa.scoring import evaluate_qa, save_qa_eval

    # Compute dp_size from num_gpus and tp_size
    if num_gpus % tp_size != 0:
        console.print(
            f"[red]Error: --num-gpus ({num_gpus}) must be divisible by --tp-size ({tp_size})[/red]"
        )
        raise typer.Exit(1)
    dp_size = num_gpus // tp_size

    # Set CUDA_VISIBLE_DEVICES for multi-GPU
    if num_gpus > 1:
        if gpu_ids:
            gpus = [int(g.strip()) for g in gpu_ids.split(",")]
            if len(gpus) != num_gpus:
                console.print(
                    f"[red]Error: --gpu-ids has {len(gpus)} GPUs but --num-gpus is {num_gpus}[/red]"
                )
                raise typer.Exit(1)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(g) for g in range(num_gpus)
            )

    # Parse batch_size: allow "auto" or integer string
    bs: int | str
    try:
        bs = int(batch_size)
    except ValueError:
        bs = batch_size

    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Dataset:[/bold] {hf_dataset} ({hf_config})")
    console.print(f"[bold]Judge:[/bold] {'LLM' if use_judge else 'contains_match only'}")
    if tp_size > 1:
        console.print(f"[bold]TP size:[/bold] {tp_size}")
    if dp_size > 1:
        console.print(f"[bold]DP size:[/bold] {dp_size}")

    summary = evaluate_qa(
        model_name=model,
        dataset=hf_dataset,
        config=hf_config,
        use_judge=use_judge,
        max_judge_workers=max_judge_workers,
        mem_fraction=mem_fraction,
        tp_size=tp_size,
        dp_size=dp_size,
        batch_size=bs,
        limit=limit,
    )

    out = save_qa_eval(summary, output)

    console.print()
    console.print(f"[bold]Overall accuracy:[/bold] {summary.overall_accuracy:.1%}")
    console.print(
        f"[bold]Correct:[/bold] {summary.total_correct} / {summary.total_questions}"
    )
    console.print(f"[bold]Content items:[/bold] {len(summary.per_content)}")
    console.print(f"[bold]Eval time:[/bold] {summary.eval_time_seconds:.1f}s")
    console.print(f"[bold]Saved to:[/bold] {out}")


@qa_app.command("score")
def score_qa(
    forget_path: str = typer.Option(
        ...,
        "--forget-path",
        help="Forget set parquet from fsid curate",
    ),
    pre_qa: str = typer.Option(
        ...,
        "--pre-qa",
        help="Pre-unlearning QA eval JSON",
    ),
    post_qa: str = typer.Option(
        ...,
        "--post-qa",
        help="Post-unlearning QA eval JSON",
    ),
    output: str = typer.Option(
        "results/fsid/qa_score.json",
        "-o",
        "--output",
        help="Output JSON path",
    ),
) -> None:
    """Score QA accuracy change split by forget/retain content sets."""
    from rich.table import Table

    from cleanslate.fsid.qa.scoring import compute_qa_score, save_qa_score

    result = compute_qa_score(
        forget_set_path=forget_path,
        pre_qa_path=pre_qa,
        post_qa_path=post_qa,
    )

    out = save_qa_score(result, output)

    # Print Rich table
    table = Table(title="QA Forget/Retain Score")
    table.add_column("Subset", style="bold")
    table.add_column("Content", justify="right")
    table.add_column("Questions", justify="right")
    table.add_column("Pre Acc", justify="right")
    table.add_column("Post Acc", justify="right")
    table.add_column("Delta", justify="right")

    forget = result["forget"]
    retain = result["retain"]
    overall = result["overall"]

    table.add_row(
        "Forget",
        str(forget["num_content"]),
        str(forget["num_questions"]),
        f"{forget['pre_accuracy']:.1%}",
        f"{forget['post_accuracy']:.1%}",
        f"{forget['delta_accuracy']:+.1%}",
    )
    table.add_row(
        "Retain",
        str(retain["num_content"]),
        str(retain["num_questions"]),
        f"{retain['pre_accuracy']:.1%}",
        f"{retain['post_accuracy']:.1%}",
        f"{retain['delta_accuracy']:+.1%}",
    )
    table.add_row(
        "Overall",
        str(overall["num_content"]),
        str(overall["num_questions"]),
        f"{overall['pre_accuracy']:.1%}",
        f"{overall['post_accuracy']:.1%}",
        f"{overall['post_accuracy'] - overall['pre_accuracy']:+.1%}",
    )

    console.print()
    console.print(table)
    console.print()
    console.print(f"[bold]Pre model:[/bold]  {result['pre_model']}")
    console.print(f"[bold]Post model:[/bold] {result['post_model']}")
    console.print(f"[bold]Saved to:[/bold] {out}")
