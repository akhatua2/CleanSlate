"""Per-model FSID curation + HF push.

Builds a forget pool from the eval HF dataset, runs every supported curation
rule against that pool, and pushes everything to a single HF dataset under
two layers: subsets (config_name) and splits.

Target layout for `unlearning-cleanslate/fsid-curated-test-<model_clean>`:
    forget_pool / train                 -- song-level forget rows (incl. lyrics)
    retain_pool / train                 -- eval rows whose content_id is NOT in forget_pool
    forget      / <rule>                -- per-rule forget_set (windowed)
    retain      / <rule>                -- per-rule retain_set (wikitext + rule_name)

Usage:
    python scripts/fsid_curate_and_push.py --model qwen-8b
"""
import re
from pathlib import Path

import typer
from datasets import Dataset, load_dataset
from rich.console import Console

from cleanslate.cli.aliases import resolve_model
from cleanslate.fsid.datasets.pool import build_candidate_pool, split_pool_records
from cleanslate.fsid.rules import get_forget_set_rule

console = Console()

RULES: tuple[str, ...] = ("baseline", "bm25-10B", "bm25-6T", "igm-10B")


def _save_and_push(
    rows: list[dict[str, any]],
    repo: str,
    config_name: str,
    split: str,
    local_root: Path,
) -> None:
    ds = Dataset.from_list(rows)
    out_dir = local_root / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.parquet"
    ds.to_parquet(str(out_path))
    console.print(f"[cyan]Saved {config_name}/{split}: {len(rows)} rows -> {out_path}[/cyan]")
    ds.push_to_hub(repo, config_name=config_name, split=split)
    console.print(f"[green]Pushed {config_name}/{split}: {len(rows)} rows -> {repo}[/green]")


def main(
    model: str = typer.Option(..., "--model", "-m"),
    hub_repo_id: str = typer.Option(..., "--hub-repo-id"),
    eval_dataset: str = typer.Option(..., "--eval-dataset"),
) -> None:
    _, model_clean = resolve_model(model)
    repo = hub_repo_id
    local_root = Path("results") / "datapath" / model_clean
    local_root.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Model:[/bold] {model} -> {model_clean}")
    console.print(f"[bold]Eval source:[/bold] {eval_dataset}")
    console.print(f"[bold]Hub repo:[/bold] {repo}")
    console.print(f"[bold]Local root:[/bold] {local_root}")
    console.print(f"[bold]Rules:[/bold] {', '.join(RULES)}")

    pool_rows, stats = build_candidate_pool(
        eval_dataset=eval_dataset,
        fraction_col="memorized_fraction",
        threshold=0.05,
        target_size=100,
        min_size=100,
        max_size=100,
        num_bins=5,
        seed=42,
        deduplicate=True,
        source_dataset="unlearning-cleanslate/cleanslate_dataset",
        source_text_col="reference_target",
    )
    forget_pool_rows, _ = split_pool_records(
        pool_rows, forget_fraction=None, num_bins=5, seed=42,
    )
    console.print(
        f"[bold]Pool:[/bold] scanned={stats.total_rows_seen} eligible={stats.eligible_rows} "
        f"selected={stats.selected_rows} mem_fraction min={stats.min_fraction:.3f} "
        f"mean={stats.mean_fraction:.3f} max={stats.max_fraction:.3f}"
    )

    forget_ids = {str(r["content_id"]) for r in forget_pool_rows}
    eval_ds = load_dataset(eval_dataset, "default", split="train")
    retain_pool_rows = [dict(r) for r in eval_ds if str(r.get("content_id")) not in forget_ids]
    console.print(
        f"[bold]Retain pool:[/bold] {len(retain_pool_rows)} rows "
        f"(eval rows not in forget_pool)"
    )

    curated: dict[str, tuple[list[dict[str, any]], list[dict[str, any]]]] = {}
    for rule_name in RULES:
        console.print(f"\n[bold]== Curating rule: {rule_name} ==[/bold]")
        rule = get_forget_set_rule(rule_name)
        forget_rows, retain_rows = rule.curate(
            forget_pool_rows,
            window_size=100,
            stride=10,
            base_dataset=None,
            seed=42,
            retain_dataset="wikitext",
            retain_config="wikitext-103-raw-v1",
            retain_split="validation",
            retain_samples=None,
            max_hits_per_song=128,
            max_windows_per_hit=25,
        )
        console.print(
            f"  forget rows={len(forget_rows)}  retain rows={len(retain_rows)}  "
            f"artifact_rule_name={rule.artifact_rule_name}"
        )
        curated[rule_name] = (forget_rows, retain_rows)

    console.print("\n[bold]== Saving locally + pushing to HF ==[/bold]")
    _save_and_push(forget_pool_rows, repo, "forget_pool", "train", local_root)
    _save_and_push(retain_pool_rows, repo, "retain_pool", "train", local_root)
    for rule_name, (f_rows, r_rows) in curated.items():
        split = re.sub(r"\W+", "_", rule_name)
        _save_and_push(f_rows, repo, "forget", split, local_root)
        _save_and_push(r_rows, repo, "retain", split, local_root)

    console.print(f"\n[bold green]Done.[/bold green] Repo: {repo}  Local: {local_root}")


if __name__ == "__main__":
    typer.run(main)
