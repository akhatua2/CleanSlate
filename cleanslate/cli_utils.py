"""CLI utility functions for CleanSlate."""

import json
import math
import re
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except ImportError:
    raise ImportError("CLI dependencies not installed. Install with: pip install cleanslate[cli]")


def generate_hf_repo_name(org: str, dataset: str, model_clean: str) -> str:
    """Generate HF repo name for eval results.

    Format: {org}/eval_{dataset}_{model}
    Example: unlearning-cleanslate/eval_cleanslate_dataset_llama-3_1-8b
    """
    def sanitize(s: str) -> str:
        s = s.split("/")[-1]
        s = re.sub(r"[^a-z0-9_-]", "_", s.lower())
        s = re.sub(r"_+", "_", s)
        return s.strip("_")

    return f"{org}/eval_{sanitize(dataset)}_{model_clean}"


def make_flame_graph(windows: list, width: int = 20, metric: str = "p_z") -> str:
    """Create a mini flame graph showing memorization/reproduction across windows.

    Args:
        windows: List of window dicts with metric values
        width: Exact width in characters (will pad/resample to match)
        metric: Which metric to visualize - "p_z" (log scale) or "rouge_l" (linear 0-1)

    Returns:
        Rich-formatted string with colored blocks, always exactly `width` chars
    """
    if not windows:
        return "[dim]" + "░" * width + "[/dim]"

    if metric == "p_z":
        values = [w.get("p_z", 0) for w in windows]
        if not values or max(values) == 0:
            return "[dim]" + "░" * width + "[/dim]"
        # Normalize using log scale (p_z values vary by orders of magnitude)
        log_values = [math.log10(max(p, 1e-30)) for p in values]
        min_log, max_log = min(log_values), max(log_values)
        if max_log == min_log:
            normalized = [0.5] * len(log_values)
        else:
            normalized = [(v - min_log) / (max_log - min_log) for v in log_values]
    else:
        # ROUGE-L or other 0-1 metrics - linear scale
        values = [w.get("max_rouge_l", 0) or w.get("best_beam_rouge_l", 0) or 0 for w in windows]
        if not values or max(values) == 0:
            return "[dim]" + "░" * width + "[/dim]"
        # Linear normalization (already 0-1 scale)
        normalized = values

    # Resample to exact width
    if len(normalized) != width:
        step = len(normalized) / width
        normalized = [normalized[min(int(i * step), len(normalized) - 1)] for i in range(width)]

    # Map to blocks and colors
    blocks = []
    for v in normalized:
        if v < 0.25:
            blocks.append("[dim]░[/dim]")
        elif v < 0.5:
            blocks.append("[dim]▒[/dim]")
        elif v < 0.75:
            blocks.append("[yellow]▓[/yellow]")
        else:
            blocks.append("[red]█[/red]")

    return "".join(blocks)


def get_gpu_handles(gpu_ids: list[int]) -> dict[int, Any]:
    """Initialize pynvml and get GPU handles.

    Returns:
        Dict mapping gpu_id to nvml handle, or empty dict if pynvml unavailable
    """
    gpu_handles = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        for gpu_id in gpu_ids:
            gpu_handles[gpu_id] = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    except Exception:
        pass
    return gpu_handles


def get_gpu_stats(gpu_id: int, gpu_handles: dict[int, Any]) -> str:
    """Get GPU utilization and memory stats.

    Returns:
        Formatted string like "87% 18G" or empty string if unavailable
    """
    if gpu_id not in gpu_handles:
        return ""
    try:
        import pynvml
        handle = gpu_handles[gpu_id]
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_gb = mem.used / (1024**3)
        return f"[dim]{util:>3}% {mem_gb:>4.0f}G[/dim]"
    except Exception:
        return ""


def shutdown_gpu_handles(gpu_handles: dict[int, Any]) -> None:
    """Shutdown pynvml if it was initialized."""
    if gpu_handles:
        try:
            import pynvml
            pynvml.nvmlShutdown()
        except Exception:
            pass


def estimate_windows(text: str, window_size: int, stride: int) -> int:
    """Estimate number of windows for a text (rough, uses char-based estimation).

    Uses ~4 chars per token as rough estimate to avoid loading tokenizer.
    """
    chars_per_token = 4  # Rough estimate
    approx_tokens = len(text) // chars_per_token
    window_total = window_size * 2  # prefix + suffix
    if approx_tokens < window_total:
        return 0
    return max(1, (approx_tokens - window_total) // stride + 1)


def greedy_bin_pack(items: list[tuple[int, int]], num_bins: int) -> list[list[int]]:
    """Greedy bin-packing: assign items to bins to balance total weight.

    Args:
        items: List of (index, weight) tuples, should be sorted by weight descending
        num_bins: Number of bins (GPUs)

    Returns:
        List of bins, each containing list of item indices
    """
    bins: list[list[int]] = [[] for _ in range(num_bins)]
    bin_weights = [0] * num_bins

    for idx, weight in items:
        # Find bin with smallest total weight
        min_bin = min(range(num_bins), key=lambda i: bin_weights[i])
        bins[min_bin].append(idx)
        bin_weights[min_bin] += weight

    return bins


def run_parallel_eval(
    console: Console,
    num_gpus: int,
    gpu_ids: str | None,
    hf_dataset: str,
    hf_config: str,
    text_col: str,
    limit: int | None,
    shuffle: bool,
    seed: int,
    model: str,
    backend: str,
    mem_fraction: float,
    cuda_graph_bs: int,
    window_size: int,
    stride: int,
    threshold: float,
    reproduce: bool,
    reproduce_all: bool,
    decode_strategy: str,
    beam_width: int,
    length_penalty: float,
    max_batch_prompts: int,
    temperature: float,
    gen_top_k: int,
    capture_top_k: int,
    output_dir: str,
    hf_push: str | None,
    no_push: bool,
    hf_org: str,
) -> None:
    """Run evaluation in parallel across multiple GPUs."""
    from datasets import Dataset, load_dataset
    import datasets
    datasets.disable_progress_bar()

    console.print(f"[bold blue]Multi-GPU mode: {num_gpus} GPUs[/bold blue]")

    # Parse GPU IDs
    if gpu_ids:
        gpus = [int(g.strip()) for g in gpu_ids.split(",")]
        if len(gpus) != num_gpus:
            console.print(f"[red]Error: --gpu-ids has {len(gpus)} GPUs but --num-gpus is {num_gpus}[/red]")
            return
    else:
        gpus = list(range(num_gpus))

    console.print(f"[bold]Using GPUs:[/bold] {gpus}")

    # Load items and estimate windows for balanced splitting
    console.print("[bold]Loading items and estimating windows...[/bold]")
    ds = load_dataset(hf_dataset, hf_config, split="train")

    if shuffle:
        ds = ds.shuffle(seed=seed)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    # Estimate window count per item
    item_windows: list[tuple[int, int]] = []  # (item_idx, estimated_windows)
    for i, row in enumerate(ds):
        text = row.get(text_col, "")
        est_windows = estimate_windows(text, window_size, stride)
        item_windows.append((i, est_windows))

    total_items = len(item_windows)
    total_windows_est = sum(w for _, w in item_windows)
    console.print(f"  Total items: {total_items}")
    console.print(f"  Estimated windows: {total_windows_est}")

    # Sort by window count (largest first) for better bin-packing
    item_windows_sorted = sorted(item_windows, key=lambda x: x[1], reverse=True)

    # Greedy bin-packing to balance windows across GPUs
    gpu_assignments = greedy_bin_pack(item_windows_sorted, num_gpus)

    # Build base command
    base_cmd = [
        sys.executable, "-m", "cleanslate.cli", "eval",
        "--hf-dataset", hf_dataset,
        "--hf-config", hf_config,
        "--text-col", text_col,
        "--model", model,
        "--backend", backend,
        "--mem-fraction", str(mem_fraction),
        "--cuda-graph-bs", str(cuda_graph_bs),
        "--window-size", str(window_size),
        "--stride", str(stride),
        "--threshold", str(threshold),
        "--decode", decode_strategy,
        "--beam-width", str(beam_width),
        "--length-penalty", str(length_penalty),
        "--max-batch-prompts", str(max_batch_prompts),
        "--temperature", str(temperature),
        "--gen-top-k", str(gen_top_k),
        "--capture-top-k", str(capture_top_k),
        "--output", output_dir,
        "--seed", str(seed),
        "--num-gpus", "1",
    ]

    if reproduce:
        base_cmd.append("--reproduce")
    if reproduce_all:
        base_cmd.append("--reproduce-all")
    else:
        base_cmd.append("--reproduce-best-only")
    # Don't pass shuffle - we already shuffled and assigned indices

    # Create progress and assignments directories
    progress_dir = Path(output_dir) / hf_config / ".progress"
    assignments_dir = Path(output_dir) / hf_config / ".assignments"
    progress_dir.mkdir(parents=True, exist_ok=True)
    assignments_dir.mkdir(parents=True, exist_ok=True)

    for f in progress_dir.glob("part_*.json"):
        f.unlink()
    for f in assignments_dir.glob("part_*.json"):
        f.unlink()

    # Save assignment files
    for i, indices in enumerate(gpu_assignments):
        assignment_file = assignments_dir / f"part_{i}.json"
        with open(assignment_file, "w") as f:
            json.dump({"indices": indices}, f)

    # Launch workers
    console.print()
    processes = []
    for i, indices in enumerate(gpu_assignments):
        gpu_id = gpus[i]
        part_name = f"part_{i}"
        assignment_file = assignments_dir / f"part_{i}.json"
        est_windows = sum(item_windows[idx][1] for idx in indices)

        cmd = base_cmd + [
            "--indices-file", str(assignment_file),
            "--part-id", part_name,
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        processes.append((i, gpu_id, est_windows, proc))

    # Initialize GPU monitoring
    gpu_handles = get_gpu_handles(gpus)

    # Build process info with both item count and estimated windows
    process_info = []
    for i, indices in enumerate(gpu_assignments):
        gpu_id = gpus[i]
        num_items = len(indices)
        est_windows = sum(item_windows[idx][1] for idx in indices)
        proc = processes[i][3]  # Get proc from processes list
        process_info.append((i, gpu_id, num_items, est_windows, proc))

    # Monitor progress (track items, but total work is balanced by windows)
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total} items"),
        TextColumn("[dim]~{task.fields[est_windows]}w[/dim]"),
        TextColumn("{task.fields[gpu_stats]}"),
        TimeElapsedColumn(),
        TextColumn("→"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        tasks = {}
        for i, gpu_id, num_items, est_windows, proc in process_info:
            task_id = progress.add_task(f"GPU {gpu_id}", total=num_items, completed=0, gpu_stats="", est_windows=est_windows)
            tasks[i] = (task_id, gpu_id, est_windows)

        while True:
            all_done = True
            for i, gpu_id, num_items, est_windows, proc in process_info:
                ret = proc.poll()
                if ret is None:
                    all_done = False

                progress_file = progress_dir / f"part_{i}.json"
                completed = 0
                if progress_file.exists():
                    try:
                        with open(progress_file) as f:
                            data = json.load(f)
                        completed = data.get("current", 0)
                    except (json.JSONDecodeError, IOError):
                        pass

                task_id, gpu_id, est_w = tasks[i]
                progress.update(task_id, completed=completed, gpu_stats=get_gpu_stats(gpu_id, gpu_handles), est_windows=est_w)

            if all_done:
                for i, gpu_id, num_items, est_windows, proc in process_info:
                    progress_file = progress_dir / f"part_{i}.json"
                    if progress_file.exists():
                        try:
                            with open(progress_file) as f:
                                data = json.load(f)
                            task_id, gpu_id, est_w = tasks[i]
                            progress.update(task_id, completed=data.get("current", 0), gpu_stats="", est_windows=est_w)
                        except (json.JSONDecodeError, IOError):
                            pass
                break

            time.sleep(0.5)

    shutdown_gpu_handles(gpu_handles)

    # Check for failures
    failed = []
    for i, gpu_id, num_items, est_windows, proc in process_info:
        ret = proc.wait()
        if ret != 0:
            failed.append((i, gpu_id, ret))
            console.print(f"[red]Worker {i} (GPU {gpu_id}) failed with code {ret}[/red]")
            stdout, _ = proc.communicate()
            if stdout:
                console.print(f"[dim]{stdout.decode()[-2000:]}[/dim]")

    if failed:
        console.print(f"[red bold]{len(failed)} workers failed.[/red bold]")
        return

    # Merge results
    output_path = Path(output_dir) / hf_config
    all_records = []

    for i in range(num_gpus):
        part_path = output_path / f"part_{i}" / "results.parquet"
        if part_path.exists():
            part_ds = Dataset.from_parquet(str(part_path))
            all_records.extend(part_ds.to_list())
        else:
            console.print(f"[yellow]Warning: part_{i} not found[/yellow]")

    if not all_records:
        console.print("[red]No records to merge![/red]")
        return

    # Save merged results
    merged_ds = Dataset.from_list(all_records)
    merged_parquet = output_path / "results.parquet"
    merged_ds.to_parquet(str(merged_parquet))

    # Compute summary stats
    items_with_memorization = sum(1 for r in all_records if r.get("memorized_windows", 0) > 0)
    total_memorized_windows = sum(r.get("memorized_windows", 0) for r in all_records)
    total_windows = sum(r.get("num_windows", 0) for r in all_records)

    # Reproduction stats (from windows)
    items_with_reproduction = sum(1 for r in all_records if r.get("reproduced_windows", 0) > 0)
    total_reproduced_windows = sum(r.get("reproduced_windows", 0) for r in all_records)

    sorted_by_pz = sorted(all_records, key=lambda r: r.get("max_p_z", 0), reverse=True)
    sorted_by_rouge = sorted(all_records, key=lambda r: r.get("avg_rouge_l", 0), reverse=True)
    top_memorized = sorted_by_pz[:3]
    top_reproduced = sorted_by_rouge[:3]

    # Display summary
    console.print()
    console.print("[bold]Summary:[/bold]")
    console.print(f"  Items: {len(all_records)}")
    console.print()
    console.print("[bold]  Memorization[/bold] [dim](p_z > τ_min, teacher forcing)[/dim]")
    console.print(f"    Items with memorization: {items_with_memorization}/{len(all_records)} ({items_with_memorization/len(all_records):.1%})")
    if total_windows > 0:
        console.print(f"    Memorized windows: {total_memorized_windows}/{total_windows} ({total_memorized_windows/total_windows:.1%})")
    console.print()
    console.print("[bold]  Reproduction[/bold] [dim](ROUGE-L > threshold, beam generation)[/dim]")
    console.print(f"    Items with reproduction: {items_with_reproduction}/{len(all_records)} ({items_with_reproduction/len(all_records):.1%})")
    if total_windows > 0:
        console.print(f"    Reproduced windows: {total_reproduced_windows}/{total_windows} ({total_reproduced_windows/total_windows:.1%})")

    def get_item_title(record: dict) -> str:
        """Get display title from record."""
        title = record.get("content_title") or record.get("content_id") or "Unknown"
        return str(title)[:25]

    console.print()
    console.print("[bold]Top memorized[/bold] [dim](by p_z)[/dim]:")
    for i, item in enumerate(top_memorized, 1):
        title = get_item_title(item)
        pz = item.get("max_p_z", 0)
        windows = item.get("windows", [])
        flame_pz = make_flame_graph(windows, width=20, metric="p_z")
        console.print(f"  {i}. {title:<25} {flame_pz} p_z={pz:.2e}")

    console.print()
    console.print("[bold]Top reproduced[/bold] [dim](by avg ROUGE-L)[/dim]:")
    for i, item in enumerate(top_reproduced, 1):
        title = get_item_title(item)
        avg_rouge = item.get("avg_rouge_l", 0)
        windows = item.get("windows", [])
        flame_rouge = make_flame_graph(windows, width=20, metric="rouge_l")
        console.print(f"  {i}. {title:<25} {flame_rouge} avg={avg_rouge:.2f}")

    console.print()
    console.print(f"[dim]Saved: {merged_parquet}[/dim]")

    # Push to HuggingFace (default behavior)
    if not no_push:
        # Derive clean model name for repo
        model_clean = model.split("/")[-1].lower().replace(".", "_")
        repo_name = hf_push or generate_hf_repo_name(hf_org, hf_dataset, model_clean)
        merged_ds.push_to_hub(repo_name, split="train")
        console.print(f"[green]Pushed: https://huggingface.co/datasets/{repo_name}[/green]")
