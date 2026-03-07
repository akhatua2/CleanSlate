"""CleanSlate CLI - Sliding window memorization evaluation."""

import os
import warnings

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Suppress SGLang/NCCL/Gloo verbose output (must be before any imports)
os.environ["SGLANG_LOG_LEVEL"] = "error"
os.environ["GLOO_LOG_LEVEL"] = "0"
os.environ["GLOO_LOG_LEVEL_STR"] = "OFF"
os.environ["NCCL_DEBUG"] = "ERROR"
os.environ["NCCL_DEBUG_SUBSYS"] = "INIT"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "1"

# Suppress verbose warnings from transformers/torch
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")
warnings.filterwarnings("ignore", message=".*cuda.cudart module is deprecated.*")
warnings.filterwarnings("ignore", message=".*cuda.nvrtc module is deprecated.*")
warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
warnings.filterwarnings("ignore", message=".*Gloo.*")

try:
    import typer
    from rich.console import Console
except ImportError:
    raise ImportError(
        "CLI dependencies not installed. Install with: pip install cleanslate[cli]"
    )

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from cleanslate.evaluator import (
    MemorizationConfig,
    MemorizationEvaluator,
)
from cleanslate.decoding import create_strategy
from cleanslate.cli_utils import run_parallel_eval, generate_hf_repo_name
from cleanslate.fsid.cli import fsid_app

# Default dataset for memorization evaluation
DEFAULT_DATASET = "unlearning-cleanslate/cleanslate_dataset"

# Metadata columns to copy from source dataset to eval results
METADATA_COLUMNS = ["content_id", "content_title", "content_creators", "content_year"]

# Model aliases: short name -> (full HF path, clean name for repos)
MODEL_ALIASES: dict[str, tuple[str, str]] = {
    "llama-8b": ("meta-llama/Llama-3.1-8B", "llama-3_1-8b"),
    "qwen-8b": ("Qwen/Qwen3-8B", "qwen3-8b"),
    "qwen-8b-base": ("Qwen/Qwen3-8B-Base", "qwen3-8b-base"),
    "olmo-7b": ("allenai/Olmo-3-1025-7B", "olmo-3-7b"),
    "gemma-12b": ("google/gemma-3-12b-pt", "gemma-3-12b"),
    "nemotron-9b": ("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "nemotron-nano-9b"),
    "trinity-nano": ("arcee-ai/Trinity-Nano-Base", "trinity-nano"),
    "gpt-oss-20b": ("openai/gpt-oss-20b", "gpt-oss-20b"),
}


def resolve_model(model: str) -> tuple[str, str]:
    """Resolve model alias to (full_path, clean_name)."""
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    # Not an alias - use as-is, derive clean name
    clean = model.split("/")[-1].lower().replace(".", "_")
    return model, clean


def get_item_name(item: dict[str, Any]) -> str:
    """Get display name for an item."""
    return str(item.get("content_title") or item.get("content_id") or "unknown")


def copy_metadata(item: dict[str, Any], target: dict[str, Any]) -> None:
    """Copy metadata columns from source item to target record."""
    for col in METADATA_COLUMNS:
        if col in item:
            target[col] = item[col]
    # Emit song_id/song_title aliases so FSID scorer can match by either name
    if "content_id" in item and "song_id" not in target:
        target["song_id"] = item["content_id"]
    if "content_title" in item and "song_title" not in target:
        target["song_title"] = item["content_title"]

app = typer.Typer(
    name="cleanslate",
    help="Sliding window memorization evaluation: compute p_z + generate + compare.",
)
app.add_typer(fsid_app, name="fsid")
console = Console()


@app.command()
def eval(
    # Dataset source
    hf_dataset: str = typer.Option(
        DEFAULT_DATASET,
        "--hf-dataset",
        "-d",
        help="HuggingFace dataset ID (default: cleanslate_dataset with 250 memorized items)",
    ),
    hf_config: str = typer.Option(
        "default",
        "--hf-config",
        "-c",
        help="Dataset config/split (e.g. direct__first_line). Default: 'default'",
    ),
    # Column mapping
    text_col: str = typer.Option(
        "reference_target",
        "--text-col",
        help="Column containing the text to evaluate",
    ),
    # Sampling
    limit: int | None = typer.Option(
        None, "--limit", "-n", help="Limit number of items to process"
    ),
    shuffle: bool = typer.Option(
        False, "--shuffle/--no-shuffle", help="Shuffle before selecting"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for shuffle"),
    offset: int = typer.Option(
        0, "--offset", help="Skip first N items (for parallel splits)"
    ),
    part_id: str | None = typer.Option(
        None, "--part-id", help="Part ID for parallel runs (e.g., 'part_0')"
    ),
    indices_file: str | None = typer.Option(
        None, "--indices-file", help="JSON file with item indices to process"
    ),
    # Model
    model: str = typer.Option(
        "Qwen/Qwen2.5-0.5B-Instruct", "--model", "-m", help="Model name"
    ),
    backend: str = typer.Option(
        "sglang", "--backend", "-b", help="Backend: sglang or huggingface"
    ),
    # SGLang memory settings
    mem_fraction: float = typer.Option(
        0.85, "--mem-fraction", help="GPU memory fraction for SGLang (default: 0.85)"
    ),
    cuda_graph_bs: int = typer.Option(
        1, "--cuda-graph-bs", help="Max batch size for CUDA graphs (default: 1)"
    ),
    # Sliding window settings
    window_size: int = typer.Option(
        50, "--window-size", "-w", help="Window size in tokens (prefix and suffix each)"
    ),
    stride: int = typer.Option(
        25, "--stride", help="Stride in tokens (default: half of window_size)"
    ),
    # Compute settings
    threshold: float = typer.Option(
        0.001,
        "--threshold",
        "-t",
        help="Memorization threshold tau_min (0.001 = 0.1%%)",
    ),
    reproduction_threshold: float = typer.Option(
        0.7,
        "--reproduction-threshold",
        help="ROUGE-L threshold for 'reproduced' (any beam >= this). Paper doesn't specify, 0.7 is reasonable.",
    ),
    # Generation settings
    reproduce: bool = typer.Option(
        False, "--reproduce", "-r", help="Enable reproduction check (generate + compare with BLEU/ROUGE)"
    ),
    reproduce_all: bool = typer.Option(
        True, "--reproduce-all/--reproduce-best-only", help="Generate for all windows or only best p_z window"
    ),
    # Decoding strategy
    decode_strategy: str = typer.Option(
        "beam", "--decode", "-D",
        help="Decoding strategy: greedy, beam"
    ),
    beam_width: int = typer.Option(
        5, "--beam-width", help="Beam width for beam search (only used with --decode beam)"
    ),
    length_penalty: float = typer.Option(
        1.0, "--length-penalty", help="Length penalty for beam search (1.0 = linear, 0.0 = none)"
    ),
    max_batch_prompts: int = typer.Option(
        64, "--max-batch-prompts", help="Max prompts per SGLang engine call for compute/generate. Reduce if OOM (default: 64)"
    ),
    # Legacy generation params (used for sampling strategy in future)
    temperature: float = typer.Option(
        0.0, "--temperature", help="Generation temperature (0 = greedy). Currently unused with --decode."
    ),
    gen_top_k: int = typer.Option(40, "--gen-top-k", help="Top-k for generation. Currently unused with --decode."),
    capture_top_k: int = typer.Option(
        5, "--capture-top-k", help="Top-k alternatives to save per token (0 = none)"
    ),
    # Output
    output_dir: str = typer.Option(
        "results/eval", "--output", "-o", help="Local output directory"
    ),
    hf_push: str | None = typer.Option(
        None,
        "--hf-push",
        help="Override HF repo name (default: auto-generated)",
    ),
    no_push: bool = typer.Option(
        False,
        "--no-push",
        help="Skip pushing to HuggingFace (local only)",
    ),
    hf_org: str = typer.Option(
        "unlearning-cleanslate",
        "--hf-org",
        help="HuggingFace org (default: unlearning-cleanslate)",
    ),
    # Parallelism
    num_gpus: int = typer.Option(
        1, "--num-gpus", "-g",
        help="Number of GPUs for data parallelism"
    ),
    gpu_ids: str | None = typer.Option(
        None, "--gpu-ids",
        help="Comma-separated GPU IDs to use (e.g. '2,3,4,5'). Default: 0 to num_gpus-1"
    ),
):
    """
    Sliding window memorization evaluation.

    By default evaluates the cleanslate_dataset (250 memorized songs/books).

    For each item:
    1. Slide a window across the text
    2. Each window: prefix (seed) + suffix (target)
    3. Compute p_z via teacher forcing
    4. (Optional) With --reproduce: generate and compare with BLEU/ROUGE

    Output includes max_p_z, mean_p_z, coverage, and most memorized window details.
    """
    import json
    import os
    import statistics

    # Resolve model alias
    model_path, model_clean = resolve_model(model)

    # Handle multi-GPU parallelism
    if num_gpus > 1 and part_id is None:
        if backend == "sglang":
            # SGLang native DP: set CUDA_VISIBLE_DEVICES and use dp_size
            if gpu_ids:
                gpus = [int(g.strip()) for g in gpu_ids.split(",")]
                if len(gpus) != num_gpus:
                    console.print(f"[red]Error: --gpu-ids has {len(gpus)} GPUs but --num-gpus is {num_gpus}[/red]")
                    raise typer.Exit(1)
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
            else:
                gpus = list(range(num_gpus))
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
            # Continue to single-process eval with dp_size set below
        else:
            # HuggingFace backend: use subprocess-based parallelism
            run_parallel_eval(
                console=console,
                num_gpus=num_gpus,
                gpu_ids=gpu_ids,
                hf_dataset=hf_dataset,
                hf_config=hf_config,
                text_col=text_col,
                limit=limit,
                shuffle=shuffle,
                seed=seed,
                model=model,
                backend=backend,
                mem_fraction=mem_fraction,
                cuda_graph_bs=cuda_graph_bs,
                window_size=window_size,
                stride=stride,
                threshold=threshold,
                reproduce=reproduce,
                reproduce_all=reproduce_all,
                decode_strategy=decode_strategy,
                beam_width=beam_width,
                length_penalty=length_penalty,
                max_batch_prompts=max_batch_prompts,
                temperature=temperature,
                gen_top_k=gen_top_k,
                capture_top_k=capture_top_k,
                output_dir=output_dir,
                hf_push=hf_push,
                no_push=no_push,
                hf_org=hf_org,
            )
            return

    console.print(f"[bold]Dataset:[/bold] {hf_dataset} / {hf_config}")
    console.print(f"[bold]Model:[/bold] {model_path}")
    console.print(f"[bold]Backend:[/bold] {backend}")
    console.print(f"[bold]Window:[/bold] {window_size} tokens prefix + {window_size} tokens suffix")
    console.print(f"[bold]Stride:[/bold] {stride} tokens")
    console.print(f"[bold]Reproduce:[/bold] {reproduce}")
    if reproduce:
        strategy_info = f"{decode_strategy}"
        if decode_strategy == "beam":
            strategy_info += f" (width={beam_width}, length_penalty={length_penalty}, batch={max_batch_prompts})"
        console.print(f"[bold]Decode:[/bold] {strategy_info}")
    console.print()

    # Load dataset
    console.print("[bold]Loading dataset...[/bold]")

    if indices_file:
        # Load specific indices from assignment file (for balanced parallel splits)
        with open(indices_file) as f:
            assignment = json.load(f)
        indices = assignment["indices"]

        # Load full dataset and select specific indices
        ds = load_dataset(hf_dataset, hf_config, split="train")
        items = [ds[i] for i in indices]
        console.print(f"  Loaded {len(items)} items from indices file")
    else:
        # Original streaming approach with offset/limit
        ds_stream = load_dataset(hf_dataset, hf_config, split="train", streaming=True)

        items = []
        for i, row in enumerate(ds_stream):
            if i < offset:
                continue
            if limit and len(items) >= limit:
                break
            items.append(row)

        if shuffle:
            import random
            random.seed(seed)
            random.shuffle(items)

        console.print(f"  Loaded {len(items)} items")
    console.print()

    # Validate text column
    if items and text_col not in items[0]:
        console.print(f"[red]Error: text column '{text_col}' not found[/red]")
        console.print(f"Available columns: {list(items[0].keys())}")
        raise typer.Exit(1)

    # Create config
    config = MemorizationConfig(
        compute_top_k=40,
        tau_min=threshold,
        do_generate=reproduce,
        gen_temperature=temperature,
        gen_top_k=gen_top_k,
        gen_max_new_tokens=window_size,  # Generate same length as suffix
        gen_do_sample=(temperature > 0),
        capture_top_k=capture_top_k,
        # Decoding strategy
        decode_strategy=decode_strategy,
        beam_width=beam_width,
        length_penalty=length_penalty,
        # Reproduction detection
        reproduction_threshold=reproduction_threshold,
        # Sliding window
        window_prefix_len=window_size,
        window_suffix_len=window_size,
        window_stride=stride,
    )

    # Create decoding strategy for direct use
    if decode_strategy == "beam":
        strategy = create_strategy(
            "beam",
            beam_width=beam_width,
            length_penalty=length_penalty,
            max_batch_prompts=max_batch_prompts,
        )
    else:
        strategy = create_strategy("greedy")

    # Create evaluator
    console.print("[bold]Loading model...[/bold]")
    backend_kwargs: dict[str, any] = {}
    if backend == "sglang":
        backend_kwargs["mem_fraction_static"] = mem_fraction
        backend_kwargs["cuda_graph_max_bs"] = cuda_graph_bs
        # Use native DP if num_gpus > 1
        if num_gpus > 1:
            backend_kwargs["dp_size"] = num_gpus
    evaluator = MemorizationEvaluator(
        model=model_path,
        backend=backend,
        config=config,
        **backend_kwargs,
    )

    # Process each song with sliding window
    if not part_id:
        console.print()
        console.print("[bold]Running sliding window evaluation...[/bold]")

    all_records = []  # One row per song (aggregated)
    total_windows = 0
    total_memorized = 0
    total_reproduced = 0

    # Progress file for parallel workers
    progress_file = None
    if part_id:
        progress_dir = Path(output_dir) / hf_config / ".progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        progress_file = progress_dir / f"{part_id}.json"

    def update_progress(current: int, total: int):
        if progress_file:
            import json as json_mod
            with open(progress_file, "w") as f:
                json_mod.dump({"current": current, "total": total}, f)

    from tqdm import tqdm
    from cleanslate.cli_utils import estimate_windows

    # Target windows per batch (balances GPU utilization vs memory)
    TARGET_WINDOWS_PER_BATCH = 500

    if not reproduce:
        # OPTIMIZED PATH: Batch by window count for balanced GPU utilization
        # Filter valid items and estimate windows
        valid_items = []
        for i, s in enumerate(items):
            text = s.get(text_col, "")
            if text and len(text.strip()) >= 10:
                est_windows = estimate_windows(text, window_size, stride)
                valid_items.append((i, s, est_windows))

        # Create batches targeting ~TARGET_WINDOWS_PER_BATCH windows each
        batches = []
        current_batch = []
        current_windows = 0
        for item_data in valid_items:
            idx, item, est_win = item_data
            if current_windows + est_win > TARGET_WINDOWS_PER_BATCH and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_windows = 0
            current_batch.append((idx, item, est_win))
            current_windows += est_win
        if current_batch:
            batches.append(current_batch)

        total_est_windows = sum(w for _, _, w in valid_items)
        if not part_id:
            console.print(f"  Estimated windows: {total_est_windows} ({len(batches)} batches)")

        batch_iter = batches if part_id else tqdm(batches, desc="Processing batches")

        for batch in batch_iter:
            batch_items = [(idx, item) for idx, item, _ in batch]

            # Prepare batch data
            texts = [item[text_col] for _, item in batch_items]
            names = [get_item_name(item) for _, item in batch_items]

            # Run batched sliding window
            sw_results = evaluator.sliding_window_batch(
                texts=texts,
                names=names,
                show_progress=False,
                max_batch_prompts=max_batch_prompts,
            )

            # Process results
            for (orig_idx, item), sw_result in zip(batch_items, sw_results):
                update_progress(orig_idx + 1, len(items))

                if not sw_result.windows:
                    continue

                text = item[text_col]

                # Find most memorized window
                best_window = max(sw_result.windows, key=lambda w: w.p_z)

                # Aggregate metrics
                p_z_values = [w.p_z for w in sw_result.windows]
                ppl_values = [w.perplexity for w in sw_result.windows if w.perplexity > 0]

                # Store all windows as array
                windows_array = []
                for w in sw_result.windows:
                    window_data = {
                        "idx": w.window_idx,
                        "start_char": w.window_start_char,
                        "end_char": w.window_end_char,
                        "seed": w.seed,
                        "target": w.target,
                        "p_z": w.p_z,
                        "log_prob": w.log_prob,
                        "perplexity": w.perplexity,
                        "is_memorized": w.is_memorized,
                        "rank_1_fraction": w.rank_1_fraction,
                    }
                    windows_array.append(window_data)

                record = {
                    # Evaluation metrics
                    "text_length_chars": len(text),
                    "num_windows": sw_result.total_windows,
                    "memorized_windows": sw_result.memorized_windows,
                    "memorized_fraction": sw_result.memorized_windows / sw_result.total_windows,
                    "reproduced_windows": 0,
                    "reproduced_fraction": 0.0,
                    "avg_rouge_l": 0.0,
                    "max_rouge_l": 0.0,
                    "coverage": sw_result.coverage,
                    # p_z statistics
                    "max_p_z": sw_result.max_p_z,
                    "mean_p_z": sw_result.mean_p_z,
                    "median_p_z": sw_result.median_p_z,
                    "min_p_z": min(p_z_values),
                    "std_p_z": statistics.stdev(p_z_values) if len(p_z_values) > 1 else 0.0,
                    # Perplexity statistics
                    "mean_perplexity": statistics.mean(ppl_values) if ppl_values else 0.0,
                    "min_perplexity": min(ppl_values) if ppl_values else 0.0,
                    # Most memorized window details
                    "best_window_idx": best_window.window_idx,
                    "best_window_p_z": best_window.p_z,
                    "best_window_seed": best_window.seed,
                    "best_window_target": best_window.target,
                    "best_window_start_char": best_window.window_start_char,
                    "best_window_end_char": best_window.window_end_char,
                    # Generation for best window (empty when reproduce=False)
                    "best_window_generated": None,
                    "best_window_beams": None,
                    "best_window_bleu_1": None,
                    "best_window_rouge_l": None,
                    "best_window_exact_match": None,
                    "best_window_any_exact_match": None,
                    "best_window_max_bleu_1": None,
                    "best_window_max_rouge_l": None,
                    # Average generation metrics (empty when reproduce=False)
                    "avg_max_bleu_1": None,
                    "avg_max_rouge_l": None,
                    "avg_any_exact_match": None,
                    # Eval config
                    "eval_model": model_path,
                    "eval_threshold": threshold,
                    "window_size": window_size,
                    "stride": stride,
                    "decode_strategy": decode_strategy,
                    "beam_width": beam_width if decode_strategy == "beam" else None,
                    "length_penalty": length_penalty if decode_strategy == "beam" else None,
                    "reproduce_all_windows": reproduce_all,
                    # All windows (array)
                    "windows": windows_array,
                }
                # Copy metadata columns from source item
                copy_metadata(item, record)
                all_records.append(record)

                total_windows += sw_result.total_windows
                total_memorized += sw_result.memorized_windows

        # Skip the per-item loop when not reproducing
        update_progress(len(items), len(items))

    else:
        # ORIGINAL PATH: Per-item processing with generation
        item_iter = items if part_id else tqdm(items, desc="Processing items")
        for item_idx, item in enumerate(item_iter):
            update_progress(item_idx, len(items))
            text = item[text_col]
            if not text or len(text.strip()) < 10:
                continue

            item_name = get_item_name(item)

            # Run sliding window
            sw_result = evaluator.sliding_window(
                text=text,
                name=item_name,
                show_progress=False,
            )

            if not sw_result.windows:
                continue

            # Find most memorized window
            best_window = max(sw_result.windows, key=lambda w: w.p_z)
            best_window_idx = best_window.window_idx

            # Generate for windows
            gen_results_map = {}  # window_idx -> (generated_text, bleu_1, rouge_l, exact_match)

            if config.do_generate:
                def compute_beam_metrics(decode_output, target):
                    """Compute metrics for ALL beams, return structured result."""
                    beams = []
                    for i, seq in enumerate(decode_output.sequences):
                        beam_text = seq.generated_text
                        beam_exact = evaluator._check_exact_match(beam_text, target)
                        beam_bleu, _, _, beam_rouge = evaluator._compute_similarity(beam_text, target)
                        beams.append({
                            "beam_idx": i,
                            "text": beam_text,
                            "bleu_1": beam_bleu,
                            "rouge_l": beam_rouge,
                            "exact_match": beam_exact,
                            "score": seq.score,
                            "log_prob": seq.total_log_prob,
                        })

                    # Summary metrics
                    any_exact = any(b["exact_match"] for b in beams)
                    best_bleu_idx = max(range(len(beams)), key=lambda i: beams[i]["bleu_1"] or 0) if beams else 0
                    best_rouge_idx = max(range(len(beams)), key=lambda i: beams[i]["rouge_l"] or 0) if beams else 0

                    return {
                        "beams": beams,
                        "best_beam_text": beams[0]["text"] if beams else "",
                        "best_beam_bleu_1": beams[0]["bleu_1"] if beams else None,
                        "best_beam_rouge_l": beams[0]["rouge_l"] if beams else None,
                        "best_beam_exact_match": beams[0]["exact_match"] if beams else None,
                        "any_exact_match": any_exact,
                        "max_bleu_1": beams[best_bleu_idx]["bleu_1"] if beams else None,
                        "max_rouge_l": beams[best_rouge_idx]["rouge_l"] if beams else None,
                        "best_match_beam_idx": best_bleu_idx,
                    }

                if reproduce_all:
                    # Batch generate for ALL windows using strategy
                    window_seeds = [w.seed for w in sw_result.windows]
                    decode_outputs = evaluator._get_backend().generate_batch(
                        window_seeds,
                        strategy=strategy,
                        max_new_tokens=window_size,
                        capture_top_k=capture_top_k,
                    )
                    for w, decode_output in zip(sw_result.windows, decode_outputs):
                        gen_results_map[w.window_idx] = compute_beam_metrics(decode_output, w.target)
                else:
                    # Generate only for best window using strategy
                    decode_output = evaluator._get_backend().generate(
                        best_window.seed,
                        strategy=strategy,
                        max_new_tokens=window_size,
                        capture_top_k=capture_top_k,
                    )
                    gen_results_map[best_window_idx] = compute_beam_metrics(decode_output, best_window.target)

            # Get best window generation results
            best_window_gen = None
            if best_window_idx in gen_results_map:
                best_window_gen = gen_results_map[best_window_idx]

            # Compute average generation metrics across all windows
            # Use max_bleu/max_rouge (best across all beams) for aggregation
            avg_bleu_1 = None
            avg_rouge_l = None
            avg_any_exact = None
            if gen_results_map:
                all_bleu = [v["max_bleu_1"] for v in gen_results_map.values() if v["max_bleu_1"] is not None]
                all_rouge = [v["max_rouge_l"] for v in gen_results_map.values() if v["max_rouge_l"] is not None]
                all_any_exact = [v["any_exact_match"] for v in gen_results_map.values()]
                if all_bleu:
                    avg_bleu_1 = sum(all_bleu) / len(all_bleu)
                if all_rouge:
                    avg_rouge_l = sum(all_rouge) / len(all_rouge)
                if all_any_exact:
                    avg_any_exact = sum(all_any_exact) / len(all_any_exact)

            # Aggregate metrics
            p_z_values = [w.p_z for w in sw_result.windows]
            ppl_values = [w.perplexity for w in sw_result.windows if w.perplexity > 0]

            # Store all windows as array (with generation if available)
            windows_array = []
            for w in sw_result.windows:
                window_data = {
                    "idx": w.window_idx,
                    "start_char": w.window_start_char,
                    "end_char": w.window_end_char,
                    "seed": w.seed,
                    "target": w.target,
                    "p_z": w.p_z,
                    "log_prob": w.log_prob,
                    "perplexity": w.perplexity,
                    "is_memorized": w.is_memorized,
                    "rank_1_fraction": w.rank_1_fraction,
                }
                # Add generation results if available
                if w.window_idx in gen_results_map:
                    gen_data = gen_results_map[w.window_idx]
                    max_rouge = gen_data["max_rouge_l"] or 0
                    window_data["generated"] = gen_data["best_beam_text"]
                    window_data["beams"] = gen_data["beams"]  # All beams with metrics
                    window_data["best_beam_bleu_1"] = gen_data["best_beam_bleu_1"]
                    window_data["best_beam_rouge_l"] = gen_data["best_beam_rouge_l"]
                    window_data["best_beam_exact_match"] = gen_data["best_beam_exact_match"]
                    window_data["max_bleu_1"] = gen_data["max_bleu_1"]
                    window_data["max_rouge_l"] = max_rouge
                    window_data["any_exact_match"] = gen_data["any_exact_match"]
                    window_data["best_match_beam_idx"] = gen_data["best_match_beam_idx"]
                    window_data["is_reproduced"] = max_rouge >= config.reproduction_threshold
                windows_array.append(window_data)

            record = {
                # Evaluation metrics
                "text_length_chars": len(text),
                "num_windows": sw_result.total_windows,
                "memorized_windows": sw_result.memorized_windows,
                "memorized_fraction": sw_result.memorized_windows / sw_result.total_windows,
                "reproduced_windows": sum(1 for w in windows_array if w.get("is_reproduced", False)),
                "reproduced_fraction": sum(1 for w in windows_array if w.get("is_reproduced", False)) / sw_result.total_windows if sw_result.total_windows > 0 else 0,
                "avg_rouge_l": sum(w.get("max_rouge_l", 0) or 0 for w in windows_array) / len(windows_array) if windows_array else 0,
                "max_rouge_l": max((w.get("max_rouge_l", 0) or 0 for w in windows_array), default=0),
                "coverage": sw_result.coverage,
                # p_z statistics
                "max_p_z": sw_result.max_p_z,
                "mean_p_z": sw_result.mean_p_z,
                "median_p_z": sw_result.median_p_z,
                "min_p_z": min(p_z_values),
                "std_p_z": statistics.stdev(p_z_values) if len(p_z_values) > 1 else 0.0,
                # Perplexity statistics
                "mean_perplexity": statistics.mean(ppl_values) if ppl_values else 0.0,
                "min_perplexity": min(ppl_values) if ppl_values else 0.0,
                # Most memorized window details
                "best_window_idx": best_window.window_idx,
                "best_window_p_z": best_window.p_z,
                "best_window_seed": best_window.seed,
                "best_window_target": best_window.target,
                "best_window_start_char": best_window.window_start_char,
                "best_window_end_char": best_window.window_end_char,
                # Generation for best window
                "best_window_generated": best_window_gen["best_beam_text"] if best_window_gen else None,
                "best_window_beams": best_window_gen["beams"] if best_window_gen else None,
                "best_window_bleu_1": best_window_gen["best_beam_bleu_1"] if best_window_gen else None,
                "best_window_rouge_l": best_window_gen["best_beam_rouge_l"] if best_window_gen else None,
                "best_window_exact_match": best_window_gen["best_beam_exact_match"] if best_window_gen else None,
                "best_window_any_exact_match": best_window_gen["any_exact_match"] if best_window_gen else None,
                "best_window_max_bleu_1": best_window_gen["max_bleu_1"] if best_window_gen else None,
                "best_window_max_rouge_l": best_window_gen["max_rouge_l"] if best_window_gen else None,
                # Average generation metrics (across all windows when reproduce_all=True)
                # These use max across beams for each window, then average across windows
                "avg_max_bleu_1": avg_bleu_1,
                "avg_max_rouge_l": avg_rouge_l,
                "avg_any_exact_match": avg_any_exact,
                # Eval config
                "eval_model": model_path,
                "eval_threshold": threshold,
                "window_size": window_size,
                "stride": stride,
                "decode_strategy": decode_strategy,
                "beam_width": beam_width if decode_strategy == "beam" else None,
                "length_penalty": length_penalty if decode_strategy == "beam" else None,
                "reproduce_all_windows": reproduce_all,
                # All windows (array)
                "windows": windows_array,
            }
            # Copy metadata columns from source item
            copy_metadata(item, record)
            all_records.append(record)

            total_windows += sw_result.total_windows
            total_memorized += sw_result.memorized_windows
            total_reproduced += record["reproduced_windows"]

        # Final progress update for else block
        update_progress(len(items), len(items))

    # Display summary (skip for workers)
    if not part_id:
        console.print()
        console.print("[bold]Summary:[/bold]")
        console.print(f"  Items processed: {len(all_records)}")
        console.print(f"  Total windows: {total_windows}")

        console.print()
        console.print("[bold]  Memorization[/bold] [dim](p_z > τ_min)[/dim]")
        if total_windows > 0:
            console.print(
                f"    Memorized windows: {total_memorized}/{total_windows} "
                f"({total_memorized/total_windows:.1%})"
            )
        if all_records:
            items_with_memorization = sum(1 for r in all_records if r["memorized_windows"] > 0)
            console.print(
                f"    Items with memorization: {items_with_memorization}/{len(all_records)} "
                f"({items_with_memorization/len(all_records):.1%})"
            )

        console.print()
        console.print("[bold]  Reproduction[/bold] [dim](ROUGE-L > {:.0%})[/dim]".format(config.reproduction_threshold))
        if total_windows > 0:
            console.print(
                f"    Reproduced windows: {total_reproduced}/{total_windows} "
                f"({total_reproduced/total_windows:.1%})"
            )
        if all_records:
            items_with_reproduction = sum(1 for r in all_records if r["reproduced_windows"] > 0)
            console.print(
                f"    Items with reproduction: {items_with_reproduction}/{len(all_records)} "
                f"({items_with_reproduction/len(all_records):.1%})"
            )

        if all_records:
            max_p_z_values = [r["max_p_z"] for r in all_records]
            max_rouge_values = [r["max_rouge_l"] for r in all_records]
            console.print()
            console.print(f"  Highest p_z: {max(max_p_z_values):.2e}")
            console.print(f"  Highest ROUGE-L: {max(max_rouge_values):.2f}")

            if config.do_generate:
                # Show average metrics (preferred aggregation)
                avg_bleu_values = [r["avg_max_bleu_1"] for r in all_records if r["avg_max_bleu_1"] is not None]
                avg_rouge_values = [r["avg_max_rouge_l"] for r in all_records if r["avg_max_rouge_l"] is not None]
                avg_exact_values = [r["avg_any_exact_match"] for r in all_records if r["avg_any_exact_match"] is not None]

                if avg_bleu_values:
                    console.print(f"  Mean max BLEU-1: {sum(avg_bleu_values)/len(avg_bleu_values):.3f}")
                    console.print(f"  Mean max ROUGE-L: {sum(avg_rouge_values)/len(avg_rouge_values):.3f}")
                    console.print(f"  Mean any exact match: {sum(avg_exact_values)/len(avg_exact_values):.3f}")
                else:
                    # Fallback to best window metrics if reproduce_all wasn't used
                    bleu_values = [r["best_window_max_bleu_1"] for r in all_records if r["best_window_max_bleu_1"] is not None]
                    any_exact = [r["best_window_any_exact_match"] for r in all_records if r["best_window_any_exact_match"] is not None]
                    if bleu_values:
                        console.print(f"  Mean max BLEU-1 (best window only): {sum(bleu_values)/len(bleu_values):.3f}")
                    if any_exact:
                        console.print(f"  Any exact match rate (best window): {sum(any_exact)/len(any_exact):.3f}")

        console.print()
        console.print("[bold]Saving results...[/bold]")

    # Use part_id for parallel runs to avoid overwriting
    if part_id:
        output_path = Path(output_dir) / hf_config / part_id
    else:
        output_path = Path(output_dir) / hf_config
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config_data = {
        "source_dataset": hf_dataset,
        "source_config": hf_config,
        "model": model,
        "backend": backend,
        "window_size": window_size,
        "stride": stride,
        "threshold": threshold,
        "decode_strategy": decode_strategy,
        "beam_width": beam_width if decode_strategy == "beam" else None,
        "length_penalty": length_penalty if decode_strategy == "beam" else None,
        "reproduce_all_windows": reproduce_all,
        "num_items": len(all_records),
        "total_windows": total_windows,
        "memorized_windows": total_memorized,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    # Save as parquet/HF Dataset
    if all_records:
        results_ds = Dataset.from_list(all_records)
        parquet_path = output_path / "results.parquet"
        results_ds.to_parquet(str(parquet_path))
        console.print(f"  Saved {len(all_records)} results to {parquet_path}")

        # Push to HuggingFace (default behavior, skip with --no-push)
        if not no_push and not part_id:
            repo_name = hf_push or generate_hf_repo_name(hf_org, hf_dataset, model_clean)
            console.print()
            console.print(f"[bold]Pushing to HuggingFace: {repo_name}[/bold]")
            results_ds.push_to_hub(repo_name, split="train")
            console.print(f"[green]Pushed to https://huggingface.co/datasets/{repo_name}[/green]")
        elif part_id:
            console.print("[dim]Skipping HF push (parallel worker)[/dim]")
    else:
        console.print("[yellow]No results to save (no valid songs)[/yellow]")

    console.print()
    console.print("[green bold]Done![/green bold]")


@app.command()
def probe(
    text: str = typer.Argument(..., help="Text to scan with sliding window"),
    model: str = typer.Option(
        "Qwen/Qwen2.5-0.5B-Instruct", "--model", "-m", help="Model name"
    ),
    backend: str = typer.Option(
        "sglang", "--backend", "-b", help="Backend: sglang or huggingface"
    ),
    window_size: int = typer.Option(
        50, "--window-size", "-w", help="Window size in tokens"
    ),
    stride: int = typer.Option(25, "--stride", help="Stride in tokens"),
):
    """Quick sliding window probe on a text string."""
    config = MemorizationConfig(
        do_generate=False,
        window_prefix_len=window_size,
        window_suffix_len=window_size,
        window_stride=stride,
    )

    model_path, _ = resolve_model(model)
    evaluator = MemorizationEvaluator(model=model_path, backend=backend, config=config)
    result = evaluator.sliding_window(text, name="probe", show_progress=True)

    console.print()
    console.print(f"[bold]Windows:[/bold] {result.total_windows}")
    console.print(f"[bold]Memorized:[/bold] {result.memorized_windows}")
    console.print(f"[bold]Coverage:[/bold] {result.coverage:.1%}")
    console.print(f"[bold]Max p_z:[/bold] {result.max_p_z:.6f}")
    console.print(f"[bold]Mean p_z:[/bold] {result.mean_p_z:.6f}")

    if result.memorized_windows > 0:
        console.print()
        console.print("[bold]Memorized windows:[/bold]")
        for w in result.windows:
            if w.is_memorized:
                console.print(f"  Window {w.window_idx}: p_z={w.p_z:.6f}")
                console.print(f"    prefix: {w.seed[:60]}...")
                console.print(f"    suffix: {w.target[:60]}...")


@app.command("eval-all")
def eval_all(
    models: str = typer.Option(
        None, "--models", "-m",
        help="Comma-separated model aliases to run (default: all)"
    ),
    num_gpus: int = typer.Option(
        1, "--num-gpus", "-g", help="Number of GPUs per model"
    ),
    gpu_ids: str = typer.Option(
        None, "--gpu-ids", help="GPU IDs to use (e.g. '0,1,2,3')"
    ),
    skip_existing: bool = typer.Option(
        True, "--skip-existing/--no-skip-existing",
        help="Skip models that already have results on HuggingFace"
    ),
):
    """Run eval for all models sequentially."""
    import subprocess
    import sys

    # Determine which models to run
    if models:
        model_list = [m.strip() for m in models.split(",")]
        invalid = [m for m in model_list if m not in MODEL_ALIASES]
        if invalid:
            console.print(f"[red]Unknown models: {invalid}[/red]")
            console.print(f"Available: {list(MODEL_ALIASES.keys())}")
            raise typer.Exit(1)
    else:
        model_list = list(MODEL_ALIASES.keys())

    console.print(f"[bold]Running eval for {len(model_list)} models:[/bold]")
    for m in model_list:
        console.print(f"  - {m} ({MODEL_ALIASES[m][0]})")
    console.print()

    # Check for existing results if skip_existing
    if skip_existing:
        from huggingface_hub import list_datasets
        existing_repos = {ds.id for ds in list_datasets(author="unlearning-cleanslate")}
        to_skip = []
        for m in model_list:
            _, clean_name = MODEL_ALIASES[m]
            repo = f"unlearning-cleanslate/eval_cleanslate_dataset_{clean_name}"
            if repo in existing_repos:
                to_skip.append(m)
        if to_skip:
            console.print(f"[yellow]Skipping (already exist): {to_skip}[/yellow]")
            model_list = [m for m in model_list if m not in to_skip]
            console.print()

    if not model_list:
        console.print("[green]All models already evaluated![/green]")
        return

    # Run eval for each model
    results = {}
    for i, model in enumerate(model_list, 1):
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print(f"[bold blue]Model {i}/{len(model_list)}: {model}[/bold blue]")
        console.print(f"[bold blue]{'='*60}[/bold blue]\n")

        cmd = [sys.executable, "-m", "cleanslate.cli", "eval", "-m", model]
        cmd.extend(["--num-gpus", str(num_gpus)])
        if gpu_ids:
            cmd.extend(["--gpu-ids", gpu_ids])

        try:
            subprocess.run(cmd, check=True)
            results[model] = "success"
            console.print(f"\n[green]✓ {model} completed[/green]")
        except subprocess.CalledProcessError as e:
            results[model] = f"failed (exit {e.returncode})"
            console.print(f"\n[red]✗ {model} failed[/red]")

    # Summary
    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print("[bold]Summary:[/bold]")
    for model, status in results.items():
        icon = "✓" if status == "success" else "✗"
        color = "green" if status == "success" else "red"
        console.print(f"  [{color}]{icon} {model}: {status}[/{color}]")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
