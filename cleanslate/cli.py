"""CleanSlate CLI - Sliding window memorization evaluation."""

try:
    import typer
    from rich.console import Console
except ImportError:
    raise ImportError(
        "CLI dependencies not installed. Install with: pip install cleanslate[cli]"
    )

from pathlib import Path

from datasets import Dataset, load_dataset

from cleanslate.evaluator import (
    MemorizationConfig,
    MemorizationEvaluator,
)
from cleanslate.decoding import create_strategy
from cleanslate.cli_utils import run_parallel_eval, generate_hf_repo_name

app = typer.Typer(
    name="cleanslate",
    help="Sliding window memorization evaluation: compute p_z + generate + compare.",
)
console = Console()


@app.command()
def eval(
    # Dataset source
    hf_dataset: str = typer.Option(
        ...,
        "--hf-dataset",
        "-d",
        help="HuggingFace dataset ID (e.g. unlearning-cleanslate/generated_songs)",
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
        help="Column containing the text to evaluate (lyrics)",
    ),
    # Sampling
    limit: int | None = typer.Option(
        None, "--limit", "-n", help="Limit number of songs to process"
    ),
    shuffle: bool = typer.Option(
        False, "--shuffle/--no-shuffle", help="Shuffle before selecting"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for shuffle"),
    offset: int = typer.Option(
        0, "--offset", help="Skip first N songs (for parallel splits)"
    ),
    part_id: str | None = typer.Option(
        None, "--part-id", help="Part ID for parallel runs (e.g., 'part_0'). Outputs to separate dir, skips HF push."
    ),
    indices_file: str | None = typer.Option(
        None, "--indices-file", help="JSON file with song indices to process (for balanced parallel splits)"
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
    no_generate: bool = typer.Option(
        False, "--no-generate", help="Skip generation, only compute p_z"
    ),
    generate_all: bool = typer.Option(
        True, "--generate-all/--generate-best-only", help="Generate for all windows (default) or only best p_z window"
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
        256, "--max-batch-prompts", help="Max prompts per SGLang call (reduce if OOM)"
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
        help="Push results to HuggingFace repo (e.g. unlearning-cleanslate/eval_results)",
    ),
    hf_push_auto: bool = typer.Option(
        False,
        "--hf-push-auto",
        help="Auto-generate HF repo name: {org}/eval__{dataset}__{model}__{strategy}",
    ),
    hf_push_org: str = typer.Option(
        "unlearning-cleanslate",
        "--hf-push-org",
        help="HuggingFace org for --hf-push-auto (default: unlearning-cleanslate)",
    ),
    # Parallelism
    num_gpus: int = typer.Option(
        1, "--num-gpus", "-g",
        help="Number of GPUs for data parallelism. Splits songs across GPUs."
    ),
    gpu_ids: str | None = typer.Option(
        None, "--gpu-ids",
        help="Comma-separated GPU IDs to use (e.g. '2,3,4,5'). Default: 0 to num_gpus-1"
    ),
):
    """
    Sliding window memorization evaluation on song lyrics.

    For each song:
    1. Slide a window across the lyrics text
    2. Each window: prefix (seed) + suffix (target)
    3. Compute p_z via teacher forcing
    4. Generate and compare with BLEU/ROUGE
    5. Aggregate results: one row per song with summary metrics

    Output includes max_p_z, mean_p_z, coverage, and most memorized window details.
    """
    import json
    import statistics

    # Handle multi-GPU parallelism (data parallel)
    if num_gpus > 1 and part_id is None:
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
            no_generate=no_generate,
            generate_all=generate_all,
            decode_strategy=decode_strategy,
            beam_width=beam_width,
            length_penalty=length_penalty,
            max_batch_prompts=max_batch_prompts,
            temperature=temperature,
            gen_top_k=gen_top_k,
            capture_top_k=capture_top_k,
            output_dir=output_dir,
            hf_push=hf_push,
            hf_push_auto=hf_push_auto,
            hf_push_org=hf_push_org,
        )
        return

    console.print(f"[bold]Dataset:[/bold] {hf_dataset} / {hf_config}")
    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Backend:[/bold] {backend}")
    console.print(f"[bold]Window:[/bold] {window_size} tokens prefix + {window_size} tokens suffix")
    console.print(f"[bold]Stride:[/bold] {stride} tokens")
    console.print(f"[bold]Generate:[/bold] {not no_generate}")
    if not no_generate:
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
        songs = [ds[i] for i in indices]
        console.print(f"  Loaded {len(songs)} songs from indices file")
    else:
        # Original streaming approach with offset/limit
        ds_stream = load_dataset(hf_dataset, hf_config, split="train", streaming=True)

        songs = []
        for i, row in enumerate(ds_stream):
            if i < offset:
                continue
            if limit and len(songs) >= limit:
                break
            songs.append(row)

        if shuffle:
            import random
            random.seed(seed)
            random.shuffle(songs)

        console.print(f"  Loaded {len(songs)} songs")
    console.print()

    # Validate text column
    if songs and text_col not in songs[0]:
        console.print(f"[red]Error: text column '{text_col}' not found[/red]")
        console.print(f"Available columns: {list(songs[0].keys())}")
        raise typer.Exit(1)

    # Create config
    config = MemorizationConfig(
        compute_top_k=40,
        tau_min=threshold,
        do_generate=not no_generate,
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
    evaluator = MemorizationEvaluator(
        model=model,
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

    song_iter = songs if part_id else tqdm(songs, desc="Processing songs")
    for song_idx, song in enumerate(song_iter):
        update_progress(song_idx, len(songs))
        text = song[text_col]
        if not text or len(text.strip()) < 10:
            continue

        # Get song metadata
        song_id = song.get("song_id", "")
        song_title = song.get("song_title", "")
        song_artists = song.get("song_artists", [])
        song_year = song.get("song_year")
        prompt_template = song.get("prompt_template", "")
        seed_mode = song.get("seed_mode", "")

        # Run sliding window
        sw_result = evaluator.sliding_window(
            text=text,
            name=song_title or song_id,
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

            if generate_all:
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
            # Song metadata
            "song_id": song_id,
            "song_title": song_title,
            "song_artists": song_artists,
            "song_year": song_year,
            "prompt_template": prompt_template,
            "seed_mode": seed_mode,
            "text_length_chars": len(text),
            # Aggregated metrics
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
            # Average generation metrics (across all windows when generate_all=True)
            # These use max across beams for each window, then average across windows
            "avg_max_bleu_1": avg_bleu_1,
            "avg_max_rouge_l": avg_rouge_l,
            "avg_any_exact_match": avg_any_exact,
            # Eval config
            "eval_model": model,
            "eval_threshold": threshold,
            "window_size": window_size,
            "stride": stride,
            "decode_strategy": decode_strategy,
            "beam_width": beam_width if decode_strategy == "beam" else None,
            "length_penalty": length_penalty if decode_strategy == "beam" else None,
            "generate_all_windows": generate_all,
            # All windows (array)
            "windows": windows_array,
        }
        all_records.append(record)

        total_windows += sw_result.total_windows
        total_memorized += sw_result.memorized_windows
        total_reproduced += record["reproduced_windows"]

    # Final progress update
    update_progress(len(songs), len(songs))

    # Display summary (skip for workers)
    if not part_id:
        console.print()
        console.print("[bold]Summary:[/bold]")
        console.print(f"  Songs processed: {len(all_records)}")
        console.print(f"  Total windows: {total_windows}")

        console.print()
        console.print("[bold]  Memorization[/bold] [dim](p_z > τ_min)[/dim]")
        if total_windows > 0:
            console.print(
                f"    Memorized windows: {total_memorized}/{total_windows} "
                f"({total_memorized/total_windows:.1%})"
            )
        if all_records:
            songs_with_memorization = sum(1 for r in all_records if r["memorized_windows"] > 0)
            console.print(
                f"    Songs with memorization: {songs_with_memorization}/{len(all_records)} "
                f"({songs_with_memorization/len(all_records):.1%})"
            )

        console.print()
        console.print("[bold]  Reproduction[/bold] [dim](ROUGE-L > {:.0%})[/dim]".format(config.reproduction_threshold))
        if total_windows > 0:
            console.print(
                f"    Reproduced windows: {total_reproduced}/{total_windows} "
                f"({total_reproduced/total_windows:.1%})"
            )
        if all_records:
            songs_with_reproduction = sum(1 for r in all_records if r["reproduced_windows"] > 0)
            console.print(
                f"    Songs with reproduction: {songs_with_reproduction}/{len(all_records)} "
                f"({songs_with_reproduction/len(all_records):.1%})"
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
                    # Fallback to best window metrics if generate_all wasn't used
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
        "generate_all_windows": generate_all,
        "num_songs": len(all_records),
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
        console.print(f"  Saved {len(all_records)} song results to {parquet_path}")

        # Push to HuggingFace (skip if part_id is set - merge script handles push)
        # Determine HF repo name (explicit or auto-generated)
        effective_hf_push = hf_push
        if hf_push_auto and not hf_push:
            effective_hf_push = generate_hf_repo_name(
                org=hf_push_org,
                dataset=hf_dataset,
                config=hf_config,
                model=model,
                decode_strategy=decode_strategy,
                beam_width=beam_width if decode_strategy == "beam" else None,
                length_penalty=length_penalty if decode_strategy == "beam" else None,
            )
            console.print(f"[dim]Auto-generated HF repo: {effective_hf_push}[/dim]")

        if effective_hf_push and not part_id:
            console.print()
            console.print(f"[bold]Pushing to HuggingFace: {effective_hf_push}[/bold]")

            results_ds.push_to_hub(
                effective_hf_push,
                config_name=hf_config,
                split="train",
            )
            console.print(
                f"[green]Pushed to https://huggingface.co/datasets/{effective_hf_push} "
                f"(config: {hf_config})[/green]"
            )
        elif effective_hf_push and part_id:
            console.print(
                f"[yellow]Skipping HF push (part_id={part_id}). Run merge script to combine and push.[/yellow]"
            )
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

    evaluator = MemorizationEvaluator(model=model, backend=backend, config=config)
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


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
