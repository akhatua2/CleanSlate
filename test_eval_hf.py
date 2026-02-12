#!/usr/bin/env python3
"""Test sliding window evaluation with HuggingFace dataset."""

import os
os.environ["HF_HOME"] = "/dfs/project/kgrlm/akhatua/.cache/huggingface"

if __name__ == "__main__":
    from datasets import Dataset, load_dataset
    from cleanslate.evaluator import MemorizationConfig, MemorizationEvaluator

    # Test configuration
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    BACKEND = "sglang"
    HF_DATASET = "unlearning-cleanslate/generated_songs"
    HF_CONFIG = "direct__first_line"
    LIMIT = 2  # Just test with 2 songs
    WINDOW_SIZE = 50  # tokens
    STRIDE = 25  # tokens

    print("=" * 60)
    print("Testing Sliding Window Evaluation")
    print("=" * 60)
    print(f"Dataset: {HF_DATASET} / {HF_CONFIG}")
    print(f"Model: {MODEL}")
    print(f"Window: {WINDOW_SIZE} tokens prefix + {WINDOW_SIZE} tokens suffix")
    print(f"Stride: {STRIDE} tokens")
    print(f"Limit: {LIMIT} songs")
    print()

    # Load dataset (streaming to avoid disk quota)
    print("Loading dataset (streaming mode)...")
    ds_stream = load_dataset(HF_DATASET, HF_CONFIG, split="train", streaming=True)

    # Take first N songs
    songs = []
    for i, row in enumerate(ds_stream):
        if i >= LIMIT:
            break
        songs.append(row)
        print(f"  Loaded: {row.get('song_title', 'Unknown')} ({row.get('song_year', '?')})")

    print(f"\nLoaded {len(songs)} songs")

    # Show first song's lyrics
    print()
    print("Sample lyrics (first 200 chars):")
    print(f"  {songs[0]['reference_target'][:200]}...")
    print()

    # Create config
    config = MemorizationConfig(
        compute_top_k=40,
        tau_min=0.001,
        do_generate=True,
        gen_temperature=0.0,
        gen_top_k=40,
        gen_max_new_tokens=WINDOW_SIZE,
        capture_top_k=5,
        window_prefix_len=WINDOW_SIZE,
        window_suffix_len=WINDOW_SIZE,
        window_stride=STRIDE,
    )

    # Create evaluator
    print("Loading model...")
    evaluator = MemorizationEvaluator(
        model=MODEL,
        backend=BACKEND,
        config=config,
    )

    # Process each song
    print("\nRunning sliding window evaluation...")
    all_records = []

    for song in songs:
        text = song["reference_target"]
        song_title = song.get("song_title", "Unknown")
        print(f"\n  Processing: {song_title}")

        # Run sliding window (compute only)
        sw_result = evaluator.sliding_window(
            text=text,
            name=song_title,
            show_progress=True,
        )

        print(f"    Windows: {sw_result.total_windows}")
        print(f"    Memorized: {sw_result.memorized_windows}")
        print(f"    Max p_z: {sw_result.max_p_z:.6f}")

        # For each window, also generate
        for w in sw_result.windows[:3]:  # Just first 3 windows per song for testing
            # Generate
            gen_result = evaluator._get_backend().generate(
                w.seed,
                max_new_tokens=WINDOW_SIZE,
                temperature=0.0,
                top_k=40,
                do_sample=False,
                capture_top_k=5,
            )

            # Compare
            exact_match = evaluator._check_exact_match(gen_result.generated_text, w.target)
            bleu_1, _, _, rouge_l = evaluator._compute_similarity(
                gen_result.generated_text, w.target
            )

            record = {
                "song_title": song_title,
                "window_idx": w.window_idx,
                "seed": w.seed[:50] + "...",
                "target": w.target[:50] + "...",
                "p_z": w.p_z,
                "is_memorized": w.is_memorized,
                "generated": gen_result.generated_text[:50] + "..." if gen_result.generated_text else None,
                "bleu_1": bleu_1,
                "rouge_l": rouge_l,
                "exact_match": exact_match,
            }
            all_records.append(record)

    # Show results
    print("\n" + "=" * 60)
    print("Results (first 3 windows per song)")
    print("=" * 60)

    for r in all_records:
        print(f"\n{r['song_title']} - Window {r['window_idx']}")
        print(f"  p_z: {r['p_z']:.6f} {'(MEMORIZED)' if r['is_memorized'] else ''}")
        print(f"  seed: {r['seed']}")
        print(f"  target: {r['target']}")
        print(f"  generated: {r['generated']}")
        bleu_str = f"{r['bleu_1']:.3f}" if r['bleu_1'] is not None else "N/A"
        rouge_str = f"{r['rouge_l']:.3f}" if r['rouge_l'] is not None else "N/A"
        print(f"  bleu_1: {bleu_str}, rouge_l: {rouge_str}, exact: {r['exact_match']}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    p_z_values = [r["p_z"] for r in all_records]
    memorized = sum(1 for r in all_records if r["is_memorized"])
    print(f"  Total windows evaluated: {len(all_records)}")
    print(f"  Memorized: {memorized}/{len(all_records)}")
    print(f"  Mean p_z: {sum(p_z_values)/len(p_z_values):.6f}")
    print(f"  Max p_z: {max(p_z_values):.6f}")

    # Test HF Dataset format
    print("\n" + "=" * 60)
    print("Testing HF Dataset format")
    print("=" * 60)
    results_ds = Dataset.from_list(all_records)
    print(f"Created HF Dataset with {len(results_ds)} rows")
    print(f"Columns: {results_ds.column_names}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
