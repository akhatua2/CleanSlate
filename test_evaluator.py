#!/usr/bin/env python3
"""Test script for the unified MemorizationEvaluator."""

if __name__ == "__main__":
    from cleanslate.evaluator import (
        MemorizationConfig,
        MemorizationEvaluator,
        save_results,
    )

    # Test configuration
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    BACKEND = "sglang"

    # Sample data
    samples = [
        {
            "name": "Shakespeare - To be or not to be",
            "seed": "To be or not to be,",
            "target": " that is the question",
        },
        {
            "name": "Opening - It was the best",
            "seed": "It was the best of times,",
            "target": " it was the worst of times",
        },
    ]

    print("=" * 60)
    print("Testing MemorizationEvaluator")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Backend: {BACKEND}")
    print(f"Samples: {len(samples)}")
    print()

    # Create config
    config = MemorizationConfig(
        compute_top_k=40,
        tau_min=0.001,
        do_generate=True,
        gen_temperature=0.0,
        gen_top_k=40,
        gen_max_new_tokens=20,
        capture_top_k=5,
    )

    # Create evaluator
    print("Loading model...")
    evaluator = MemorizationEvaluator(
        model=MODEL,
        backend=BACKEND,
        config=config,
    )

    # Test single evaluation
    print("\n" + "=" * 60)
    print("Test 1: Single evaluation")
    print("=" * 60)

    result = evaluator.evaluate_single(
        seed=samples[0]["seed"],
        target=samples[0]["target"],
        sample_name=samples[0]["name"],
    )

    print(f"Sample: {result.sample_name}")
    print(f"Seed: {result.seed!r}")
    print(f"Target: {result.target!r}")
    print()
    print("Probability Metrics:")
    print(f"  p_z: {result.p_z:.6f}")
    print(f"  log_prob: {result.log_prob:.4f}")
    print(f"  perplexity: {result.perplexity:.2f}")
    print(f"  is_memorized: {result.is_memorized}")
    print(f"  prompts_to_extract: {result.prompts_to_extract:.1f}")
    print(f"  rank_1_fraction: {result.rank_1_fraction:.2%}")
    print()
    print("Generation Output:")
    print(f"  generated: {result.generated_text!r}")
    print(f"  exact_match: {result.exact_match}")
    print(f"  bleu_1: {result.bleu_1:.3f}" if result.bleu_1 is not None else "  bleu_1: N/A")
    print(f"  rouge_l: {result.rouge_l:.3f}" if result.rouge_l is not None else "  rouge_l: N/A")
    print()
    print(f"Timing: compute={result.compute_time_ms:.1f}ms, generate={result.generate_time_ms:.1f}ms")

    # Test batch evaluation
    print("\n" + "=" * 60)
    print("Test 2: Batch evaluation")
    print("=" * 60)

    results = evaluator.evaluate_batch(samples, show_progress=True)

    print(f"\nProcessed {len(results)} samples")
    for r in results:
        bleu_str = f"{r.bleu_1:.3f}" if r.bleu_1 is not None else "N/A"
        print(f"  {r.sample_name}: p_z={r.p_z:.6f}, memorized={r.is_memorized}, bleu_1={bleu_str}")

    # Compute summary
    summary = evaluator.compute_summary(results, dataset_name="test")
    print()
    print("Summary:")
    print(f"  Memorized: {summary.memorized_count}/{summary.num_samples}")
    print(f"  Mean p_z: {summary.p_z_mean:.6f}")
    print(f"  Max p_z: {summary.p_z_max:.6f}")
    if summary.bleu_1_mean is not None:
        print(f"  Mean BLEU-1: {summary.bleu_1_mean:.3f}")

    # Test sliding window (if text is long enough)
    print("\n" + "=" * 60)
    print("Test 3: Sliding window")
    print("=" * 60)

    # Reuse the same evaluator but update config for smaller windows
    evaluator.config.window_prefix_len = 10
    evaluator.config.window_suffix_len = 10
    evaluator.config.window_stride = 5
    evaluator.config.do_generate = False

    test_text = "To be or not to be, that is the question. Whether tis nobler in the mind to suffer."
    print(f"Text: {test_text!r}")
    print(f"Window: {evaluator.config.window_prefix_len} prefix + {evaluator.config.window_suffix_len} suffix, stride {evaluator.config.window_stride}")

    sw_result = evaluator.sliding_window(test_text, name="hamlet_excerpt", show_progress=True)

    print()
    print(f"Total windows: {sw_result.total_windows}")
    print(f"Memorized windows: {sw_result.memorized_windows}")
    print(f"Coverage: {sw_result.coverage:.1%}")
    print(f"Max p_z: {sw_result.max_p_z:.6f}")
    print(f"Mean p_z: {sw_result.mean_p_z:.6f}")

    # Save test results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    # Reset config for saving (use original config)
    save_config = MemorizationConfig(
        compute_top_k=40,
        tau_min=0.001,
        do_generate=True,
        gen_temperature=0.0,
        gen_top_k=40,
        gen_max_new_tokens=20,
        capture_top_k=5,
    )

    save_results(
        results=results,
        output_dir="results/test_eval",
        config=save_config,
        model_name=MODEL,
        dataset_name="test",
        summary=summary,
    )

    print("\nDone!")
