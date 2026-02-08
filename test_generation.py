#!/usr/bin/env python
"""Test the new generation mode."""

from cleanslate.generation_experiment import GenerationExperiment
from cleanslate.samples import SHAKESPEARE_SAMPLES

print("=" * 80)
print("Testing Generation Mode (No Metrics)")
print("=" * 80)
print()

# Test with first 2 Shakespeare samples
test_samples = SHAKESPEARE_SAMPLES[:2]

exp = GenerationExperiment(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    name="test_generation",
    backend="huggingface",
)

# Generate 3 outputs per seed
generations = exp.run_generation(
    samples=test_samples,
    num_generations=3,
    max_tokens=50,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    output_path="test_generations",
    save_format="parquet",
)

print()
print("=" * 80)
print("Sample Generation Data:")
print("=" * 80)
gen = generations[0]
print(f"Sample: {gen.sample_name}")
print(f"Seed: {gen.seed[:50]}...")
print(f"Generated: {gen.generated_text[:100]}...")
print(f"Tokens: {gen.num_tokens}")
print(f"Mean logprob: {gen.mean_logprob:.3f}")
print(f"Finish reason: {gen.finish_reason}")
print()
print("✅ Generation mode works!")
