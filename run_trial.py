#!/usr/bin/env python
"""Comprehensive test of CleanSlate with HuggingFace backend."""

from cleanslate import Experiment
from cleanslate.samples import SHAKESPEARE_SAMPLES

print("=" * 80)
print("CleanSlate - Production-Ready Test")
print("Using HuggingFace Backend (Recommended)")
print("=" * 80)
print()

# Initialize experiment
exp = Experiment(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    name="shakespeare_memorization",
    backend="huggingface",
)

print("Running experiment on all Shakespeare samples...")
print("-" * 80)

# Run on all samples
results = exp.run(SHAKESPEARE_SAMPLES)

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(results.summary())

# Save results
output_file = "results/shakespeare_test.json"
results.save(output_file)
print()
print(f"✅ Results saved to: {output_file}")
print()
print("=" * 80)
print("Trial run complete! The vLLM backend has been implemented,")
print("but HuggingFace backend is recommended for this use case.")
print("=" * 80)
