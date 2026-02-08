"""Experiment harness for running memorization probes."""

import logging
from datetime import datetime
from typing import Literal

from cleanslate.backends.base import InferenceBackend
from cleanslate.backends.huggingface import HuggingFaceBackend
from cleanslate.metrics.bleu import BLEU1
from cleanslate.metrics.rouge import ROUGE1, ROUGEL
from cleanslate.results import ExperimentResults, SampleResult

logger = logging.getLogger(__name__)


BackendType = Literal["huggingface", "vllm", "sglang"]


class Experiment:
    """
    Harness for running sequence likelihood experiments.

    Example:
        exp = Experiment(model="Qwen/Qwen3-0.6B", name="shakespeare_memorization")
        results = exp.run(SHAKESPEARE_SAMPLES)
        print(results.summary())
        results.save("results/shakespeare.json")
    """

    def __init__(
        self,
        model: str,
        name: str = "experiment",
        backend: BackendType = "huggingface",
        device: str | None = None,
    ):
        """
        Initialize experiment.

        Args:
            model: HuggingFace model name or path
            name: Experiment name for tracking
            backend: "huggingface" (CPU/GPU), "vllm" (GPU only), or "sglang" (GPU only)
            device: Device override. None for auto-detection.
        """
        self.model_name = model
        self.experiment_name = name
        self._backend_type = backend
        self._device = device
        self._estimator: InferenceBackend | None = None

    def _get_estimator(self) -> InferenceBackend:
        """Lazy-load the model."""
        if self._estimator is None:
            if self._backend_type == "huggingface":
                self._estimator = HuggingFaceBackend(
                    self.model_name, device=self._device
                )
            elif self._backend_type == "vllm":
                from cleanslate.backends.vllm_backend import VLLMBackend

                self._estimator = VLLMBackend(self.model_name)
            elif self._backend_type == "sglang":
                from .backends.sglang_backend import SGLangBackend

                self._estimator = SGLangBackend(self.model_name)
            else:
                raise ValueError(f"Unknown backend: {self._backend_type}")
        return self._estimator

    def run(self, samples: list[dict]) -> ExperimentResults:
        """
        Run experiment on a list of samples.

        Args:
            samples: List of dicts with keys: name, seed, target

        Returns:
            ExperimentResults with per-sample and aggregate metrics
        """
        estimator = self._get_estimator()

        sample_results = []
        for i, sample in enumerate(samples):
            name = sample.get("name", f"sample_{i}")
            seed = sample["seed"]
            target = sample["target"]

            logger.info("[%d/%d] %s...", i + 1, len(samples), name)

            result = estimator.compute(seed, target)

            sample_results.append(
                SampleResult(
                    name=name,
                    seed=seed,
                    target=target,
                    log_likelihood=result.log_prob,
                    perplexity=result.perplexity,
                    num_tokens=result.num_tokens,
                    rank_1_fraction=result.rank_1_fraction,
                    avg_rank=result.avg_rank,
                    token_log_probs=result.token_log_probs,
                    token_ranks=result.token_ranks or [],
                )
            )

        return ExperimentResults(
            model_name=self.model_name,
            experiment_name=self.experiment_name,
            timestamp=datetime.now().isoformat(),
            device=estimator.device,
            samples=sample_results,
        )

    def run_single(self, seed: str, target: str, name: str = "single") -> SampleResult:
        """Run on a single seed/target pair."""
        estimator = self._get_estimator()
        result = estimator.compute(seed, target)

        return SampleResult(
            name=name,
            seed=seed,
            target=target,
            log_likelihood=result.log_prob,
            perplexity=result.perplexity,
            num_tokens=result.num_tokens,
            rank_1_fraction=result.rank_1_fraction,
            avg_rank=result.avg_rank,
            token_log_probs=result.token_log_probs,
            token_ranks=result.token_ranks or [],
        )

    def compare_targets(
        self,
        seed: str,
        targets: dict[str, str],
    ) -> dict[str, SampleResult]:
        """
        Compare multiple targets given the same seed.

        Args:
            seed: The seed/prompt text
            targets: Dict mapping name to target text

        Returns:
            Dict mapping name to SampleResult
        """
        results = {}
        for name, target in targets.items():
            results[name] = self.run_single(seed, target, name=name)
        return results
