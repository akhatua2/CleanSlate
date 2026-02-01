"""vLLM backend - GPU optimized inference (skeleton for now)."""

from .base import InferenceBackend, ProbabilityResult


class VLLMBackend(InferenceBackend):
    """
    vLLM backend for GPU-optimized inference.

    Requires: pip install cleanslate[vllm]

    TODO: Implement when GPU available.
    """

    def __init__(self, model_name: str):
        raise NotImplementedError(
            "vLLM backend not yet implemented. Use 'huggingface' backend instead."
        )

    @property
    def device(self) -> str:
        return "cuda"

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def compute(self, seed: str, target: str) -> ProbabilityResult:
        raise NotImplementedError
