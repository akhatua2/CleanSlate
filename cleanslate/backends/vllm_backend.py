"""vLLM backend - GPU optimized inference."""

import logging

import torch
from torch.nn import functional as F

try:
    from vllm import LLM
except ImportError:
    raise ImportError(
        "vLLM not installed. Install with: pip install cleanslate[vllm] or uv sync --extra vllm"
    )

from cleanslate.backends.base import InferenceBackend, ProbabilityResult

logger = logging.getLogger(__name__)


class VLLMBackend(InferenceBackend):
    """
    vLLM backend for GPU-optimized inference.

    Requires: pip install cleanslate[vllm]

    Uses vLLM's LLM engine for efficient GPU inference.
    """

    def __init__(
        self, model_name: str, gpu_memory_utilization: float = 0.5, **llm_kwargs
    ):
        """
        Initialize the vLLM backend.

        Args:
            model_name: HuggingFace model name or path
            gpu_memory_utilization: Fraction of GPU memory to use (default: 0.5)
            **llm_kwargs: Additional arguments to pass to vLLM's LLM class
                          (e.g., tensor_parallel_size, max_model_len, etc.)
        """
        self._model_name = model_name

        # Default kwargs for vLLM LLM - VERY conservative settings for log prob computation
        default_kwargs = {
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": 2048,  # Reduce to 2K tokens to save memory
            "max_num_seqs": 1,  # Only process 1 sequence at a time
            "enforce_eager": True,  # Disable CUDA graphs
            "disable_log_stats": True,  # Disable logging overhead
        }
        default_kwargs.update(llm_kwargs)

        logger.info("Loading %s with vLLM on CUDA...", model_name)
        logger.info("  Memory utilization: %s", gpu_memory_utilization)
        logger.info("  Max sequence length: %s", default_kwargs["max_model_len"])
        self.llm = LLM(model=model_name, **default_kwargs)

        # Get tokenizer from the LLM engine
        self.tokenizer = self.llm.get_tokenizer()

        logger.info("Model loaded with vLLM on CUDA.")

    @property
    def device(self) -> str:
        return "cuda"

    @property
    def model_name(self) -> str:
        return self._model_name

    def compute(self, seed: str, target: str) -> ProbabilityResult:
        """
        Compute P(target | seed) using vLLM.

        The math:
        - P(target | seed) = ∏ P(t_i | seed, t_1, ..., t_{i-1})
        - log P(target | seed) = Σ log P(t_i | seed, t_1, ..., t_{i-1})
        """
        # Tokenize seed and full text separately to find boundary
        seed_ids = self.tokenizer.encode(seed, add_special_tokens=True)
        full_text = seed + target
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)

        # Target tokens start after seed
        seed_len = len(seed_ids)
        target_ids = full_ids[seed_len:]

        if len(target_ids) == 0:
            return ProbabilityResult(
                log_prob=0.0,
                perplexity=1.0,
                num_tokens=0,
                token_log_probs=[],
            )

        # Use vLLM's internal model to get logits
        # vLLM v0.15.0 has different internal structure
        # Access through the engine_core client
        try:
            # Try v0.15.0+ API structure
            if hasattr(self.llm, "llm_engine"):
                engine = self.llm.llm_engine
                # In v0.15.0, the structure changed
                if hasattr(engine, "engine_core"):
                    # V1 engine with engine_core
                    raise RuntimeError(
                        "vLLM v0.15.0 uses V1 engine which doesn't expose model directly. "
                        "Use HuggingFace backend instead for log probability computation."
                    )
                elif hasattr(engine, "model_executor"):
                    # V0 engine (older API)
                    model = engine.model_executor.driver_worker.model_runner.model
                else:
                    raise RuntimeError("Unknown vLLM engine structure")
            else:
                raise RuntimeError("Cannot access vLLM engine")

        except Exception as e:
            raise RuntimeError(
                f"Cannot access vLLM model for log probability computation: {e}\n\n"
                "vLLM v0.15.0+ doesn't expose the model directly for custom forward passes.\n"
                "Recommendation: Use 'huggingface' backend for log probability computation.\n"
                "It's simpler, more reliable, and uses less memory for this use case."
            )

        # Prepare input tensor
        input_ids = torch.tensor([full_ids], device="cuda", dtype=torch.long)

        # Run forward pass to get logits
        # vLLM models typically have a forward method that returns logits
        with torch.no_grad():
            try:
                # Try to use the model's forward method directly
                # Most vLLM models support this
                outputs = model.forward(input_ids, use_cache=False)

                # Extract logits from output
                if isinstance(outputs, tuple):
                    logits = outputs[0]  # First element is usually logits
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

            except Exception as e:
                # Fallback: try accessing through model's layers
                # Some vLLM models structure things differently
                try:
                    # Try using the model's compute_logits if available
                    if hasattr(model, "compute_logits"):
                        # This might require different input format
                        logits = model.compute_logits(input_ids)
                    else:
                        # Last resort: use the underlying transformer model
                        if hasattr(model, "model"):
                            base_model = model.model
                            outputs = base_model(input_ids)
                            logits = (
                                outputs.logits
                                if hasattr(outputs, "logits")
                                else outputs[0]
                            )
                        else:
                            raise RuntimeError(
                                f"Could not extract logits from vLLM model: {e}"
                            )
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to get logits from vLLM model. "
                        f"Original error: {e}, Fallback error: {e2}"
                    )

        # Get log probabilities via log_softmax
        log_probs = F.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]

        # Extract log probs and ranks for target tokens
        # logits[i] predicts token[i+1]
        # So for target token at position seed_len, we look at logits[seed_len-1]
        token_log_probs: list[float] = []
        token_ranks: list[int] = []

        for i, token_id in enumerate(target_ids):
            logit_pos = seed_len - 1 + i
            if logit_pos >= log_probs.shape[1]:
                break

            # Get log prob for this token
            token_log_prob = log_probs[0, logit_pos, token_id].item()
            token_log_probs.append(token_log_prob)

            # Get rank: how many tokens have higher probability?
            # rank 1 = this was the model's top choice
            probs_at_pos = log_probs[0, logit_pos]
            rank = int((probs_at_pos > token_log_prob).sum().item()) + 1
            token_ranks.append(rank)

        # Aggregate
        total_log_prob = sum(token_log_probs)
        num_tokens = len(token_log_probs)

        # Perplexity = exp(-avg_log_prob) = exp(-total_log_prob / num_tokens)
        perplexity = (
            float(torch.exp(torch.tensor(-total_log_prob / num_tokens)).item())
            if num_tokens > 0
            else 1.0
        )

        return ProbabilityResult(
            log_prob=total_log_prob,
            perplexity=perplexity,
            num_tokens=num_tokens,
            token_log_probs=token_log_probs,
            token_ranks=token_ranks,
        )
