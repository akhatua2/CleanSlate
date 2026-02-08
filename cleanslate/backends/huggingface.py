"""HuggingFace Transformers backend - works on CPU and GPU."""

import logging

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import InferenceBackend, ProbabilityResult

logger = logging.getLogger(__name__)


class HuggingFaceBackend(InferenceBackend):
    """
    HuggingFace Transformers backend.

    Supports CPU, CUDA, and MPS (Apple Silicon).
    Auto-detects best available device if not specified.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize the HuggingFace backend.

        Args:
            model_name: HuggingFace model name or path (e.g., "Qwen/Qwen3-1.7B")
            device: Device to use. None for auto-detection.
                    Options: "cpu", "cuda", "cuda:0", "mps", "auto"
            dtype: Torch dtype for model. None for auto (float16 on GPU, float32 on CPU)
        """
        self._model_name = model_name
        self._device = self._resolve_device(device)

        # Auto-select dtype based on device
        if dtype is None:
            if self._device == "cpu":
                dtype = torch.float32
            else:
                dtype = torch.bfloat16

        self._dtype = dtype

        logger.info("Loading %s on %s with %s...", model_name, self._device, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self._dtype,
            device_map=self._device if self._device != "mps" else None,
            trust_remote_code=True,
        )

        # For MPS, need explicit .to() call
        if self._device == "mps":
            self.model = self.model.to(self._device)

        self.model.eval()
        logger.info("Model loaded on %s.", self._device)

    def _resolve_device(self, device: str | None) -> str:
        """Resolve device string to actual device."""
        if device is None or device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_name(self) -> str:
        return self._model_name

    @torch.no_grad()
    def compute(self, seed: str, target: str) -> ProbabilityResult:
        """
        Compute P(target | seed) using HuggingFace forward pass.

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

        # Forward pass
        input_ids = torch.tensor([full_ids], device=self._device)
        outputs = self.model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

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

    def generate(
        self,
        seed: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        return_logprobs: bool = True,
    ) -> dict:
        """
        Generate text from seed with detailed token information.

        Args:
            seed: The prompt/seed text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            return_logprobs: Whether to return log probabilities

        Returns:
            Dictionary with:
                - generated_text: The generated text
                - tokens: List of token IDs
                - token_strings: List of decoded tokens
                - token_logprobs: List of log probabilities (if return_logprobs=True)
                - top_logprobs: List of top-k alternatives per token (if return_logprobs=True)
                - finish_reason: Why generation stopped
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(
            seed, return_tensors="pt", add_special_tokens=True
        )
        input_ids = input_ids.to(self._device)

        # Generate with logprobs
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=return_logprobs,
            )

        # Extract generated tokens (excluding input)
        generated_ids = outputs.sequences[0][len(input_ids[0]) :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Decode individual tokens
        token_strings = [self.tokenizer.decode([tid]) for tid in generated_ids]

        # Determine finish reason
        finish_reason = "stop"
        if len(generated_ids) >= max_tokens:
            finish_reason = "length"
        elif generated_ids[-1].item() == self.tokenizer.eos_token_id:
            finish_reason = "eos"

        result = {
            "generated_text": generated_text,
            "tokens": generated_ids.cpu().tolist(),
            "token_strings": token_strings,
            "num_tokens": len(generated_ids),
            "finish_reason": finish_reason,
        }

        # Add log probabilities if requested
        if return_logprobs and hasattr(outputs, "scores") and outputs.scores:
            token_logprobs = []
            top_logprobs_data = []

            for i, score in enumerate(outputs.scores):
                # Get log probabilities
                log_probs = F.log_softmax(score[0], dim=-1)

                # Get log prob for the selected token
                selected_token = generated_ids[i].item()
                token_logprob = log_probs[selected_token].item()
                token_logprobs.append(token_logprob)

                # Get top-k alternatives
                top_k_logprobs, top_k_indices = torch.topk(
                    log_probs, k=min(10, len(log_probs))
                )
                top_alternatives = [
                    {
                        "token_id": int(top_k_indices[j]),
                        "token_str": self.tokenizer.decode([top_k_indices[j]]),
                        "logprob": float(top_k_logprobs[j]),
                    }
                    for j in range(len(top_k_indices))
                ]
                top_logprobs_data.append(top_alternatives)

            result["token_logprobs"] = token_logprobs
            result["top_logprobs"] = top_logprobs_data
            result["cumulative_logprob"] = sum(token_logprobs)
            result["mean_logprob"] = (
                sum(token_logprobs) / len(token_logprobs) if token_logprobs else 0.0
            )

        return result
