"""SGLang backend - Efficient GPU inference with better memory management."""

import torch
from torch.nn import functional as F

try:
    import sglang as sgl
except ImportError:
    raise ImportError(
        "SGLang not installed. Install with: pip install cleanslate[sglang] or uv sync --extra sglang"
    )

from .base import InferenceBackend, ProbabilityResult, GenerationResult


class SGLangBackend(InferenceBackend):
    """
    SGLang backend for GPU-optimized inference.

    Requires: pip install cleanslate[sglang]

    SGLang is more memory-efficient than vLLM and has better APIs for log probability computation.
    """

    def __init__(self, model_name: str, **runtime_kwargs):
        """
        Initialize the SGLang backend.

        Args:
            model_name: HuggingFace model name or path
            **runtime_kwargs: Additional arguments to pass to SGLang's Runtime
                             (e.g., tp_size, mem_fraction_static, etc.)
        """
        self._model_name = model_name
        
        # Default kwargs for SGLang Runtime
        default_kwargs = {
            "trust_remote_code": True,
            "mem_fraction_static": 0.7,  # Use 70% of GPU memory
        }
        default_kwargs.update(runtime_kwargs)
        
        print(f"Loading {model_name} with SGLang on CUDA...")
        
        # Initialize SGLang runtime
        self.runtime = sgl.Runtime(
            model_path=model_name,
            **default_kwargs
        )
        
        # Get tokenizer
        self.tokenizer = self.runtime.tokenizer
        
        print(f"Model loaded with SGLang on CUDA.")

    @property
    def device(self) -> str:
        return "cuda"

    @property
    def model_name(self) -> str:
        return self._model_name

    def compute(self, seed: str, target: str) -> ProbabilityResult:
        """
        Compute P(target | seed) using SGLang.

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

        # Use SGLang's forward method to get logits
        # SGLang provides a cleaner API for this
        with torch.no_grad():
            # Get logits from SGLang runtime
            # We'll use the model's forward pass directly
            input_ids = torch.tensor([full_ids], device="cuda", dtype=torch.long)
            
            # Access the underlying model
            model = self.runtime.model
            outputs = model(input_ids)
            
            # Extract logits
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

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
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        capture_top_k: int = 50,
    ) -> GenerationResult:
        """
        Generate text from seed and capture token-level probabilities.

        Args:
            seed: The prefix/prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering for sampling
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (vs greedy decode)
            capture_top_k: Number of top tokens/probs to capture per position

        Returns:
            GenerationResult with generated text and probability data
        """
        import time
        start_time = time.time()

        with torch.no_grad():
            # Tokenize seed
            seed_ids = self.tokenizer.encode(seed, add_special_tokens=True)
            input_ids = torch.tensor([seed_ids], device="cuda", dtype=torch.long)

            # Get the model from runtime
            model = self.runtime.model

            # Track generated tokens and their logprobs
            generated_token_ids: list[int] = []
            token_log_probs: list[float] = []
            top_k_tokens_list: list[list[int]] = []
            top_k_probs_list: list[list[float]] = []

            # Generate tokens one by one
            current_input = input_ids
            eos_token_id = self.tokenizer.eos_token_id

            for step in range(max_new_tokens):
                # Forward pass
                outputs = model(current_input)

                # Extract logits from last position
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                # Get last token logits and apply temperature
                next_token_logits = logits[0, -1, :] / temperature if temperature > 0 else logits[0, -1, :]

                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_vals, top_k_indices = torch.topk(next_token_logits, k=min(top_k, next_token_logits.shape[0]))
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_vals

                # Get log probabilities
                log_probs = F.log_softmax(next_token_logits, dim=-1)

                # Sample or greedy select
                if do_sample and temperature > 0:
                    probs = torch.exp(log_probs)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = log_probs.argmax(dim=-1).item()

                # Capture token info
                token_log_prob = log_probs[next_token].item()
                generated_token_ids.append(next_token)
                token_log_probs.append(token_log_prob)

                # Capture top-k alternatives
                top_k_count = min(capture_top_k, log_probs.shape[0])
                top_k_logprobs, top_k_indices = torch.topk(log_probs, k=top_k_count)
                top_k_tokens_list.append(top_k_indices.cpu().tolist())
                top_k_probs_list.append(top_k_logprobs.cpu().tolist())

                # Check for end of sequence
                if next_token == eos_token_id:
                    break

                # Append token and continue
                current_input = torch.cat([
                    current_input,
                    torch.tensor([[next_token]], device="cuda", dtype=torch.long)
                ], dim=1)

        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        generated_tokens = [
            self.tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in generated_token_ids
        ]

        generation_time_ms = (time.time() - start_time) * 1000

        return GenerationResult(
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            generated_tokens=generated_tokens,
            token_log_probs=token_log_probs,
            top_k_tokens=top_k_tokens_list,
            top_k_probs=top_k_probs_list,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            generation_time_ms=generation_time_ms,
        )

    def __del__(self):
        """Cleanup SGLang runtime."""
        if hasattr(self, 'runtime'):
            self.runtime.shutdown()
