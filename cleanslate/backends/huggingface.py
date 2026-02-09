"""HuggingFace Transformers backend - works on CPU and GPU."""

import logging
import time
from typing import Any

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cleanslate.backends.base import GenerationResult, InferenceBackend, ProbabilityResult
from cleanslate.chat_format import format_chat as _format_chat

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
    def compute(
        self, seed: str, target: str, **decoding_options: Any
    ) -> ProbabilityResult:
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

    def format_chat(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        """
        Format chat messages using the model's chat template.
        
        Args:
            messages: List of message dicts with "role" and "content" keys
                      e.g., [{"role": "system", "content": "..."}, 
                             {"role": "user", "content": "..."}]
            add_generation_prompt: Whether to add the generation prompt at the end
            
        Returns:
            Formatted prompt string ready for generation
        """
        return _format_chat(
            self.tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    @torch.no_grad()
    def generate_chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        capture_top_k: int = 50,
    ) -> GenerationResult:
        """
        Generate text from chat messages with captured logits.
        
        Uses the model's chat template to format messages, then generates.
        
        Args:
            messages: List of message dicts with "role" and "content" keys
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (vs greedy)
            capture_top_k: Number of top tokens to save per position
            
        Returns:
            GenerationResult with generated text and probability data
        """
        prompt = self.format_chat(messages)
        return self.generate(
            seed=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            capture_top_k=capture_top_k,
        )

    @torch.no_grad()
    def generate_chat_batch(
        self,
        messages_batch: list[list[dict]],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        capture_top_k: int = 50,
    ) -> list[GenerationResult]:
        """
        Generate text from multiple chat message lists in a batch.
        
        Args:
            messages_batch: List of message lists, each formatted for chat
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (vs greedy)
            capture_top_k: Number of top tokens to save per position
            
        Returns:
            List of GenerationResult, one per message list
        """
        prompts = [self.format_chat(msgs) for msgs in messages_batch]
        return self.generate_batch(
            seeds=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            capture_top_k=capture_top_k,
        )

    @torch.no_grad()
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
        start_time = time.time()

        # Tokenize input
        input_ids = self.tokenizer.encode(
            seed, return_tensors="pt", add_special_tokens=True
        )
        input_ids = input_ids.to(self._device)

        # Build generation kwargs
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            if top_p is not None:
                gen_kwargs["top_p"] = top_p

        outputs = self.model.generate(input_ids, **gen_kwargs)

        generation_time_ms = (time.time() - start_time) * 1000

        # Extract generated tokens (excluding input)
        generated_ids: list[int] = outputs.sequences[0][len(input_ids[0]) :].tolist()
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=False
        )
        generated_tokens = [self.tokenizer.decode([tid]) for tid in generated_ids]

        # Extract log probabilities and top-k alternatives
        token_log_probs: list[float] = []
        top_k_tokens_list: list[list[int]] = []
        top_k_probs_list: list[list[float]] = []

        if hasattr(outputs, "scores") and outputs.scores:
            for i, score in enumerate(outputs.scores):
                if i >= len(generated_ids):
                    break

                log_probs = F.log_softmax(score[0], dim=-1)

                # Log prob for the selected token
                token_log_probs.append(log_probs[generated_ids[i]].item())

                # Top-k alternatives
                k = min(capture_top_k, log_probs.shape[-1])
                top_k_logprobs, top_k_indices = torch.topk(log_probs, k=k)
                top_k_tokens_list.append(top_k_indices.tolist())
                top_k_probs_list.append(top_k_logprobs.tolist())

        return GenerationResult(
            generated_text=generated_text,
            generated_token_ids=generated_ids,
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
