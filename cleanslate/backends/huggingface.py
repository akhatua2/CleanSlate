"""HuggingFace Transformers backend - works on CPU and GPU."""

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import GenerationResult, InferenceBackend, ProbabilityResult


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

        print(f"Loading {model_name} on {self._device} with {dtype}...")

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
        print(f"Model loaded on {self._device}.")

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

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits.
        
        For tokens that appear in input_ids, divide their logits by penalty
        if logit > 0, multiply by penalty if logit < 0.
        
        Args:
            logits: [1, seq_len, vocab_size]
            input_ids: [1, seq_len] - tokens seen so far
            penalty: Repetition penalty factor (>1 discourages repetition)
        """
        if penalty == 1.0:
            return logits
            
        # For each position, penalize tokens that appeared before that position
        for i in range(logits.shape[1]):
            # Tokens seen up to position i
            seen_tokens = input_ids[0, :i+1].unique()
            
            for token_id in seen_tokens:
                if logits[0, i, token_id] > 0:
                    logits[0, i, token_id] = logits[0, i, token_id] / penalty
                else:
                    logits[0, i, token_id] = logits[0, i, token_id] * penalty
                    
        return logits

    def _apply_top_p(
        self,
        logits: torch.Tensor,
        top_p: float,
    ) -> torch.Tensor:
        """
        Apply nucleus (top-p) sampling filter.
        
        Keep smallest set of tokens with cumulative probability >= top_p.
        Set all other logits to -inf.
        
        Args:
            logits: [1, seq_len, vocab_size]
            top_p: Cumulative probability threshold (e.g., 0.9)
        """
        if top_p >= 1.0:
            return logits
            
        # Sort logits descending
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        # Shift right to keep at least one token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter back to original ordering
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        
        return logits

    @torch.no_grad()
    def compute(
        self,
        seed: str,
        target: str,
        **decoding_options,
    ) -> ProbabilityResult:
        """
        Compute P(target | seed) using HuggingFace forward pass.

        The math:
        - P(target | seed) = ∏ P(t_i | seed, t_1, ..., t_{i-1})
        - log P(target | seed) = Σ log P(t_i | seed, t_1, ..., t_{i-1})
        
        Decoding options (kwargs):
        - temperature (float): Divide logits by temperature before softmax.
          Higher = flatter distribution, lower = peakier. Default 1.0.
        - top_k (int): Only consider top-k tokens at each position.
          If target outside top-k, pz=0. Default None (all tokens).
        - top_p (float): Nucleus sampling - keep smallest set of tokens with
          cumulative prob >= top_p. Default None (disabled).
        - repetition_penalty (float): Penalize tokens that appeared earlier.
          >1.0 discourages repetition, <1.0 encourages it. Default 1.0.
        """
        # Extract decoding options with defaults
        temperature = decoding_options.get("temperature", 1.0)
        top_k = decoding_options.get("top_k", None)
        top_p = decoding_options.get("top_p", None)
        repetition_penalty = decoding_options.get("repetition_penalty", 1.0)
        
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
                target_tokens=[],
                predicted_tokens=[],
                decoding_options=decoding_options or None,
            )

        # Forward pass
        input_ids = torch.tensor([full_ids], device=self._device)
        outputs = self.model(input_ids)
        logits = outputs.logits.clone()  # [1, seq_len, vocab_size]

        # Apply repetition penalty (before other transforms)
        if repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, input_ids, repetition_penalty)

        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering: set non-top-k logits to -inf
        if top_k is not None:
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            kth_vals = top_k_vals[:, :, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < kth_vals, float("-inf"))

        # Apply top-p (nucleus) filtering
        if top_p is not None:
            logits = self._apply_top_p(logits, top_p)

        # Get log probabilities via log_softmax
        log_probs = F.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]

        # Extract log probs, ranks, and predicted tokens for target tokens
        # logits[i] predicts token[i+1]
        # So for target token at position seed_len, we look at logits[seed_len-1]
        token_log_probs: list[float] = []
        token_ranks: list[int] = []
        target_tokens_decoded: list[str] = []
        predicted_tokens: list[str] = []
        has_impossible_token = False

        for i, token_id in enumerate(target_ids):
            logit_pos = seed_len - 1 + i
            if logit_pos >= log_probs.shape[1]:
                break

            # Get log prob for this token
            token_log_prob = log_probs[0, logit_pos, token_id].item()
            
            # Check for impossible token (outside top-k/top-p)
            if not torch.isfinite(torch.tensor(token_log_prob)):
                has_impossible_token = True
                token_log_probs.append(float("-inf"))
            else:
                token_log_probs.append(token_log_prob)

            # Get rank: how many tokens have higher probability?
            # rank 1 = this was the model's top choice
            probs_at_pos = log_probs[0, logit_pos]
            finite_mask = torch.isfinite(probs_at_pos)
            rank = int((probs_at_pos[finite_mask] > token_log_prob).sum().item()) + 1
            token_ranks.append(rank)

            # Decode target token
            target_tokens_decoded.append(self.tokenizer.decode([token_id]))

            # Get model's top-1 prediction (among valid tokens)
            top_token_id = probs_at_pos.argmax().item()
            predicted_tokens.append(self.tokenizer.decode([top_token_id]))

        # Aggregate
        num_tokens = len(token_log_probs)
        
        if has_impossible_token:
            total_log_prob = float("-inf")
            perplexity = float("inf")
        else:
            total_log_prob = sum(token_log_probs)
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
            target_tokens=target_tokens_decoded,
            predicted_tokens=predicted_tokens,
            decoding_options=decoding_options or None,
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
        
        Uses HuggingFace generate() with output_scores=True to capture logits
        at each generation step.
        """
        import time
        
        start_time = time.perf_counter()
        
        # Tokenize seed
        inputs = self.tokenizer(seed, return_tensors="pt").to(self._device)
        seed_length = inputs.input_ids.shape[1]
        
        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "output_scores": True,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "attention_mask": inputs.attention_mask,
        }
        
        if temperature != 1.0:
            gen_kwargs["temperature"] = temperature
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
            
        # Generate
        outputs = self.model.generate(inputs.input_ids, **gen_kwargs)
        
        generation_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Extract generated token IDs (excluding seed)
        generated_ids = outputs.sequences[0, seed_length:].tolist()
        
        # Decode tokens
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_tokens = [self.tokenizer.decode([tid]) for tid in generated_ids]
        
        # Process scores (logits at each generation step)
        # outputs.scores is a tuple of tensors, one per generated token
        # Each tensor is [batch_size, vocab_size]
        token_log_probs: list[float] = []
        top_k_tokens: list[list[int]] = []
        top_k_probs: list[list[float]] = []
        
        for i, (token_id, score) in enumerate(zip(generated_ids, outputs.scores)):
            # Apply temperature if not already applied (depends on HF version)
            # scores from generate() are pre-softmax logits
            logits = score[0]  # [vocab_size]
            
            # Compute log probs
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get log prob for the actually generated token
            token_log_probs.append(log_probs[token_id].item())
            
            # Get top-k tokens and their log probs
            topk_vals, topk_ids = torch.topk(log_probs, min(capture_top_k, log_probs.size(-1)))
            top_k_tokens.append(topk_ids.tolist())
            top_k_probs.append(topk_vals.tolist())
        
        return GenerationResult(
            generated_text=generated_text,
            generated_token_ids=generated_ids,
            generated_tokens=generated_tokens,
            token_log_probs=token_log_probs,
            top_k_tokens=top_k_tokens,
            top_k_probs=top_k_probs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            generation_time_ms=generation_time_ms,
        )

    @torch.no_grad()
    def generate_batch(
        self,
        seeds: list[str],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        capture_top_k: int = 50,
    ) -> list[GenerationResult]:
        """
        Generate text from multiple seeds in a single batched forward pass.
        
        Much faster than sequential generation on GPU.
        """
        import time
        
        if len(seeds) == 0:
            return []
        
        start_time = time.perf_counter()
        
        # Tokenize all seeds with padding (left padding for generation)
        # Left padding is required for batched generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        inputs = self.tokenizer(
            seeds,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._device)
        
        # Track seed lengths (without padding) for each sequence
        attention_mask = inputs.attention_mask
        seed_lengths = attention_mask.sum(dim=1).tolist()  # Actual token counts per seed
        
        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "output_scores": True,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "attention_mask": attention_mask,
        }
        
        if temperature != 1.0:
            gen_kwargs["temperature"] = temperature
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        
        # Generate batch
        outputs = self.model.generate(inputs.input_ids, **gen_kwargs)
        
        generation_time_ms = (time.perf_counter() - start_time) * 1000
        time_per_seq = generation_time_ms / len(seeds)
        
        # Process each sequence in the batch
        results: list[GenerationResult] = []
        batch_size = len(seeds)
        input_length = inputs.input_ids.shape[1]  # Padded length
        
        for batch_idx in range(batch_size):
            # Generated IDs start after the padded input
            # outputs.sequences is [batch_size, input_len + generated_len]
            full_seq = outputs.sequences[batch_idx]
            generated_ids = full_seq[input_length:].tolist()
            
            # Remove any trailing pad tokens
            while generated_ids and generated_ids[-1] == self.tokenizer.pad_token_id:
                generated_ids.pop()
            
            # Decode tokens
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_tokens = [self.tokenizer.decode([tid]) for tid in generated_ids]
            
            # Process scores for this batch element
            # outputs.scores is tuple of [batch_size, vocab_size] tensors
            token_log_probs: list[float] = []
            top_k_tokens_list: list[list[int]] = []
            top_k_probs_list: list[list[float]] = []
            
            for step_idx, token_id in enumerate(generated_ids):
                if step_idx >= len(outputs.scores):
                    break
                    
                score = outputs.scores[step_idx]  # [batch_size, vocab_size]
                logits = score[batch_idx]  # [vocab_size]
                
                # Compute log probs
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get log prob for the actually generated token
                token_log_probs.append(log_probs[token_id].item())
                
                # Get top-k tokens and their log probs
                topk_vals, topk_ids = torch.topk(
                    log_probs, min(capture_top_k, log_probs.size(-1))
                )
                top_k_tokens_list.append(topk_ids.tolist())
                top_k_probs_list.append(topk_vals.tolist())
            
            results.append(GenerationResult(
                generated_text=generated_text,
                generated_token_ids=generated_ids,
                generated_tokens=generated_tokens,
                token_log_probs=token_log_probs,
                top_k_tokens=top_k_tokens_list,
                top_k_probs=top_k_probs_list,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                generation_time_ms=time_per_seq,
            ))
        
        return results
