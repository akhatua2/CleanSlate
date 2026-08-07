"""vLLM backend using prompt_logprobs for teacher-forcing P(target|seed).

Requires vLLM >= 0.19.0. Uses the official LLM.generate() API with
SamplingParams(prompt_logprobs=1, max_tokens=1) to extract per-token
log probabilities for the prompt.

Supports data parallelism via the ``data_parallel`` config key.  When
``data_parallel > 1``, worker processes are spawned — each pinned to its own
GPU(s) via ``CUDA_VISIBLE_DEVICES`` — and batches are split across them.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
from multiprocessing.context import SpawnProcess
from typing import Any

from cleanslate.backends.base import InferenceBackend, ProbabilityResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker process helpers (run in spawned subprocesses)
# ---------------------------------------------------------------------------

def _worker_loop(
    rank: int,
    model_name: str,
    vllm_config: dict[str, Any],
    gpu_ids: list[int],
    task_queue: mp.Queue[list[tuple[str, str]] | None],
    result_queue: mp.Queue[list[ProbabilityResult]],
) -> None:
    """Long-lived worker: load model once, process batches until sentinel."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    from vllm import LLM, SamplingParams

    cfg = dict(vllm_config)
    cfg.pop("data_parallel", None)

    llm_kwargs: dict[str, Any] = dict(
        model=model_name,
        trust_remote_code=cfg.get("trust_remote_code", True),
        dtype=cfg.get("dtype", "bfloat16"),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.90),
        max_model_len=cfg.get("max_model_len", 4096),
        tensor_parallel_size=cfg.get("num_gpus", 1),
        enforce_eager=cfg.get("enforce_eager", True),
        disable_log_stats=cfg.get("disable_log_stats", True),
    )
    if "enable_chunked_prefill" in cfg:
        llm_kwargs["enable_chunked_prefill"] = cfg["enable_chunked_prefill"]
    if cfg.get("quantization"):
        llm_kwargs["quantization"] = cfg["quantization"]
    if cfg.get("allow_deprecated_quantization"):
        llm_kwargs["allow_deprecated_quantization"] = True
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)

    while True:
        pairs = task_queue.get()
        if pairs is None:  # sentinel → shutdown
            break

        prompts: list[str] = []
        seed_lens: list[int] = []
        for seed, target in pairs:
            seed_ids = tokenizer.encode(seed, add_special_tokens=True)
            prompts.append(seed + target)
            seed_lens.append(len(seed_ids))

        outputs = llm.generate(prompts, params, use_tqdm=False)
        results = [_parse_output(output, seed_lens[i]) for i, output in enumerate(outputs)]
        result_queue.put(results)


def _parse_output(output: Any, seed_len: int) -> ProbabilityResult:
    prompt_logprobs = output.prompt_logprobs
    prompt_token_ids = output.prompt_token_ids

    if prompt_logprobs is None or len(prompt_token_ids) <= seed_len:
        return ProbabilityResult(log_prob=0.0, num_tokens=0, token_log_probs=[])

    token_log_probs: list[float] = []
    token_ranks: list[int] = []

    for pos in range(seed_len, len(prompt_token_ids)):
        token_id = prompt_token_ids[pos]
        logprob_dict = prompt_logprobs[pos]
        if logprob_dict is None:
            continue

        if token_id in logprob_dict:
            entry = logprob_dict[token_id]
            token_log_probs.append(float(entry.logprob))
            token_ranks.append(int(entry.rank) if entry.rank is not None else 0)
        else:
            token_log_probs.append(float("-inf"))
            token_ranks.append(0)

    total = sum(lp for lp in token_log_probs if lp != float("-inf"))
    return ProbabilityResult(
        log_prob=total,
        num_tokens=len(token_log_probs),
        token_log_probs=token_log_probs,
        token_ranks=token_ranks,
    )


# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------

class VLLMBackend(InferenceBackend):
    """vLLM-backed inference for p_z computation.

    When ``data_parallel > 1`` in *vllm_config*, the model is replicated across
    GPU groups and batches are split across them for higher throughput.
    """

    def __init__(self, model_name: str, vllm_config: dict[str, Any] | None = None) -> None:
        self._model_name = model_name
        cfg = vllm_config or {}
        self._dp_size = cfg.get("data_parallel", 1)
        tp_size = cfg.get("num_gpus", 1)

        gpu_ids = cfg.get("gpu_ids")
        if gpu_ids is not None:
            expected = self._dp_size * tp_size
            if len(gpu_ids) != expected:
                raise ValueError(
                    f"gpu_ids has {len(gpu_ids)} entries but data_parallel*num_gpus = {expected}"
                )
            self._gpu_ids: list[int] = [int(g) for g in gpu_ids]
        else:
            self._gpu_ids = list(range(self._dp_size * tp_size))

        if self._dp_size > 1:
            self._init_dp(model_name, cfg, tp_size)
        else:
            self._init_single(model_name, cfg)

    # -- single-process init (original path) --------------------------------

    def _init_single(self, model_name: str, cfg: dict[str, Any]) -> None:
        from vllm import LLM

        if cfg.get("gpu_ids") is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self._gpu_ids)

        logger.info("Loading %s with vLLM...", model_name)
        llm_kwargs: dict[str, Any] = dict(
            model=model_name,
            trust_remote_code=cfg.get("trust_remote_code", True),
            dtype=cfg.get("dtype", "bfloat16"),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.90),
            max_model_len=cfg.get("max_model_len", 4096),
            tensor_parallel_size=cfg.get("num_gpus", 1),
            enforce_eager=cfg.get("enforce_eager", True),
            disable_log_stats=cfg.get("disable_log_stats", True),
        )
        if "enable_chunked_prefill" in cfg:
            llm_kwargs["enable_chunked_prefill"] = cfg["enable_chunked_prefill"]
        if cfg.get("quantization"):
            llm_kwargs["quantization"] = cfg["quantization"]
        if cfg.get("allow_deprecated_quantization"):
            llm_kwargs["allow_deprecated_quantization"] = True
        self._llm: LLM | None = LLM(**llm_kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        self._workers: list[SpawnProcess] = []
        self._task_queues: list[mp.Queue[list[tuple[str, str]] | None]] = []
        self._result_queues: list[mp.Queue[list[ProbabilityResult]]] = []
        atexit.register(self._shutdown_engine)
        logger.info("Model loaded.")

    # -- data-parallel init -------------------------------------------------

    def _init_dp(self, model_name: str, cfg: dict[str, Any], tp_size: int) -> None:
        self._llm = None
        total_gpus = self._dp_size * tp_size
        logger.info(
            "Starting %d DP workers (TP=%d each, %d GPUs total) for %s...",
            self._dp_size, tp_size, total_gpus, model_name,
        )

        # Prevent main process from claiming GPU memory
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        ctx = mp.get_context("spawn")
        self._task_queues = []
        self._result_queues = []
        self._workers = []

        for rank in range(self._dp_size):
            gpu_ids = self._gpu_ids[rank * tp_size : (rank + 1) * tp_size]
            tq: mp.Queue[list[tuple[str, str]] | None] = ctx.Queue()
            rq: mp.Queue[list[ProbabilityResult]] = ctx.Queue()
            p = ctx.Process(
                target=_worker_loop,
                args=(rank, model_name, cfg, gpu_ids, tq, rq),
                daemon=False,
            )
            p.start()
            self._task_queues.append(tq)
            self._result_queues.append(rq)
            self._workers.append(p)

        # Wait for workers to be ready by sending an empty batch to each
        for tq in self._task_queues:
            tq.put([])
        for rq in self._result_queues:
            rq.get()  # empty list back

        # Load tokenizer in main process (lightweight, no GPU)
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        atexit.register(self._shutdown_workers)
        logger.info("All %d DP workers ready.", self._dp_size)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def compute(self, seed: str, target: str) -> ProbabilityResult:
        return self.compute_batch([(seed, target)])[0]

    def compute_batch(self, pairs: list[tuple[str, str]]) -> list[ProbabilityResult]:
        if not pairs:
            return []

        if self._dp_size > 1:
            return self._compute_batch_dp(pairs)
        return self._compute_batch_single(pairs)

    # -- single-process batch -----------------------------------------------

    def _compute_batch_single(self, pairs: list[tuple[str, str]]) -> list[ProbabilityResult]:
        from vllm import SamplingParams

        assert self._llm is not None
        prompts: list[str] = []
        seed_lens: list[int] = []
        for seed, target in pairs:
            seed_ids = self._tokenizer.encode(seed, add_special_tokens=True)
            prompts.append(seed + target)
            seed_lens.append(len(seed_ids))

        params = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)
        outputs = self._llm.generate(prompts, params, use_tqdm=False)

        return [_parse_output(output, seed_lens[i]) for i, output in enumerate(outputs)]

    # -- data-parallel batch ------------------------------------------------

    def _compute_batch_dp(self, pairs: list[tuple[str, str]]) -> list[ProbabilityResult]:
        n = len(pairs)
        chunk_size = (n + self._dp_size - 1) // self._dp_size

        # Send chunks to workers
        chunks: list[list[tuple[str, str]]] = []
        for rank in range(self._dp_size):
            chunk = pairs[rank * chunk_size : (rank + 1) * chunk_size]
            chunks.append(chunk)
            self._task_queues[rank].put(chunk)

        # Collect results in order
        results: list[ProbabilityResult] = []
        for rank in range(self._dp_size):
            if chunks[rank]:
                results.extend(self._result_queues[rank].get())

        return results

    def _shutdown_engine(self) -> None:
        """Shut down vLLM engine cleanly, suppressing child-process noise."""
        if self._llm is None:
            return
        try:
            self._llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        # Silence stderr — child processes emit ResourceTracker noise on exit
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)
        except OSError:
            pass

    def _shutdown_workers(self) -> None:
        """Shut down DP worker processes."""
        for tq in getattr(self, "_task_queues", []):
            try:
                tq.put(None)  # sentinel
            except Exception:
                pass
        for w in getattr(self, "_workers", []):
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()
