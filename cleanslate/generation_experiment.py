"""Generation experiment - generate N outputs per seed and save everything."""

from datetime import datetime
import time
from typing import Optional

from .backends.base import InferenceBackend
from .backends.huggingface import HuggingFaceBackend
from .generation_data import GenerationOutput, GenerationBatch, save_generations_parquet, save_generations_hf_dataset


class GenerationExperiment:
    """
    Experiment for generating multiple outputs per seed and saving all data.
    
    Unlike the standard Experiment which computes metrics, this saves raw generations
    for later analysis.
    
    Example:
        exp = GenerationExperiment(model="Qwen/Qwen2.5-0.5B-Instruct")
        exp.run_generation(
            samples=SHAKESPEARE_SAMPLES,
            num_generations=3,
            output_path="generations.parquet"
        )
    """

    def __init__(
        self,
        model: str,
        name: str = "generation_experiment",
        backend: str = "huggingface",
        device: str | None = None,
    ):
        """
        Initialize generation experiment.

        Args:
            model: HuggingFace model name or path
            name: Experiment name for tracking
            backend: Backend to use (currently only "huggingface" supported)
            device: Device override. None for auto-detection.
        """
        self.model_name = model
        self.experiment_name = name
        self._backend_type = backend
        self._device = device
        self._estimator: InferenceBackend | None = None

    def _get_backend(self) -> InferenceBackend:
        """Lazy-load the model."""
        if self._estimator is None:
            if self._backend_type == "huggingface":
                self._estimator = HuggingFaceBackend(
                    self.model_name, device=self._device
                )
            else:
                raise ValueError(f"Backend {self._backend_type} not supported for generation yet")
        return self._estimator

    def generate_from_seed(
        self,
        seed: str,
        num_generations: int = 3,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        sample_id: str = "sample",
        sample_name: str = "sample",
        original_target: Optional[str] = None,
    ) -> list[GenerationOutput]:
        """
        Generate multiple outputs from a single seed.

        Args:
            seed: The prompt/seed text
            num_generations: Number of generations to create
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            sample_id: Unique ID for this sample
            sample_name: Human-readable name
            original_target: Original target text (for memorization tests)

        Returns:
            List of GenerationOutput objects
        """
        backend = self._get_backend()
        
        if not hasattr(backend, 'generate'):
            raise NotImplementedError(
                f"Backend {self._backend_type} doesn't support generation yet"
            )

        generations = []
        
        for i in range(num_generations):
            print(f"  Generation {i+1}/{num_generations}...")
            
            start_time = time.time()
            
            # Generate
            result = backend.generate(
                seed=seed,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                return_logprobs=True,
            )
            
            generation_time_ms = (time.time() - start_time) * 1000
            
            # Create GenerationOutput
            gen_output = GenerationOutput(
                sample_id=sample_id,
                sample_name=sample_name,
                generation_index=i,
                seed=seed,
                original_target=original_target,
                model_name=self.model_name,
                backend=self._backend_type,
                device=backend.device,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                generated_text=result["generated_text"],
                num_tokens=result["num_tokens"],
                finish_reason=result["finish_reason"],
                tokens=result["tokens"],
                token_strings=result["token_strings"],
                token_logprobs=result.get("token_logprobs", []),
                top_logprobs_data=result.get("top_logprobs", []),
                cumulative_logprob=result.get("cumulative_logprob"),
                mean_logprob=result.get("mean_logprob"),
                timestamp=datetime.now().isoformat(),
                generation_time_ms=generation_time_ms,
            )
            
            generations.append(gen_output)
        
        return generations

    def run_generation(
        self,
        samples: list[dict],
        num_generations: int = 3,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        output_path: Optional[str] = None,
        save_format: str = "parquet",
    ) -> list[GenerationOutput]:
        """
        Run generation experiment on multiple samples.

        Args:
            samples: List of dicts with keys: name, seed (and optionally: target)
            num_generations: Number of generations per seed
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            output_path: Where to save results (optional)
            save_format: "parquet", "hf_dataset", or "both"

        Returns:
            List of all GenerationOutput objects
        """
        all_generations = []
        
        print(f"Generating {num_generations} outputs per seed for {len(samples)} samples...")
        print(f"Total generations: {len(samples) * num_generations}")
        print()
        
        for i, sample in enumerate(samples):
            sample_name = sample.get("name", f"sample_{i}")
            sample_id = f"sample_{i:04d}"
            seed = sample["seed"]
            original_target = sample.get("target", None)
            
            print(f"[{i + 1}/{len(samples)}] {sample_name}")
            
            generations = self.generate_from_seed(
                seed=seed,
                num_generations=num_generations,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                sample_id=sample_id,
                sample_name=sample_name,
                original_target=original_target,
            )
            
            all_generations.extend(generations)
            print()
        
        print(f"✅ Generated {len(all_generations)} outputs total")
        print()
        
        # Save if requested
        if output_path:
            if save_format == "parquet" or save_format == "both":
                parquet_path = output_path if output_path.endswith(".parquet") else f"{output_path}.parquet"
                save_generations_parquet(all_generations, parquet_path)
            
            if save_format == "hf_dataset" or save_format == "both":
                hf_path = output_path.replace(".parquet", "")
                save_generations_hf_dataset(all_generations, hf_path)
        
        return all_generations
