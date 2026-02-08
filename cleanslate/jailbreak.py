"""Jailbreak experiment runner for testing lyrics extraction.

Runs generation experiments with different prompt templates and seed modes,
comparing model outputs against ground truth lyrics.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from .backends.base import GenerationResult, InferenceBackend
from .backends.huggingface import HuggingFaceBackend
from .prompts import PromptTemplate, SeedMode, get_templates, ALL_TEMPLATES
from .samples.billboard import get_lyrics_prefix


@dataclass
class JailbreakRow:
    """
    Single row in the jailbreak experiment output.
    
    Each row = one generation for one (song, template, seed_mode) combination.
    """
    
    # Identifiers
    model_name: str
    experiment_name: str
    song_name: str
    song_title: str
    song_artists: list[str]
    song_year: int
    generation_idx: int  # For multiple generations per config
    
    # Prompt configuration
    prompt_template: str
    seed_mode: str
    system_prompt: str | None
    user_prompt: str
    formatted_prompt: str  # Full prompt after chat template applied
    
    # Ground truth
    ground_truth_lyrics: str
    lyrics_hint: str  # What was provided as hint
    
    # Generation output
    generated_text: str
    generated_token_ids: list[int]
    generated_tokens: list[str]
    
    # Token-level probability data
    token_log_probs: list[float]
    top_k_tokens: list[list[int]]
    top_k_probs: list[list[float]]
    
    # Generation parameters
    max_new_tokens: int
    temperature: float
    top_p: float | None
    top_k: int | None
    do_sample: bool
    
    # Metadata
    timestamp: str
    device: str
    generation_time_ms: float
    
    def to_dict(self) -> dict:
        """Convert to dict for DataFrame/parquet."""
        return asdict(self)


BackendType = Literal["huggingface", "vllm"]


class JailbreakExperiment:
    """
    Experiment for testing lyrics extraction with different prompts.
    
    Example:
        exp = JailbreakExperiment(model="Qwen/Qwen3-0.6B")
        
        songs = load_billboard_songs(year_range=(2020, 2025), limit=50)
        
        results = exp.run(
            songs=songs,
            templates=["direct", "helpful_assistant", "music_expert"],
            seed_modes=["none", "first_line"],
            output_path="outputs/jailbreak_2020s.parquet",
        )
    """
    
    def __init__(
        self,
        model: str,
        backend: BackendType = "huggingface",
        device: str | None = None,
        experiment_name: str = "jailbreak",
    ):
        self.model_name = model
        self.experiment_name = experiment_name
        self._backend_type = backend
        self._device = device
        self._backend: InferenceBackend | None = None
    
    def _get_backend(self) -> HuggingFaceBackend:
        """Lazy-load the model."""
        if self._backend is None:
            if self._backend_type == "huggingface":
                self._backend = HuggingFaceBackend(
                    self.model_name, device=self._device
                )
            elif self._backend_type == "vllm":
                from .backends.vllm_backend import VLLMBackend
                self._backend = VLLMBackend(self.model_name)
            else:
                raise ValueError(f"Unknown backend: {self._backend_type}")
        return self._backend
    
    def run(
        self,
        songs: list[dict],
        templates: list[str] | None = None,
        seed_modes: list[SeedMode] | None = None,
        output_path: str | Path = "outputs/jailbreak",
        num_generations: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        capture_top_k: int = 50,
        batch_size: int = 4,
    ) -> list[JailbreakRow]:
        """
        Run jailbreak experiment on songs with various prompts.
        
        Args:
            songs: List of song dicts from load_billboard_songs()
            templates: List of template names to test (default: DEFAULT_TEMPLATES)
            seed_modes: List of seed modes to test (default: ["none", "first_line"])
            output_path: Base path for output files (saves .parquet and .json)
            num_generations: Number of generations per (song, template, seed_mode)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (vs greedy)
            capture_top_k: Number of top tokens to save per position
            batch_size: Number of generations per batch
            
        Returns:
            List of JailbreakRow objects
        """
        backend = self._get_backend()
        
        # Default templates and seed modes
        if templates is None:
            templates = ["direct", "helpful_assistant", "music_expert", "educational"]
        if seed_modes is None:
            seed_modes = ["none", "first_line"]
        
        # Get template objects
        template_objs = [ALL_TEMPLATES[t] for t in templates]
        
        # Build all jobs
        # Job: (song_idx, template, seed_mode, gen_idx)
        jobs: list[tuple[int, PromptTemplate, SeedMode, int]] = []
        for song_idx in range(len(songs)):
            for template in template_objs:
                for seed_mode in seed_modes:
                    for gen_idx in range(num_generations):
                        jobs.append((song_idx, template, seed_mode, gen_idx))
        
        total_jobs = len(jobs)
        num_batches = (total_jobs + batch_size - 1) // batch_size
        
        print(f"Jailbreak Experiment: {self.experiment_name}")
        print(f"  Model: {self.model_name}")
        print(f"  Songs: {len(songs)}")
        print(f"  Templates: {templates}")
        print(f"  Seed modes: {seed_modes}")
        print(f"  Generations per config: {num_generations}")
        print(f"  Total generations: {total_jobs}")
        print(f"  Batches: {num_batches} (batch_size={batch_size})")
        print()
        
        rows: list[JailbreakRow] = []
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_jobs)
            batch_jobs = jobs[batch_start:batch_end]
            
            print(f"[Batch {batch_idx + 1}/{num_batches}] Processing {len(batch_jobs)} generations...")
            
            # Prepare messages for batch
            batch_messages: list[list[dict]] = []
            batch_metadata: list[tuple] = []  # (song, template, seed_mode, gen_idx, hint)
            
            for song_idx, template, seed_mode, gen_idx in batch_jobs:
                song = songs[song_idx]
                lyrics_hint = get_lyrics_prefix(song.get("lyrics", ""), seed_mode)
                messages = template.format_messages(song, lyrics_hint)
                
                batch_messages.append(messages)
                batch_metadata.append((song, template, seed_mode, gen_idx, lyrics_hint, messages))
            
            # Generate batch
            results = backend.generate_chat_batch(
                messages_batch=batch_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                capture_top_k=capture_top_k,
            )
            
            # Create rows
            timestamp = datetime.now().isoformat()
            for (song, template, seed_mode, gen_idx, lyrics_hint, messages), result in zip(batch_metadata, results):
                # Get formatted prompt for reference
                formatted_prompt = backend.format_chat(messages)
                
                # Extract user prompt from messages
                user_prompt = ""
                for msg in messages:
                    if msg["role"] == "user":
                        user_prompt = msg["content"]
                        break
                
                row = JailbreakRow(
                    model_name=self.model_name,
                    experiment_name=self.experiment_name,
                    song_name=song["name"],
                    song_title=song["title"],
                    song_artists=song.get("artists", []),
                    song_year=song.get("year", 0),
                    generation_idx=gen_idx,
                    prompt_template=template.name,
                    seed_mode=seed_mode,
                    system_prompt=template.system_prompt,
                    user_prompt=user_prompt,
                    formatted_prompt=formatted_prompt,
                    ground_truth_lyrics=song.get("lyrics", ""),
                    lyrics_hint=lyrics_hint,
                    generated_text=result.generated_text,
                    generated_token_ids=result.generated_token_ids,
                    generated_tokens=result.generated_tokens,
                    token_log_probs=result.token_log_probs,
                    top_k_tokens=result.top_k_tokens,
                    top_k_probs=result.top_k_probs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    timestamp=timestamp,
                    device=backend.device,
                    generation_time_ms=result.generation_time_ms,
                )
                rows.append(row)
        
        # Save outputs
        path = Path(output_path)
        base = path.parent / path.stem
        
        self._save_output(rows, f"{base}.parquet", format="parquet")
        self._save_output(rows, f"{base}.json", format="json")
        
        print()
        print(f"Saved {len(rows)} generations to {base}.parquet and {base}.json")
        
        return rows
    
    def _save_output(
        self,
        rows: list[JailbreakRow],
        path: str | Path,
        format: str = "parquet",
    ) -> None:
        """Save rows to file in specified format."""
        import json
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [row.to_dict() for row in rows]
        
        if format == "json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif format == "parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                
                table = pa.Table.from_pylist(data)
                pq.write_table(table, path)
            except ImportError:
                import pandas as pd
                df = pd.DataFrame(data)
                df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def run_single(
        self,
        song: dict,
        template: str | PromptTemplate,
        seed_mode: SeedMode = "none",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> JailbreakRow:
        """
        Run a single generation for testing/debugging.
        
        Args:
            song: Song dict with title, artists, lyrics, etc.
            template: Template name or PromptTemplate object
            seed_mode: How much of lyrics to provide as hint
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Single JailbreakRow
        """
        backend = self._get_backend()
        
        if isinstance(template, str):
            template = ALL_TEMPLATES[template]
        
        lyrics_hint = get_lyrics_prefix(song.get("lyrics", ""), seed_mode)
        messages = template.format_messages(song, lyrics_hint)
        
        result = backend.generate_chat(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        
        formatted_prompt = backend.format_chat(messages)
        user_prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                user_prompt = msg["content"]
                break
        
        return JailbreakRow(
            model_name=self.model_name,
            experiment_name=self.experiment_name,
            song_name=song.get("name", f"{song['title']} - Unknown"),
            song_title=song["title"],
            song_artists=song.get("artists", []),
            song_year=song.get("year", 0),
            generation_idx=0,
            prompt_template=template.name,
            seed_mode=seed_mode,
            system_prompt=template.system_prompt,
            user_prompt=user_prompt,
            formatted_prompt=formatted_prompt,
            ground_truth_lyrics=song.get("lyrics", ""),
            lyrics_hint=lyrics_hint,
            generated_text=result.generated_text,
            generated_token_ids=result.generated_token_ids,
            generated_tokens=result.generated_tokens,
            token_log_probs=result.token_log_probs,
            top_k_tokens=result.top_k_tokens,
            top_k_probs=result.top_k_probs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=kwargs.get("top_p"),
            top_k=kwargs.get("top_k"),
            do_sample=kwargs.get("do_sample", True),
            timestamp=datetime.now().isoformat(),
            device=backend.device,
            generation_time_ms=result.generation_time_ms,
        )
