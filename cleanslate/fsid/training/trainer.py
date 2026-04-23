"""FSID unlearning trainer: HF Trainer subclass + entry-point."""

import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, override

import torch
import torch.distributed as dist
import yaml  # type: ignore[import-untyped]
from accelerate.logging import get_logger  # type: ignore[import-untyped]
from transformers import Trainer

from cleanslate.fsid.config import TrainerConfig
from cleanslate.fsid.datasets.io import load_local_records
from cleanslate.fsid.datasets.unlearning import (
    ForgetPairDataset,
    TextSequenceDataset,
    build_forget_pairs,
    tokenize_text_rows,
)
from cleanslate.fsid.training.data import ForgetRetainDataset, UnlearnDataCollator
from cleanslate.fsid.training.losses import (
    _build_extra_inputs,
    compute_embed_diff_retain_loss,
    compute_kl_retain_loss,
    compute_npo_loss,
    compute_rmu_forget_loss,
    compute_simnpo_loss,
    compute_undial_loss,
)
from cleanslate.fsid.training.modeling import load_model, load_tokenizer

logger = get_logger(__name__, log_level="INFO")


def _get_unique_module(model: Any, regex: str) -> Any:
    """Return the unique named sub-module whose qualified name fullmatches `regex`."""
    pat = re.compile(regex)
    matches = [(name, mod) for name, mod in model.named_modules() if pat.fullmatch(name)]
    if not matches:
        raise ValueError(
            f"No module matches {regex!r}. Sample names: {[n for n, _ in list(model.named_modules())[:5]]}"
        )
    if len(matches) > 1:
        raise ValueError(f"Regex {regex!r} matched {len(matches)} modules: {[n for n, _ in matches]}")
    return matches[0][1]


def _save_resolved_config(config: TrainerConfig, run_name: str, run_dir: Path) -> None:
    snapshot = asdict(config)
    snapshot["_run_name"] = run_name
    snapshot["_run_dir"] = str(run_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(snapshot, default_flow_style=False, sort_keys=False))


def _setup_file_logging(log_path: Path) -> logging.FileHandler:
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger(__name__).addHandler(fh)
    return fh


class CleanSlateUnlearnTrainer(Trainer):
    """HF Trainer subclass with objective-specific loss dispatch.

    Objectives: ga | npo | simnpo | rmu | undial. For objectives that need a frozen
    reference model, `ref_model` is loaded from disk after Trainer init and prepared
    via the accelerator (avoids deepcopy that materializes all FSDP shards on every rank).
    """

    def __init__(
        self,
        objective: str,
        *,
        ref_model: Any | None = None,
        npo_beta: float = 0.1,
        simnpo_delta: float = 0.0,
        forget_weight: float = 1.0,
        retain_weight: float = 1.0,
        retain_loss_type: str = "nll",
        rmu_steering_coeff: float = 2.0,
        rmu_module_regex: str = r"model\.layers\.7",
        rmu_control_vec_seed: int = 0,
        rmu_control_vec_path: str | None = None,
        undial_beta: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.objective: str = objective
        self.ref_model: Any = ref_model
        self.npo_beta: float = npo_beta
        self.simnpo_delta: float = simnpo_delta
        self.forget_weight: float = forget_weight
        self.retain_weight: float = retain_weight
        self.retain_loss_type: str = retain_loss_type
        self.rmu_steering_coeff: float = rmu_steering_coeff
        self.rmu_module_regex: str = rmu_module_regex
        self.rmu_control_vec_seed: int = rmu_control_vec_seed
        self.rmu_control_vec_path: str | None = rmu_control_vec_path
        self.undial_beta: float = undial_beta
        # RMU state — populated lazily on first RMU step so the (FSDP-wrapped) policy
        # and ref models exist.
        self._rmu_model_module: Any | None = None
        self._rmu_ref_module: Any | None = None
        self._rmu_control_vec: torch.Tensor | None = None

    @staticmethod
    def prepare_ref_model(config: TrainerConfig, accelerator: Any, policy_dtype: torch.dtype) -> Any:
        """Load a frozen reference model from disk and prepare it through the accelerator.

        Loading from disk (vs deepcopy) keeps each rank lean until FSDP re-shards.
        Explicit cast to policy_dtype prevents silent fp32/bf16 drift in log-prob subtraction.
        """
        ref = load_model(config).to(dtype=policy_dtype)
        for p in ref.parameters():
            p.requires_grad = False
        ref = accelerator.prepare_model(ref, evaluation_mode=True)
        ref.eval()
        return ref

    def _ensure_rmu_state(self, model: Any) -> None:
        """Materialize RMU state (module handles, control_vec) once."""
        if self._rmu_model_module is None:
            base = getattr(model, "module", model)
            self._rmu_model_module = _get_unique_module(base, self.rmu_module_regex)
        if self._rmu_ref_module is None:
            if self.ref_model is None:
                raise ValueError("rmu requires a ref_model for EMBED_DIFF retain loss")
            ref_base = getattr(self.ref_model, "module", self.ref_model)
            self._rmu_ref_module = _get_unique_module(ref_base, self.rmu_module_regex)
        if self._rmu_control_vec is None:
            self._rmu_control_vec = self._initialize_control_vec(int(model.config.hidden_size))

    def _initialize_control_vec(self, dim: int) -> torch.Tensor:
        """Produce the RMU random control vector, either from disk or freshly sampled."""
        if self.rmu_control_vec_path is not None:
            loaded: torch.Tensor = torch.load(self.rmu_control_vec_path, map_location="cpu")
            if loaded.shape[-1] != dim:
                raise ValueError(
                    f"rmu_control_vec_path {self.rmu_control_vec_path!r}: "
                    f"dim {loaded.shape[-1]} != hidden_size {dim}"
                )
            vec = loaded.float()
            if vec.dim() == 1:
                vec = vec.view(1, 1, -1)
            elif vec.dim() == 2:
                vec = vec.unsqueeze(0)
            logger.info(
                "RMU control_vec loaded from %s: dim=%d, norm=%.6f",
                self.rmu_control_vec_path, dim, float(vec.norm().item()),
            )
            return vec

        # Rank-count-invariant target norm. Under DDP/FSDP with N ranks each generating
        # an independent random unit vector, gradient averaging across ranks ≈ 1-GPU
        # training against the mean vector; with zero-mean samples (randn) the mean
        # of N i.i.d. unit vectors has expected norm ≈ 1/sqrt(N) in high-d. Scale each
        # rank by sqrt(world_size) so the effective post-average target norm stays
        # rmu_steering_coeff regardless of N. (torch.rand would concentrate in the
        # positive orthant and invalidate this cancellation.)
        world_size = int(getattr(self.args, "world_size", 1) or 1)
        effective_coeff = self.rmu_steering_coeff * (world_size ** 0.5)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        g = torch.Generator()
        g.manual_seed(int(self.rmu_control_vec_seed) + rank)
        v = torch.randn(1, 1, dim, generator=g)
        fresh_vec = (v / v.norm()) * effective_coeff
        logger.info(
            "RMU control_vec init: dim=%d seed=%d rank=%d world_size=%d per_rank_norm=%.6f",
            dim, self.rmu_control_vec_seed, rank, world_size, float(fresh_vec.norm().item()),
        )
        return torch.as_tensor(fresh_vec)

    @override
    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        forget_inputs = inputs["forget"]
        retain_inputs = inputs.get("retain")
        f_ids, f_mask, f_labels = forget_inputs["input_ids"], forget_inputs["attention_mask"], forget_inputs["labels"]

        if self.objective == "rmu":
            if retain_inputs is None:
                raise ValueError("rmu requires a retain batch")
            self._ensure_rmu_state(model)
            assert self._rmu_model_module is not None
            assert self._rmu_ref_module is not None
            assert self._rmu_control_vec is not None

            forget_loss, _ = compute_rmu_forget_loss(
                model, f_ids, f_mask, f_labels, self._rmu_model_module, self._rmu_control_vec,
            )
            r_ids = retain_inputs["input_ids"]
            r_mask = retain_inputs["attention_mask"]
            r_labels = retain_inputs["labels"]
            retain_loss = compute_embed_diff_retain_loss(
                model, self.ref_model, r_ids, r_mask, r_labels,
                self._rmu_model_module, self._rmu_ref_module,
            )
            return self._combine_losses(forget_loss, retain_loss, return_outputs, f_ids.device)

        forget_loss = self._compute_forget_loss(model, f_ids, f_mask, f_labels)
        retain_loss = self._compute_retain_loss(model, retain_inputs, f_ids.device)
        return self._combine_losses(forget_loss, retain_loss, return_outputs, f_ids.device)

    def _compute_forget_loss(
        self, model: Any, f_ids: torch.Tensor, f_mask: torch.Tensor, f_labels: torch.Tensor,
    ) -> torch.Tensor:
        match self.objective:
            case "ga":
                outputs = model(
                    input_ids=f_ids,
                    attention_mask=f_mask,
                    labels=f_labels,
                    **_build_extra_inputs(model, f_ids),
                )
                ga_loss: torch.Tensor = outputs.loss
                return -ga_loss
            case "npo":
                return compute_npo_loss(model, self.ref_model, f_ids, f_mask, f_labels, self.npo_beta)
            case "simnpo":
                return compute_simnpo_loss(
                    model, f_ids, f_mask, f_labels, self.npo_beta, self.simnpo_delta
                )
            case "undial":
                if self.ref_model is None:
                    raise ValueError("undial requires a frozen ref_model (teacher)")
                return compute_undial_loss(
                    model, self.ref_model, f_ids, f_mask, f_labels, self.undial_beta
                )
            case obj:
                raise ValueError(f"Unknown objective: {obj}")

    def _compute_retain_loss(
        self, model: Any, retain_inputs: dict[str, Any] | None, device: torch.device,
    ) -> torch.Tensor:
        if retain_inputs is None or self.retain_weight == 0:
            return torch.tensor(0.0, device=device)
        r_ids = retain_inputs["input_ids"]
        r_mask = retain_inputs["attention_mask"]
        r_labels = retain_inputs["labels"]
        if self.retain_loss_type == "kl" and self.ref_model is not None:
            return compute_kl_retain_loss(model, self.ref_model, r_ids, r_mask, r_labels)
        out = model(
            input_ids=r_ids,
            attention_mask=r_mask,
            labels=r_labels,
            **_build_extra_inputs(model, r_ids),
        )
        retain_loss: torch.Tensor = out.loss
        return retain_loss

    def _combine_losses(
        self,
        forget_loss: torch.Tensor,
        retain_loss: torch.Tensor,
        return_outputs: bool,
        _device: torch.device,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        total = self.forget_weight * forget_loss + self.retain_weight * retain_loss
        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log(
                {
                    "forget_loss": float(forget_loss.detach().item()),
                    "retain_loss": float(retain_loss.detach().item()),
                }
            )
        if return_outputs:
            return total, {"forget_loss": forget_loss, "retain_loss": retain_loss}
        return total


def train_unlearning(config: TrainerConfig) -> dict[str, Any]:
    """Entry point: train an unlearned model from a resolved TrainerConfig."""
    config.validate()

    run_name = config.make_run_name()
    training_args = config.to_training_arguments(run_name=run_name)
    run_dir = Path(training_args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config.model_name)
    pad_id = int(tokenizer.pad_token_id)

    logger.info("Loading forget set from %s", config.forget_path)
    forget_rows = load_local_records(config.forget_path)
    rule_name = forget_rows[0].get("rule_name", "<unknown>")
    forget_pairs = build_forget_pairs(forget_rows, tokenizer)
    if not forget_pairs:
        raise ValueError(f"No forget training pairs built from {config.forget_path!r}")
    logger.info("Built %d forget pairs (rule_name=%s)", len(forget_pairs), rule_name)

    retain_rows: list[dict[str, Any]] = []
    retain_examples: list[dict[str, Any]] = []
    if config.retain_path:
        logger.info("Loading retain set from %s", config.retain_path)
        retain_rows = load_local_records(config.retain_path)
        retain_examples = tokenize_text_rows(retain_rows, tokenizer)
        logger.info("Prepared %d retain sequences", len(retain_examples))

    forget_dataset = ForgetPairDataset(forget_pairs)
    retain_dataset = TextSequenceDataset(retain_examples) if retain_examples else None
    train_dataset = ForgetRetainDataset(forget_dataset, retain_dataset)

    logger.info("Loading model: %s", config.model_name)
    model = load_model(config)

    train_collator = UnlearnDataCollator(pad_token_id=pad_id)

    trainer = CleanSlateUnlearnTrainer(
        objective=config.objective,
        npo_beta=config.npo_beta,
        simnpo_delta=config.simnpo_delta,
        forget_weight=config.forget_weight,
        retain_weight=config.retain_weight,
        retain_loss_type=config.retain_loss_type,
        rmu_steering_coeff=config.rmu_steering_coeff,
        rmu_module_regex=config.rmu_module_regex,
        rmu_control_vec_seed=config.rmu_control_vec_seed,
        undial_beta=config.undial_beta,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_collator,
        processing_class=tokenizer,
    )

    if config.objective in ("npo", "simnpo") and config.retain_weight == 0:
        logger.warning(
            "%s with retain_weight=0: no constraint on non-forget data; model may degrade globally",
            config.objective.upper(),
        )

    file_handler: logging.FileHandler | None = None
    if trainer.is_world_process_zero():
        file_handler = _setup_file_logging(run_dir / "train.log")
        _save_resolved_config(config, run_name, run_dir)
        logger.info(
            "FSID train: model=%s objective=%s run_dir=%s",
            config.model_name, config.objective, run_dir,
        )

    needs_ref = config.objective in ("npo", "rmu", "undial") or config.retain_loss_type in ("kl", "embed_diff")
    if needs_ref:
        assert trainer.model is not None
        policy_dtype = next(trainer.model.parameters()).dtype
        logger.info("Creating frozen reference model for %s (dtype=%s)", config.objective, policy_dtype)
        trainer.ref_model = CleanSlateUnlearnTrainer.prepare_ref_model(
            config, trainer.accelerator, policy_dtype
        )

    run_start = time.time()
    logger.info("Starting training: objective=%s epochs=%d", config.objective, config.num_epochs)
    trainer.train()
    run_seconds = time.time() - run_start

    trainer.accelerator.wait_for_everyone()
    trainer.save_model()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info("Saved checkpoint to %s", training_args.output_dir)

    summary: dict[str, Any] = {
        "run_name": run_name,
        "run_dir": training_args.output_dir,
        "config": asdict(config),
        "num_forget_rows": len(forget_rows),
        "num_forget_pairs": len(forget_pairs),
        "num_retain_rows": len(retain_rows),
        "num_retain_sequences": len(retain_examples),
        "num_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "num_total_params": sum(p.numel() for p in model.parameters()),
        "run_seconds": run_seconds,
        "checkpoint_dir": training_args.output_dir,
        "is_main_process": trainer.is_world_process_zero(),
    }
    if trainer.is_world_process_zero():
        (Path(training_args.output_dir) / "train_summary.json").write_text(json.dumps(summary, indent=2))
        logger.info("Training complete (%.1fs)", run_seconds)

        if config.push_to_hub:
            assert config.hub_repo_id is not None
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(
                repo_id=config.hub_repo_id,
                repo_type="model",
                private=config.hub_private,
                exist_ok=True,
            )
            logger.info("Pushing checkpoint dir %s to %s", training_args.output_dir, config.hub_repo_id)
            api.upload_folder(
                folder_path=training_args.output_dir,
                repo_id=config.hub_repo_id,
                repo_type="model",
                commit_message=f"upload {run_name}",
            )
            summary["hub_repo_id"] = config.hub_repo_id
            logger.info("Pushed to https://huggingface.co/%s", config.hub_repo_id)

    if file_handler is not None:
        logging.getLogger(__name__).removeHandler(file_handler)
        file_handler.close()

    return summary
