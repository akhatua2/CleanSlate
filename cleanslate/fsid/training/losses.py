"""Loss utilities for FSID unlearning, ported from open-unlearning patterns."""

import inspect
from typing import Any

import torch
import torch.nn.functional as F

# Per-model-class cache: does this causal LM require token_type_ids during training?
# Only Gemma3's multimodal forward (Gemma3ForConditionalGeneration / Gemma3Model) does —
# it raises when token_type_ids is None during training. We pass zeros to mark every
# position as a text token. Cached by class (not instance) to avoid re-introspecting
# every forward.
_TOKEN_TYPE_CACHE: dict[type, bool] = {}


def _model_needs_token_type_ids(model: Any) -> bool:
    cls = type(model)
    if (cached := _TOKEN_TYPE_CACHE.get(cls)) is not None:
        return cached
    sig = inspect.signature(model.forward)
    needs = "token_type_ids" in sig.parameters and getattr(model.config, "model_type", "") == "gemma3"
    _TOKEN_TYPE_CACHE[cls] = needs
    return needs


def _build_extra_inputs(model: Any, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    if _model_needs_token_type_ids(model):
        return {"token_type_ids": torch.zeros_like(input_ids)}
    return {}


def compute_batch_nll(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Per-sample summed NLL over non-masked tokens. Shape: (batch_size,)."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **_build_extra_inputs(model, input_ids),
    )
    logits = outputs.logits[:, :-1, :].float()
    target = labels[:, 1:]
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(target.shape)
    mask = target != -100
    return (per_token * mask).sum(dim=1)


def compute_npo_loss(
    model: Any,
    ref_model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """NPO: -2/β * logsigmoid(-β * (logπ_θ − logπ_ref)). Summed (not averaged) log-probs."""
    policy_log_probs = -compute_batch_nll(model, input_ids, attention_mask, labels)
    with torch.no_grad():
        ref_log_probs = -compute_batch_nll(ref_model, input_ids, attention_mask, labels)
    return -(2.0 / beta) * F.logsigmoid(-beta * (policy_log_probs - ref_log_probs)).mean()


def compute_simnpo_loss(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    delta: float,
) -> torch.Tensor:
    """SimNPO — line-for-line port of open-unlearning/src/trainer/unlearn/simnpo.py."""
    per_sample_nll = compute_batch_nll(model, input_ids, attention_mask, labels)
    loss_mask = labels != -100
    avg_nll = per_sample_nll / loss_mask.sum(dim=-1) - delta
    return -F.logsigmoid(beta * avg_nll).mean() * 2 / beta


def compute_undial_loss(
    model: Any,
    ref_model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """UNDIAL: self-distill against a teacher whose logit for the correct next-token is
    knocked down by β before softmax.

    Line-for-line port of open-unlearning/src/trainer/utils.py::compute_undial_loss.
    """
    extras = _build_extra_inputs(model, input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, **extras)
    logits = outputs.logits

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    with torch.no_grad():
        teacher = ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **_build_extra_inputs(ref_model, input_ids),
        ).logits
    shift_teacher = teacher[..., :-1, :].contiguous()

    # OU indexes with raw `shift_labels` (contains -100 at masked positions). Python's
    # negative indexing wraps -100 to vocab[V-100], so the teacher distribution at
    # masked positions has vocab[V-100] knocked down instead. Preserved for byte-parity.
    mask = torch.zeros_like(shift_teacher)
    b_idx = torch.arange(mask.shape[0], device=mask.device).view(-1, 1, 1)
    s_idx = torch.arange(mask.shape[1], device=mask.device).view(1, -1, 1)
    mask[b_idx, s_idx, shift_labels.unsqueeze(-1)] = 1.0

    soft_label = F.softmax(shift_teacher - mask * beta, dim=-1)
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        soft_label.reshape(-1, soft_label.size(-1)),
        reduction="none",
    )
    return loss.mean()


def forward_with_cache(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    module: Any,
    no_grad: bool = True,
) -> tuple[torch.Tensor, Any]:
    """Forward while caching the output of `module` via a forward hook.

    Raises if the hook never fires (wrong module reference / skipped by the model).
    """
    cache: list[torch.Tensor] = []

    def _hook(_mod: Any, _inp: Any, out: Any) -> None:
        cache.append(out[0] if isinstance(out, tuple) else out)

    handle = module.register_forward_hook(_hook)
    try:
        with torch.set_grad_enabled(not no_grad):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **_build_extra_inputs(model, input_ids),
            )
    finally:
        handle.remove()
    if not cache:
        raise RuntimeError(f"forward_with_cache: hook on {module} never fired")
    return cache[0], outputs


def compute_activation_mse(
    act1: torch.Tensor,
    act2: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sample MSE averaged over non-ignored tokens.

    Byte-for-byte port of open-unlearning/src/trainer/unlearn/rmu.py::compute_activation_loss.
    OU's (b,)/(b,1) broadcast inflates to (b,b) before .mean(); we preserve this to stay
    numerically equivalent to OU (verified via scripts/verify_tofu.py).
    """
    sq = F.mse_loss(act1, act2, reduction="none")
    expanded = loss_mask.unsqueeze(-1).expand_as(sq)
    squared_diff_sum = (sq * expanded).mean(dim=2).sum(dim=1)
    num_tokens = loss_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    return (squared_diff_sum / num_tokens).mean()


def compute_rmu_forget_loss(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    module: Any,
    control_vec: torch.Tensor,
) -> tuple[torch.Tensor, Any]:
    """RMU forget loss: MSE(module_activation, control_vec) on the forget batch."""
    act, outputs = forward_with_cache(model, input_ids, attention_mask, module, no_grad=False)
    if control_vec.shape[-1] != act.shape[-1]:
        raise ValueError(
            f"RMU control_vec last dim {control_vec.shape[-1]} != activation last dim {act.shape[-1]}"
        )
    cv = control_vec.to(dtype=act.dtype, device=act.device).expand_as(act)
    return compute_activation_mse(act, cv, labels != -100), outputs


def compute_embed_diff_retain_loss(
    model: Any,
    ref_model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    model_module: Any,
    ref_module: Any,
) -> torch.Tensor:
    """EMBED_DIFF retain: MSE(policy_activation, ref_activation) at the target layer."""
    act_m, _ = forward_with_cache(model, input_ids, attention_mask, model_module, no_grad=False)
    act_r, _ = forward_with_cache(ref_model, input_ids, attention_mask, ref_module, no_grad=True)
    if act_m.shape != act_r.shape:
        raise ValueError(f"EMBED_DIFF shape mismatch: policy {tuple(act_m.shape)} vs ref {tuple(act_r.shape)}")
    return compute_activation_mse(act_m, act_r.to(act_m.device), labels != -100)


def compute_kl_retain_loss(
    model: Any,
    ref_model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """KL(ref || model) retain loss on non-masked tokens."""
    extras = _build_extra_inputs(model, input_ids)
    with torch.no_grad():
        ref_logits = ref_model(input_ids=input_ids, attention_mask=attention_mask, **extras).logits[:, :-1, :].float()
    model_logits = model(input_ids=input_ids, attention_mask=attention_mask, **extras).logits[:, :-1, :].float()

    target = labels[:, 1:]
    mask = (target != -100).float()
    ref_probs = F.softmax(ref_logits, dim=-1)
    model_log_probs = F.log_softmax(model_logits, dim=-1)
    per_token_kl = F.kl_div(model_log_probs, ref_probs, reduction="none", log_target=False).sum(dim=-1)
    return (per_token_kl * mask).sum() / mask.sum().clamp(min=1)
