"""Backend registry and factory."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

from cleanslate.backends.base import InferenceBackend


@dataclass(frozen=True)
class BackendSpec:
    module: str
    class_name: str
    accepts_device: bool = False


BACKEND_REGISTRY: dict[str, BackendSpec] = {
    "huggingface": BackendSpec(
        module="cleanslate.backends.huggingface",
        class_name="HuggingFaceBackend",
        accepts_device=True,
    ),
    "sglang": BackendSpec(
        module="cleanslate.backends.sglang_backend",
        class_name="SGLangBackend",
        accepts_device=False,
    ),
    "vllm": BackendSpec(
        module="cleanslate.backends.vllm_backend",
        class_name="VLLMBackend",
        accepts_device=False,
    ),
}


def list_backends() -> list[str]:
    return sorted(BACKEND_REGISTRY.keys())


def create_backend(
    name: str,
    model_name: str,
    *,
    device: str | None = None,
    **kwargs: Any,
) -> InferenceBackend:
    if name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {name}. Available: {list_backends()}")

    spec = BACKEND_REGISTRY[name]
    if device is not None:
        if spec.accepts_device:
            kwargs["device"] = device
        else:
            raise ValueError(f"Backend '{name}' does not accept device overrides.")

    module = importlib.import_module(spec.module)
    cls = getattr(module, spec.class_name)
    return cls(model_name, **kwargs)
