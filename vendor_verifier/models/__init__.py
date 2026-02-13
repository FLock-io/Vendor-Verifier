from typing import List, Optional, Type

from .base import ModelAdapter
from .generic import GenericModel
from .kimi import KimiModel

# All known adapters — order matters (first match wins).
_REGISTRY: List[Type[ModelAdapter]] = [
    KimiModel,
]

# Map of type names for --model-type override
_TYPE_MAP = {
    "kimi": KimiModel,
    "generic": GenericModel,
}


def get_adapter(
    model_name: str,
    force_type: Optional[str] = None,
    **kwargs,
) -> ModelAdapter:
    """
    Find the right adapter for a model name.

    Args:
        model_name: The model identifier (e.g. "kimi-k2-0905-preview", "gpt-4o").
        force_type: If set, skip auto-detection and use this adapter type.
        **kwargs: Extra arguments passed to the adapter constructor.

    Returns:
        An instantiated ModelAdapter.
    """
    if force_type:
        adapter_cls = _TYPE_MAP.get(force_type.lower())
        if adapter_cls is None:
            available = ", ".join(sorted(_TYPE_MAP.keys()))
            raise ValueError(
                f"Unknown model type '{force_type}'. Available: {available}"
            )
        return adapter_cls(model_name=model_name, **kwargs)

    model_lower = model_name.lower()
    for adapter_cls in _REGISTRY:
        for pattern in adapter_cls.model_patterns:
            if pattern in model_lower:
                return adapter_cls(model_name=model_name, **kwargs)

    return GenericModel(model_name=model_name, **kwargs)


def register_adapter(
    adapter_cls: Type[ModelAdapter], type_name: Optional[str] = None
) -> None:
    """Register a custom adapter."""
    _REGISTRY.insert(0, adapter_cls)
    if type_name:
        _TYPE_MAP[type_name.lower()] = adapter_cls
