from .base import ModelAdapter


class GenericModel(ModelAdapter):
    """
    Works with any OpenAI-compatible API out of the box.
    Uses standard /v1/chat/completions with no transformations.
    """

    model_patterns = []  # Fallback — used when no other adapter matches
