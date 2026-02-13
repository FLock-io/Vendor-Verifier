from typing import Any, Dict, List, Optional


class ModelAdapter:
    """
    Base class for model-specific behavior.

    All methods have default implementations suitable for standard
    OpenAI-compatible APIs. Subclass and override only what your model needs.
    """

    # Subclasses set this for auto-discovery by model name.
    # e.g. ["kimi", "moonshot"] will match any model name containing those strings.
    model_patterns: List[str] = []

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    def transform_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Transform messages before sending to the API. Default: no-op."""
        return messages

    def extract_tool_calls_from_text(
        self, text: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Parse tool calls from raw text output (completions endpoint).
        Default: None (standard OpenAI tool_calls field is used instead).
        """
        return None

    def get_tokenizer(self) -> Optional[Any]:
        """Return a tokenizer for raw completions mode. Default: None."""
        return None

    def supports_raw_completions(self) -> bool:
        """Whether this model uses /v1/completions instead of /v1/chat/completions."""
        return False
