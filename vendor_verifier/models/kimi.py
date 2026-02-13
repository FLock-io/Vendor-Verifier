import re
from typing import Any, Dict, List, Optional

from loguru import logger

from .base import ModelAdapter

# Kimi-specific tool call markers (for raw completions endpoint)
TOOL_CALLS_BEGIN = "<|tool_calls_section_begin|>"
TOOL_CALLS_END = "<|tool_calls_section_end|>"
TOOL_CALL_BEGIN = "<|tool_call_begin|>"
TOOL_CALL_ARG_BEGIN = "<|tool_call_argument_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"

ROLE_INPUT = "_input"
ROLE_SYSTEM = "system"


class KimiModel(ModelAdapter):
    """Adapter for Kimi models (kimi-k2, kimi-k2.5, etc.)."""

    model_patterns = ["kimi", "moonshot"]

    def __init__(
        self,
        model_name: str,
        tokenizer_model: Optional[str] = None,
        use_raw_completions: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.tokenizer_model = tokenizer_model
        self.use_raw_completions = use_raw_completions
        self._tokenizer = None

    def transform_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Kimi-specific transformations:
        1. Convert _input role to system role.
        2. Inject reasoning_content for assistant messages with tool_calls
           (required by thinking models like kimi-k2.5).
        """
        for message in messages:
            if message.get("role") == ROLE_INPUT:
                message["role"] = ROLE_SYSTEM
            if (
                message.get("role") == "assistant"
                and message.get("tool_calls")
                and "reasoning_content" not in message
            ):
                message["reasoning_content"] = " "
        return messages

    def extract_tool_calls_from_text(
        self, text: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls from Kimi's <|tool_call_*|> marker format."""
        if TOOL_CALLS_BEGIN not in text:
            return None

        section_pattern = (
            rf"{re.escape(TOOL_CALLS_BEGIN)}(.*?){re.escape(TOOL_CALLS_END)}"
        )
        tool_calls_sections = re.findall(section_pattern, text, re.DOTALL)
        if not tool_calls_sections:
            return None

        func_call_pattern = (
            rf"{re.escape(TOOL_CALL_BEGIN)}\s*"
            r"(?P<tool_call_id>[\w\.]+:\d+)\s*"
            rf"{re.escape(TOOL_CALL_ARG_BEGIN)}\s*"
            r"(?P<function_arguments>.*?)\s*"
            rf"{re.escape(TOOL_CALL_END)}"
        )

        tool_calls = []
        for match in re.finditer(
            func_call_pattern, tool_calls_sections[0], re.DOTALL
        ):
            function_id = match.group("tool_call_id")
            function_args = match.group("function_arguments")

            try:
                function_name = function_id.split(".")[1].split(":")[0]
            except IndexError:
                logger.warning(f"Unable to parse function_id: {function_id}")
                continue

            tool_calls.append(
                {
                    "id": function_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": function_args,
                    },
                }
            )

        return tool_calls if tool_calls else None

    def get_tokenizer(self) -> Optional[Any]:
        if self.use_raw_completions and self.tokenizer_model:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_model, trust_remote_code=True
                )
            return self._tokenizer
        return None

    def supports_raw_completions(self) -> bool:
        return self.use_raw_completions
