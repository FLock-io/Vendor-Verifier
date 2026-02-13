import argparse
import asyncio
import json
import os

from loguru import logger

from .client import ApiClient
from .config import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_SUMMARY_FILE,
    DEFAULT_TIMEOUT,
)
from .models import get_adapter
from .validator import ToolCallsValidator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LLM Tool Calls Validator\n\n"
            "Validate LLM tool call functionality via OpenAI-compatible API "
            "with concurrency support and optional incremental re-run.\n"
            "Each line in the test set file must be a complete LLM request body (JSON format)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "file_path",
        help="Test set file path (JSONL format)",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="API endpoint URL, e.g., https://api.openai.com/v1",
    )
    parser.add_argument(
        "--api-key",
        help="API key (can also be set via OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name, e.g., gpt-4o, kimi-k2-0905-preview",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        help="Force a specific model adapter (e.g., kimi, generic). "
        "If not set, auto-detected from model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature (overrides request temperature)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum token count (overrides request max_tokens)",
    )
    parser.add_argument(
        "--extra-body",
        type=str,
        help="Extra request body parameters (JSON string)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Maximum concurrent requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Detailed results output file path (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--summary",
        default=DEFAULT_SUMMARY_FILE,
        help=f"Aggregated summary output file path (default: {DEFAULT_SUMMARY_FILE})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Number of retries on failure (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode: only rerun failed or new requests",
    )

    # Kimi-specific arguments
    parser.add_argument(
        "--use-raw-completions",
        action="store_true",
        help="Use /v1/completions endpoint (Kimi-specific, requires --tokenizer-model)",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        help="HuggingFace tokenizer model name for raw completions (Kimi-specific)",
    )

    return parser


async def async_main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    extra_body = {}
    if args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse --extra-body JSON: {e}")
            return

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    # Build adapter kwargs from CLI args
    adapter_kwargs = {}
    if args.use_raw_completions:
        adapter_kwargs["use_raw_completions"] = True
    if args.tokenizer_model:
        adapter_kwargs["tokenizer_model"] = args.tokenizer_model

    # Resolve model adapter
    if args.model_type:
        adapter = get_adapter(
            args.model, force_type=args.model_type, **adapter_kwargs
        )
    else:
        adapter = get_adapter(args.model, **adapter_kwargs)

    client = ApiClient(
        adapter=adapter,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        concurrency=args.concurrency,
        timeout=args.timeout,
        max_retries=args.retries,
        extra_body=extra_body,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    async with ToolCallsValidator(
        client=client,
        output_file=args.output,
        summary_file=args.summary,
        incremental=args.incremental,
    ) as validator:
        await validator.validate_file(args.file_path)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
