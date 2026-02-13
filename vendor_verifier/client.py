import asyncio
import json
import random
from typing import Any, Dict, List, Optional, Tuple

import httpx
from httpcore import ConnectError as HttpcoreConnectError
from httpcore import ConnectTimeout as HttpcoreConnectTimeout
from httpcore import ReadError as HttpcoreReadError
from httpcore import RemoteProtocolError
from loguru import logger
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from .config import (
    DEFAULT_RATE_LIMIT_BASE_DELAY,
    DEFAULT_RATE_LIMIT_MAX_DELAY,
    HTTPX_STREAM_TIMEOUT,
)
from .models.base import ModelAdapter

RETRYABLE_READ_ERRORS = (
    HttpcoreReadError,
    RemoteProtocolError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)


def _compute_backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter, capped at DEFAULT_RATE_LIMIT_MAX_DELAY."""
    delay = min(
        DEFAULT_RATE_LIMIT_BASE_DELAY * (2**attempt), DEFAULT_RATE_LIMIT_MAX_DELAY
    )
    return delay + (delay * random.uniform(0, 0.25))


def _is_retryable_exception(e: BaseException) -> bool:
    """Return True for errors that should be retried indefinitely."""
    if isinstance(e, RateLimitError):
        return True
    if isinstance(e, APIStatusError):
        return getattr(e, "status_code", None) == 429
    if isinstance(e, (APIConnectionError, APITimeoutError, *RETRYABLE_READ_ERRORS)):
        return True
    return False


def serialize_error(e: BaseException) -> Dict[str, str]:
    """Serialize exception for JSONL output."""
    return {
        "error_type": type(e).__name__,
        "error_message": str(e),
        "error": str(e),
    }


class ApiClient:
    """Async API client with retry, rate limiting, and streaming support."""

    def __init__(
        self,
        adapter: ModelAdapter,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        concurrency: int = 5,
        timeout: int = 600,
        max_retries: int = 3,
        extra_body: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.adapter = adapter
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.http_client = httpx.AsyncClient(
            timeout=HTTPX_STREAM_TIMEOUT,
            limits=httpx.Limits(
                max_connections=concurrency * 2,
                max_keepalive_connections=concurrency,
            ),
        )
        self.openai_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client=self.http_client,
        )

    async def close(self) -> None:
        try:
            await self.openai_client.close()
        except Exception as e:
            logger.warning(f"Error closing AsyncOpenAI client: {e}")
        try:
            await self.http_client.aclose()
        except Exception as e:
            logger.warning(f"Error closing httpx client: {e}")

    def prepare_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess request: apply model overrides and adapter transformations."""
        req = request.copy()

        if "messages" in req:
            req["messages"] = self.adapter.transform_messages(req["messages"])

        if self.model:
            req["model"] = self.model

        if self.temperature is not None:
            req["temperature"] = self.temperature
        if self.max_tokens is not None:
            req["max_tokens"] = self.max_tokens

        # Ensure streaming usage is included.
        use_raw = self.adapter.supports_raw_completions()
        if req.get("stream", False) and not use_raw:
            so = req.get("stream_options")
            if not isinstance(so, dict):
                so = {}
            so.setdefault("include_usage", True)
            req["stream_options"] = so

        # Convert messages to prompt if using raw completions endpoint
        tokenizer = self.adapter.get_tokenizer()
        if use_raw and tokenizer:
            req["prompt"] = tokenizer.apply_chat_template(
                req["messages"],
                tokenize=False,
                tools=req.get("tools", None),
                add_generation_prompt=True,
            )
            req.pop("messages")
            if "tools" in req:
                req.pop("tools")

        return req

    async def send_request(
        self, request: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Send a single request; retry indefinitely on retryable errors."""
        attempt = 0
        while True:
            try:
                async with self.semaphore:
                    return await self._send_once(request)
            except Exception as e:
                if not _is_retryable_exception(e):
                    logger.error(f"Request failed: {e}")
                    return "failed", serialize_error(e)

                delay = _compute_backoff_delay(attempt)
                attempt += 1
                logger.warning(
                    f"Retryable error ({type(e).__name__}), attempt {attempt}, "
                    f"retrying in {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)

    def _raw_request_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _raise_for_api_status(self, resp: httpx.Response) -> None:
        try:
            body = resp.json()
        except Exception:
            body = {"error": {"message": resp.text}}
        msg = f"Error code: {resp.status_code} - {body}"
        if resp.status_code == 429:
            raise RateLimitError(message=msg, response=resp, body=body)
        raise APIStatusError(message=msg, response=resp, body=body)

    async def _send_once(
        self, request: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Perform one network attempt (may raise)."""
        if request.get("stream", False):
            return await self._handle_stream_request(request)

        if self.adapter.supports_raw_completions():
            response = await self.openai_client.completions.create(
                **request, extra_body=self.extra_body
            )
            return "success", response.model_dump()

        # Chat completions via raw httpx to preserve extra fields
        body = {**request, **self.extra_body}
        resp = await self.http_client.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers=self._raw_request_headers(),
        )
        if resp.status_code >= 400:
            self._raise_for_api_status(resp)
        return "success", resp.json()

    async def _handle_stream_request(
        self, request: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Accumulate a streaming response into a non-stream response dict."""
        if self.adapter.supports_raw_completions():
            stream = await self.openai_client.completions.create(
                **request, extra_body=self.extra_body
            )
            return await self._accumulate_sdk_stream(stream, request)

        # Chat completions streaming via raw httpx
        body = {**request, **self.extra_body}

        request_id = None
        created = None
        full_content: List[str] = []
        full_reasoning_content: List[str] = []
        tool_calls: Dict[int, Dict[str, Any]] = {}
        finish_reason = None
        usage = None

        async with self.http_client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=body,
            headers=self._raw_request_headers(),
        ) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                self._raise_for_api_status(resp)

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse SSE event: {data[:200]}")
                    continue

                if event.get("id"):
                    request_id = event["id"]
                if event.get("created"):
                    created = event["created"]

                choices = event.get("choices")
                if not choices:
                    if event.get("usage"):
                        usage = event["usage"]
                    continue

                choice = choices[0]
                delta = choice.get("delta") or {}

                if delta.get("content"):
                    full_content.append(delta["content"])
                if delta.get("reasoning_content"):
                    full_reasoning_content.append(delta["reasoning_content"])
                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls:
                            tool_calls[idx] = {
                                "id": tc.get("id"),
                                "type": tc.get("type", "function"),
                                "function": {"name": "", "arguments": ""},
                            }
                        func = tc.get("function") or {}
                        if func.get("name"):
                            tool_calls[idx]["function"]["name"] = func["name"]
                        if func.get("arguments"):
                            tool_calls[idx]["function"]["arguments"] += func[
                                "arguments"
                            ]

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                if choice.get("usage"):
                    usage = choice["usage"]
                if event.get("usage"):
                    usage = event["usage"]

        content_text = "".join(full_content)
        reasoning_content_text = (
            "".join(full_reasoning_content) if full_reasoning_content else None
        )
        tool_calls_list = list(tool_calls.values()) if tool_calls else None

        message_dict: Dict[str, Any] = {
            "role": "assistant",
            "content": content_text,
            "tool_calls": tool_calls_list,
        }
        if reasoning_content_text:
            message_dict["reasoning_content"] = reasoning_content_text

        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": request.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": finish_reason or "stop",
                }
            ],
            "usage": usage,
        }
        return "success", response

    async def _accumulate_sdk_stream(
        self, stream: Any, request: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Accumulate a stream from the OpenAI SDK (raw completions) into a response dict."""
        request_id = None
        created = None
        full_content: List[str] = []
        full_reasoning_content: List[str] = []
        tool_calls: Dict[int, Dict[str, Any]] = {}
        finish_reason = None
        usage = None

        async for event in stream:
            if getattr(event, "id", None):
                request_id = event.id
            if getattr(event, "created", None):
                created = event.created

            if not getattr(event, "choices", None):
                logger.warning("Empty choices in stream event")
                continue

            choice = event.choices[0]

            if getattr(choice, "delta", None):
                delta = choice.delta
                if getattr(delta, "content", None):
                    full_content.append(delta.content)
                if getattr(delta, "reasoning_content", None):
                    full_reasoning_content.append(delta.reasoning_content)
                if getattr(delta, "tool_calls", None):
                    self._accumulate_tool_calls(delta.tool_calls, tool_calls)
            elif getattr(choice, "text", None):
                full_content.append(choice.text)

            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            if getattr(choice, "usage", None):
                usage = choice.usage

        if usage is not None and hasattr(usage, "model_dump"):
            usage = usage.model_dump()

        content_text = "".join(full_content)
        reasoning_content_text = (
            "".join(full_reasoning_content) if full_reasoning_content else None
        )

        # Let the adapter try to extract tool calls from raw text
        extracted = self.adapter.extract_tool_calls_from_text(content_text)
        if extracted:
            tool_calls = {i: tc for i, tc in enumerate(extracted)}
            finish_reason = "tool_calls"

        tool_calls_list = list(tool_calls.values()) if tool_calls else None

        message_dict: Dict[str, Any] = {
            "role": "assistant",
            "content": content_text,
            "tool_calls": tool_calls_list,
        }
        if reasoning_content_text:
            message_dict["reasoning_content"] = reasoning_content_text

        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": request.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": finish_reason or "stop",
                }
            ],
            "usage": usage,
        }
        return "success", response

    def _accumulate_tool_calls(
        self,
        delta_tool_calls: List[Any],
        tool_calls: Dict[int, Dict[str, Any]],
    ) -> None:
        for tc in delta_tool_calls:
            idx = tc.index if tc.index is not None else 0
            if idx not in tool_calls:
                tool_calls[idx] = {
                    "id": tc.id if hasattr(tc, "id") else None,
                    "type": tc.type if hasattr(tc, "type") else "function",
                    "function": {"name": "", "arguments": ""},
                }
            if hasattr(tc, "function") and tc.function:
                if hasattr(tc.function, "name") and tc.function.name:
                    tool_calls[idx]["function"]["name"] = tc.function.name
                if hasattr(tc.function, "arguments") and tc.function.arguments:
                    tool_calls[idx]["function"]["arguments"] += (
                        tc.function.arguments
                    )
