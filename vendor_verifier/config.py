import httpx

DEFAULT_CONCURRENCY = 5
DEFAULT_TIMEOUT = 600
DEFAULT_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT_BASE_DELAY = 2.0
DEFAULT_RATE_LIMIT_MAX_DELAY = 60.0
DEFAULT_OUTPUT_FILE = "results.jsonl"
DEFAULT_SUMMARY_FILE = "summary.json"

# Unlimited read timeout for streaming; allow slow model output.
HTTPX_STREAM_TIMEOUT = httpx.Timeout(timeout=None, connect=60.0)
