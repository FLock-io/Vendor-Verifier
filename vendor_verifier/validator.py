import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import megfile
from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from .client import ApiClient
from .validation import compute_hash, validate_tool_call


class ToolCallsValidator:
    """Orchestrate tool call evaluation: file processing, stats, incremental mode."""

    def __init__(
        self,
        client: ApiClient,
        output_file: str = "results.jsonl",
        summary_file: str = "summary.json",
        incremental: bool = False,
    ):
        self.client = client
        self.output_file = output_file
        self.summary_file = summary_file
        self.incremental = incremental

        self.results: List[Dict[str, Any]] = []
        self.finish_reason_stat: Dict[str, int] = {}
        self.eval_start_ts: Optional[float] = None
        self.eval_end_ts: Optional[float] = None
        self.eval_started_at: Optional[str] = None
        self.eval_finished_at: Optional[str] = None

        self.file_lock = asyncio.Lock()
        self.stats_lock = asyncio.Lock()

        logger.info(f"Model: {self.client.model}")
        logger.info(f"Results will be saved to: {self.output_file}")
        logger.info(f"Summary will be saved to: {self.summary_file}")
        logger.info(f"Concurrency: {self.client.concurrency}")
        endpoint = (
            "/v1/completions"
            if self.client.adapter.supports_raw_completions()
            else "/v1/chat/completions"
        )
        logger.info(f"Request endpoint: {endpoint}")
        if self.incremental:
            logger.info("Incremental mode: enabled")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()
        return False

    def read_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Read test set file in JSONL format."""
        if not megfile.smart_exists(file_path):
            raise FileNotFoundError(f"Test file not found: {file_path}")

        requests = []
        with megfile.smart_open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_req = json.loads(line)
                    prepared_req = self.client.prepare_request(raw_req)
                    requests.append(
                        {
                            "data_index": line_num,
                            "raw": raw_req,
                            "prepared": prepared_req,
                            "hash": compute_hash(prepared_req),
                        }
                    )
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error at line {line_num}: {e}")
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")

        logger.info(f"Successfully read {len(requests)} requests")
        return requests

    def read_result_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Read result file in JSONL format."""
        results = []
        with megfile.smart_open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Parse error at line {line_num} in result file: {e}"
                    )
        return results

    async def process_request(
        self, prepared_req: Dict[str, Any], data_index: int
    ) -> Dict[str, Any]:
        """Process a single request, record duration and status."""
        start_time = time.time()
        status, response = await self.client.send_request(prepared_req["prepared"])
        duration_ms = int((time.time() - start_time) * 1000)

        finish_reason = None
        tool_calls_valid = None

        if response and "choices" in response and response["choices"]:
            choice = response["choices"][0]
            finish_reason = choice.get("finish_reason")

            if finish_reason == "tool_calls":
                tools = prepared_req["raw"].get("tools", [])
                tool_calls = choice.get("message", {}).get("tool_calls", [])
                if tool_calls:
                    tool_calls_valid = all(
                        validate_tool_call(tc, tools) for tc in tool_calls
                    )

        result = {
            "data_index": data_index,
            "request": prepared_req["prepared"],
            "extra_body": self.client.extra_body,
            "response": response,
            "status": status,
            "finish_reason": finish_reason,
            "tool_calls_valid": tool_calls_valid,
            "last_run_at": datetime.now().isoformat(),
            "duration_ms": duration_ms,
            "hash": prepared_req["hash"],
        }
        return result

    async def validate_file(self, file_path: str) -> None:
        """Validate all requests from test file."""
        self.eval_start_ts = time.time()
        self.eval_end_ts = None
        self.eval_started_at = datetime.now().isoformat()
        self.eval_finished_at = None

        all_requests = self.read_jsonl(file_path)
        if not all_requests:
            logger.warning("Test set is empty, no requests to process")
            return

        existing_hash_map = {}

        if self.incremental and megfile.smart_exists(self.output_file):
            existing_results = self.read_result_jsonl(self.output_file)
            for r in existing_results:
                existing_hash_map[r["hash"]] = r
            logger.info(
                f"Incremental mode: loaded {len(existing_results)} existing results"
            )
        else:
            async with self.file_lock:
                with megfile.smart_open(
                    self.output_file, "w", encoding="utf-8"
                ) as f:
                    pass
            logger.info(f"Initialized output file: {self.output_file}")

        await self.update_summary_file()

        tasks = []
        self.results = []

        for req in all_requests:
            h = req["hash"]
            data_index = req["data_index"]

            if self.incremental and h in existing_hash_map:
                r = existing_hash_map[h]
                if r.get("status") == "success":
                    self.results.append(r)
                    continue

            tasks.append(self.process_request(req, data_index))

        if not tasks:
            logger.info(
                "All requests already processed successfully, no need to rerun"
            )
            return

        logger.info(f"Preparing to process {len(tasks)} requests")

        with tqdm_asyncio(
            total=len(tasks), desc="Processing", unit="req"
        ) as pbar:
            for task in asyncio.as_completed(tasks):
                try:
                    res = await task
                    finish_reason = res.get("finish_reason")
                    self.finish_reason_stat[finish_reason] = (
                        self.finish_reason_stat.get(finish_reason, 0) + 1
                    )
                    self.results.append(res)
                    await self.save_result_and_update_stats(res)
                except Exception as e:
                    logger.error(f"Task execution failed: {e}")
                finally:
                    pbar.update(1)

        await self.deduplicate_and_sort_results()

        self.eval_end_ts = time.time()
        self.eval_finished_at = datetime.now().isoformat()

        await self.update_summary_file()

        logger.info(f"Results saved to: {self.output_file}")
        logger.info(f"Summary saved to: {self.summary_file}")

    async def save_result_and_update_stats(
        self, result: Dict[str, Any]
    ) -> None:
        async with self.file_lock:
            with megfile.smart_open(
                self.output_file, "a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        async with self.stats_lock:
            summary = self.compute_summary()
            logger.info(
                f"[Stats] Total: {summary['success_count'] + summary['failure_count']}, "
                f"Success: {summary['success_count']}, "
                f"Failed: {summary['failure_count']}, "
                f"Stop: {summary['finish_stop']}, "
                f"ToolCalls: {summary['finish_tool_calls']}, "
                f"ToolCallValid: {summary['successful_tool_call_count']}, "
                f"ToolCallInvalid: {summary['schema_validation_error_count']}"
            )

    async def deduplicate_and_sort_results(self) -> None:
        """Deduplicate and sort results by data_index."""
        if not megfile.smart_exists(self.output_file):
            logger.warning(f"Output file does not exist: {self.output_file}")
            return

        all_results = self.read_result_jsonl(self.output_file)
        if not all_results:
            logger.info("No results to process")
            return

        logger.info(
            f"Processing {len(all_results)} results for deduplication and sorting"
        )

        results_by_index: Dict[int, Dict[str, Any]] = {}
        for result in all_results:
            data_index = result.get("data_index")
            if data_index is None:
                continue
            last_run_at = result.get("last_run_at")
            if last_run_at is None:
                continue
            if data_index not in results_by_index:
                results_by_index[data_index] = result
            else:
                existing_last_run = results_by_index[data_index].get(
                    "last_run_at"
                )
                if (
                    existing_last_run is None
                    or last_run_at > existing_last_run
                ):
                    results_by_index[data_index] = result

        deduplicated_results = list(results_by_index.values())
        deduplicated_results.sort(key=lambda x: x.get("data_index", 0))

        logger.info(
            f"Deduplicated from {len(all_results)} to "
            f"{len(deduplicated_results)} results"
        )

        async with self.file_lock:
            with megfile.smart_open(
                self.output_file, "w", encoding="utf-8"
            ) as f:
                for result in deduplicated_results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        self.results = deduplicated_results
        logger.info(
            f"Results deduplicated, sorted, and saved to: {self.output_file}"
        )

    async def update_summary_file(self) -> None:
        summary = self.compute_summary()
        with megfile.smart_open(
            self.summary_file, "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)

    def compute_summary(self) -> Dict[str, Any]:
        """Compute summary stats from self.results."""
        summary = {
            "model": self.client.model,
            "success_count": 0,
            "failure_count": 0,
            "finish_stop": 0,
            "finish_tool_calls": 0,
            "finish_others": 0,
            "finish_others_detail": {},
            "schema_validation_error_count": 0,
            "successful_tool_call_count": 0,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "eval_started_at": self.eval_started_at,
            "eval_finished_at": self.eval_finished_at,
            "eval_duration_ms": None,
        }

        for r in self.results:
            status = r.get("status")
            finish_reason = r.get("finish_reason")
            tool_calls_valid = r.get("tool_calls_valid")

            usage = (r.get("response") or {}).get("usage")
            if isinstance(usage, dict):
                pt = usage.get("prompt_tokens")
                ct = usage.get("completion_tokens")
                tt = usage.get("total_tokens")
                if isinstance(pt, int):
                    summary["usage"]["prompt_tokens"] += pt
                if isinstance(ct, int):
                    summary["usage"]["completion_tokens"] += ct
                if isinstance(tt, int):
                    summary["usage"]["total_tokens"] += tt

            if status == "success":
                summary["success_count"] += 1
            else:
                summary["failure_count"] += 1

            if finish_reason == "stop":
                summary["finish_stop"] += 1
            elif finish_reason == "tool_calls":
                summary["finish_tool_calls"] += 1
                if tool_calls_valid:
                    summary["successful_tool_call_count"] += 1
                else:
                    summary["schema_validation_error_count"] += 1
            elif finish_reason:
                summary["finish_others"] += 1
                summary["finish_others_detail"].setdefault(finish_reason, 0)
                summary["finish_others_detail"][finish_reason] += 1

        if isinstance(self.eval_start_ts, (int, float)):
            end_ts = (
                self.eval_end_ts
                if isinstance(self.eval_end_ts, (int, float))
                else time.time()
            )
            summary["eval_duration_ms"] = int(
                max(0.0, (end_ts - self.eval_start_ts) * 1000)
            )

        return summary
