"""
Compare vendor results against a baseline to compute ToolCall-Trigger Similarity
(precision, recall, F1) and ToolCall-Schema Accuracy.

Usage:
    python -m vendor_verifier.compare \
        --baseline tool-calls/kimi-k2-0905-preview_results.jsonl \
        --vendor tool-calls/gpt-4o-results.jsonl \
        --output tool-calls/gpt-4o-comparison.json
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from loguru import logger


def load_results(path: str) -> Dict[int, Dict[str, Any]]:
    """Load a results JSONL file, keyed by data_index."""
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("data_index")
            if idx is not None:
                results[idx] = obj
    return results


def is_tool_calls(finish_reason: Optional[str]) -> bool:
    return finish_reason == "tool_calls"


def compute_comparison(
    baseline: Dict[int, Dict[str, Any]],
    vendor: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compare vendor results against baseline.

    Metrics:
    - ToolCall-Trigger Similarity: TP/FP/FN/TN -> precision, recall, F1
    - ToolCall-Schema Accuracy: count_finish_reason_tool_calls, count_successful_tool_call, schema_accuracy
    """
    tp = fp = fn = tn = 0
    count_finish_reason_tool_calls = 0
    count_successful_tool_call = 0
    matched = 0
    skipped = 0

    # Only compare data_indexes present in both
    common_indices = sorted(set(baseline.keys()) & set(vendor.keys()))

    for idx in common_indices:
        b = baseline[idx]
        v = vendor[idx]

        # Skip if either failed
        if b.get("status") != "success" or v.get("status") != "success":
            skipped += 1
            continue

        matched += 1
        b_tc = is_tool_calls(b.get("finish_reason"))
        v_tc = is_tool_calls(v.get("finish_reason"))

        if v_tc and b_tc:
            tp += 1
        elif v_tc and not b_tc:
            fp += 1
        elif not v_tc and b_tc:
            fn += 1
        else:
            tn += 1

        # Schema accuracy (vendor side only)
        if v_tc:
            count_finish_reason_tool_calls += 1
            if v.get("tool_calls_valid"):
                count_successful_tool_call += 1

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    schema_accuracy = (
        count_successful_tool_call / count_finish_reason_tool_calls
        if count_finish_reason_tool_calls > 0
        else 0.0
    )

    return {
        "total_baseline": len(baseline),
        "total_vendor": len(vendor),
        "common_indices": len(common_indices),
        "matched_success": matched,
        "skipped_failures": skipped,
        "tool_call_trigger_similarity": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "tool_call_schema_accuracy": {
            "count_finish_reason_tool_calls": count_finish_reason_tool_calls,
            "count_successful_tool_call": count_successful_tool_call,
            "schema_accuracy": f"{schema_accuracy:.2%}",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare vendor results against a baseline."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline results JSONL (e.g. official API results)",
    )
    parser.add_argument(
        "--vendor",
        required=True,
        help="Path to vendor results JSONL to evaluate",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save comparison JSON (default: print to stdout)",
    )
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    vendor = load_results(args.vendor)

    logger.info(f"Baseline: {len(baseline)} results from {args.baseline}")
    logger.info(f"Vendor:   {len(vendor)} results from {args.vendor}")

    report = compute_comparison(baseline, vendor)

    output_str = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_str + "\n")
        logger.info(f"Comparison saved to {args.output}")
    else:
        print(output_str)

    # Print summary to stderr for quick review
    ts = report["tool_call_trigger_similarity"]
    sa = report["tool_call_schema_accuracy"]
    logger.info(
        f"Trigger Similarity: F1={ts['f1']:.2%}  "
        f"(P={ts['precision']:.2%}, R={ts['recall']:.2%})  "
        f"TP={ts['TP']} FP={ts['FP']} FN={ts['FN']} TN={ts['TN']}"
    )
    logger.info(
        f"Schema Accuracy: {sa['schema_accuracy']}  "
        f"({sa['count_successful_tool_call']}/{sa['count_finish_reason_tool_calls']})"
    )


if __name__ == "__main__":
    main()
