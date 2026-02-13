import hashlib
import json
from typing import Any, Dict, List

from jsonschema import ValidationError, validate
from loguru import logger


def compute_hash(obj: Dict[str, Any]) -> str:
    """Compute a stable hash for incremental mode."""
    serialized = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def validate_tool_call(
    tool_call: Dict[str, Any], tools: List[Dict[str, Any]]
) -> bool:
    """
    Validate tool call arguments against JSON Schema.

    Args:
        tool_call: Tool call object with function.name and function.arguments.
        tools: Available tools list from the request.

    Returns:
        Whether validation passed.
    """
    try:
        tool_name = tool_call["function"]["name"]

        schema = next(
            (
                t["function"]["parameters"]
                for t in tools
                if t["function"]["name"] == tool_name
            ),
            None,
        )

        if not schema:
            logger.warning(f"No schema found for tool '{tool_name}'")
            return False

        args = tool_call["function"]["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSON parse failed for tool '{tool_name}' arguments: {e}"
                )
                return False

        validate(instance=args, schema=schema)
        return True

    except ValidationError as e:
        logger.warning(
            f"Schema validation failed for tool '{tool_name}': {e.message}"
        )
        return False
    except KeyError as e:
        logger.warning(f"Tool call format error, missing field: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error during validation: {e}")
        return False
