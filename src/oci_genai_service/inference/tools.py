"""Function/tool calling support with @tool decorator."""

from __future__ import annotations

import inspect
import json
from functools import wraps
from typing import Any, Callable, get_type_hints


_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def build_tool_schema(func: Callable) -> dict:
    """Build an OpenAI-format tool schema from a Python function."""
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        param_type = hints.get(name, str)
        json_type = _TYPE_MAP.get(param_type, "string")
        properties[name] = {"type": json_type}

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def tool(func: Callable) -> Callable:
    """Decorator to mark a function as an agent tool.

    Attaches .schema with the OpenAI-format tool definition.
    """
    func.schema = build_tool_schema(func)
    func.is_tool = True
    return func
