"""Tests for function/tool calling support."""

import pytest
from oci_genai_service.inference.tools import tool, build_tool_schema


class TestToolDecorator:
    def test_basic_tool(self):
        @tool
        def get_weather(city: str, unit: str = "celsius") -> str:
            """Get weather for a city."""
            return f"Sunny in {city}"

        schema = get_weather.schema
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert "city" in schema["function"]["parameters"]["properties"]
        assert "city" in schema["function"]["parameters"]["required"]
        assert "unit" not in schema["function"]["parameters"]["required"]

    def test_tool_is_still_callable(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert add(2, 3) == 5


class TestBuildToolSchema:
    def test_from_function(self):
        def search(query: str, max_results: int = 10) -> list:
            """Search the database."""
            pass

        schema = build_tool_schema(search)
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search the database."
