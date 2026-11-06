"""Tests for the agent framework."""

from unittest.mock import MagicMock
from oci_genai_service.agents.agent import Agent
from oci_genai_service.agents.memory import InMemoryMemory
from oci_genai_service.inference.tools import tool


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 22C in {city}"


class TestAgent:
    def test_creates_with_tools(self):
        client = MagicMock()
        agent = Agent(client=client, model="meta.llama-4-maverick", tools=[add_numbers, get_weather])
        assert len(agent.tools) == 2
        assert "add_numbers" in agent._tool_map
        assert "get_weather" in agent._tool_map

    def test_creates_with_system_prompt(self):
        client = MagicMock()
        agent = Agent(client=client, model="test", tools=[], system_prompt="Be helpful")
        assert agent.system_prompt == "Be helpful"

    def test_call_tool_rejects_bad_arguments(self):
        agent = Agent(client=MagicMock(), model="test", tools=[add_numbers])
        assert agent._call_tool("add_numbers", {"a": 1, "b": 2}) == "3"
        assert "unknown argument" in agent._call_tool("add_numbers", {"a": 1, "b": 2, "c": 3})
        assert "missing argument" in agent._call_tool("add_numbers", {"a": 1})
        assert "unknown tool" in agent._call_tool("nope", {})


class TestInMemoryMemory:
    def test_add_and_get_messages(self):
        memory = InMemoryMemory()
        memory.add("session1", "user", "Hello")
        memory.add("session1", "assistant", "Hi there")
        messages = memory.get("session1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["content"] == "Hi there"

    def test_separate_sessions(self):
        memory = InMemoryMemory()
        memory.add("s1", "user", "Hello")
        memory.add("s2", "user", "Bonjour")
        assert len(memory.get("s1")) == 1
        assert len(memory.get("s2")) == 1

    def test_clear_session(self):
        memory = InMemoryMemory()
        memory.add("s1", "user", "Hello")
        memory.clear("s1")
        assert len(memory.get("s1")) == 0
