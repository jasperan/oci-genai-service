"""Tool-calling agent with configurable memory and reasoning loop."""

from __future__ import annotations

import inspect
import json
from typing import Optional

from oci_genai_service.client import GenAIClient
from oci_genai_service.agents.memory import BaseMemory, InMemoryMemory


class AgentResponse:
    """Response from an agent run."""

    def __init__(
        self,
        text: str,
        tool_calls: Optional[list[dict]] = None,
        steps: int = 0,
        stopped_reason: Optional[str] = None,
    ):
        self.text = text
        self.tool_calls = tool_calls or []
        self.steps = steps
        self.stopped_reason = stopped_reason


class Agent:
    """Tool-calling agent powered by OCI GenAI."""

    def __init__(
        self,
        client: GenAIClient,
        model: str = "meta.llama-4-maverick",
        tools: Optional[list] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.client = client
        self.model = model
        self.tools = tools or []
        self.memory = memory or InMemoryMemory()
        self.system_prompt = system_prompt or "You are a helpful assistant with access to tools."
        self.max_iterations = max_iterations

        self._tool_map = {}
        self._tool_schemas = []
        for t in self.tools:
            self._tool_map[t.__name__] = t
            self._tool_schemas.append(t.schema)

    def _call_tool(self, fn_name: str, fn_args: dict) -> str:
        """Call a registered tool only with arguments its signature can accept."""
        fn = self._tool_map.get(fn_name)
        if fn is None:
            return f"Error: unknown tool '{fn_name}'"
        try:
            parameters = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            parameters = None
        if parameters is not None and not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        ):
            allowed = {name for name in parameters if name not in ("self", "cls")}
            unknown = sorted(set(fn_args) - allowed)
            if unknown:
                return f"Error: unknown argument(s) for tool '{fn_name}': {', '.join(unknown)}"
            missing = sorted(
                name
                for name, parameter in parameters.items()
                if name not in ("self", "cls")
                and name not in fn_args
                and parameter.default is inspect.Parameter.empty
                and parameter.kind
                in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            )
            if missing:
                return f"Error: missing argument(s) for tool '{fn_name}': {', '.join(missing)}"
        return str(fn(**fn_args))

    def run(self, prompt: str, session_id: str = "default") -> AgentResponse:
        """Run the agent on a prompt."""
        self.memory.add(session_id, "user", prompt)

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.memory.get(session_id))

        all_tool_calls = []

        for step in range(self.max_iterations):
            response = self.client.chat(
                prompt=messages,
                model=self.model,
                tools=self._tool_schemas if self._tool_schemas else None,
            )

            raw = response.raw
            if raw and raw.choices[0].message.tool_calls:
                tool_calls = raw.choices[0].message.tool_calls
                messages.append(raw.choices[0].message.model_dump())

                for tc in tool_calls:
                    fn_name = tc.function.name
                    try:
                        decoded = json.loads(tc.function.arguments or "{}")
                    except (TypeError, ValueError):
                        decoded = None

                    if isinstance(decoded, dict):
                        fn_args = decoded
                        result = self._call_tool(fn_name, fn_args)
                    else:
                        fn_args = {}
                        result = f"Error: arguments for tool '{fn_name}' are not a JSON object"

                    all_tool_calls.append({
                        "name": fn_name,
                        "args": fn_args,
                        "result": str(result),
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })
            else:
                self.memory.add(session_id, "assistant", response.text)
                return AgentResponse(
                    text=response.text,
                    tool_calls=all_tool_calls,
                    steps=step + 1,
                )

        self.memory.add(session_id, "assistant", response.text)
        return AgentResponse(
            text=response.text,
            tool_calls=all_tool_calls,
            steps=self.max_iterations,
            stopped_reason="max_iterations",
        )

    def session(self, session_id: str) -> "AgentSession":
        """Create a named session for multi-turn conversations."""
        return AgentSession(agent=self, session_id=session_id)


class AgentSession:
    """A named conversation session with an agent."""

    def __init__(self, agent: Agent, session_id: str):
        self.agent = agent
        self.session_id = session_id

    def run(self, prompt: str) -> AgentResponse:
        return self.agent.run(prompt, session_id=self.session_id)

    def clear(self) -> None:
        self.agent.memory.clear(self.session_id)
