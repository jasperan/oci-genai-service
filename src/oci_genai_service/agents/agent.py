"""Tool-calling agent with configurable memory and reasoning loop."""

from __future__ import annotations

import json
from typing import Optional

from oci_genai_service.client import GenAIClient
from oci_genai_service.agents.memory import BaseMemory, InMemoryMemory


class AgentResponse:
    """Response from an agent run."""

    def __init__(self, text: str, tool_calls: list[dict] = None, steps: int = 0):
        self.text = text
        self.tool_calls = tool_calls or []
        self.steps = steps


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
                    fn_args = json.loads(tc.function.arguments)

                    if fn_name in self._tool_map:
                        result = self._tool_map[fn_name](**fn_args)
                    else:
                        result = f"Error: unknown tool '{fn_name}'"

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
        return AgentResponse(text=response.text, tool_calls=all_tool_calls, steps=self.max_iterations)

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
