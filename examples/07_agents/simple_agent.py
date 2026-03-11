"""Simple Agent -- An agent that uses tools to answer questions.

Usage:
    export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..xxx
    python simple_agent.py

Requirements:
    pip install oci-genai-service
"""

from oci_genai_service import GenAIClient
from oci_genai_service.agents import Agent, tool


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))  # noqa: S307


@tool
def lookup_capital(country: str) -> str:
    """Look up the capital city of a country."""
    capitals = {"france": "Paris", "japan": "Tokyo", "spain": "Madrid"}
    return capitals.get(country.lower(), "Unknown")


client = GenAIClient()
agent = Agent(
    client=client,
    model="meta.llama-4-maverick",
    tools=[calculate, lookup_capital],
    system_prompt="You are a helpful assistant. Use tools when needed.",
)

response = agent.run("What is the capital of France, and what is 42 * 17?")
print(f"Answer: {response.text}")
print(f"Tool calls: {len(response.tool_calls)}")
for tc in response.tool_calls:
    print(f"  - {tc['name']}({tc['args']}) -> {tc['result']}")
