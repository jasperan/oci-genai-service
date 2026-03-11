"""Thinking Traces -- Extract model reasoning from Grok Code responses.

Usage:
    export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..xxx
    python grok_reasoning.py

Requirements:
    pip install oci-genai-service
"""

from oci_genai_service import GenAIClient
from oci_genai_service.inference.thinking import extract_thinking

client = GenAIClient()
response = client.chat(
    "What is 15% of 847?",
    model="xai.grok-code-fast-1",
)

result = extract_thinking(response.text)
if result.thinking:
    print("=== Model's Reasoning ===")
    print(result.thinking)
    print()
print("=== Answer ===")
print(result.answer)
