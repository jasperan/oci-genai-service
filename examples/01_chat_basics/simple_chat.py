"""Simple Chat -- Send a message and get a response.

Usage:
    export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..xxx
    python simple_chat.py

Requirements:
    pip install oci-genai-service
"""

from oci_genai_service import GenAIClient

client = GenAIClient()
response = client.chat(
    "Explain quantum computing in 3 sentences.",
    model="meta.llama-4-maverick",
    temperature=0.7,
)

print(f"Model: {response.model}")
print(f"Response: {response.text}")
