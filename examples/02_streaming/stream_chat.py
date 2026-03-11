"""Streaming Chat -- Stream tokens as they are generated.

Usage:
    export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..xxx
    python stream_chat.py

Requirements:
    pip install oci-genai-service
"""

from oci_genai_service import GenAIClient

client = GenAIClient()

print("Streaming response: ", end="", flush=True)
for chunk in client.chat(
    "Write a haiku about Oracle Cloud.",
    model="meta.llama-4-maverick",
    stream=True,
):
    print(chunk, end="", flush=True)
print()
