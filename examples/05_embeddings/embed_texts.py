"""Text Embeddings -- Generate vector embeddings for text.

Usage:
    export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..xxx
    python embed_texts.py

Requirements:
    pip install oci-genai-service
"""

from oci_genai_service import GenAIClient

client = GenAIClient()
result = client.embed(
    ["Oracle Cloud is fast", "Machine learning is powerful", "The sky is blue"],
    model="cohere.embed-english-v3.0",
)

print(f"Model: {result.model}")
print(f"Input count: {result.input_count}")
for i, vec in enumerate(result.vectors):
    print(f"  Text {i}: {len(vec)} dimensions, first 5 values: {vec[:5]}")
