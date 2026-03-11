"""List Models -- Browse available OCI GenAI models.

Usage:
    python list_models.py

Requirements:
    pip install oci-genai-service
"""

from oci_genai_service import list_models, get_model

# List all models
print("=== All Models ===")
for m in list_models():
    print(f"  {m.id:<45} {m.name}")

# Filter by capability
print("\n=== Vision Models ===")
for m in list_models(capability="vision"):
    print(f"  {m.id:<45} Regions: {', '.join(m.regions)}")

# Get detailed info
print("\n=== Model Details ===")
info = get_model("meta.llama-4-maverick")
print(f"  Name: {info.name}")
print(f"  Context: {info.context_window} tokens")
print(f"  Capabilities: {', '.join(info.capabilities)}")
print(f"  API: {info.api}")
