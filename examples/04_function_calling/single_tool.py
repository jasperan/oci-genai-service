"""Function Calling -- Let the model call a Python function.

Usage:
    export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..xxx
    python single_tool.py

Requirements:
    pip install oci-genai-service
"""

from oci_genai_service import GenAIClient
from oci_genai_service.inference.tools import tool


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city."""
    # In production, this would call a weather API
    return f"Sunny, 22 degrees {unit} in {city}"


client = GenAIClient()

# The model will decide to call get_weather
response = client.chat(
    "What's the weather like in Tokyo?",
    model="meta.llama-3.3-70b-instruct",
    tools=[get_weather.schema],
)

print(response.text)
