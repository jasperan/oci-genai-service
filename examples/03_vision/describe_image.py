"""Image Description -- Describe an image using a vision model.

Usage:
    export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..xxx
    python describe_image.py <image_path_or_url>

Requirements:
    pip install oci-genai-service
"""

import sys

from oci_genai_service import GenAIClient

image = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/"
    "Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
)

client = GenAIClient()
response = client.chat(
    "Describe this image in detail.",
    model="meta.llama-3.2-90b-vision-instruct",
    images=[image],
)

print(response.text)
