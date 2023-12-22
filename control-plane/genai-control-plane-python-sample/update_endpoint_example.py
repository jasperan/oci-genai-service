import oci

CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

# Service endpoint
endpoint = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

generative_ai_client = oci.generative_ai.GenerativeAiClient(config=config, service_endpoint=endpoint)

update_endpoint_details = oci.generative_ai.models.UpdateEndpointDetails()
update_endpoint_details.description = "Test description"
update_endpoint_details.display_name = "Please delete me"

response = generative_ai_client.update_endpoint("<endpoint_ocid>", update_endpoint_details)

print("**************************UpdateEndpoint Result**************************")
print(response.data)