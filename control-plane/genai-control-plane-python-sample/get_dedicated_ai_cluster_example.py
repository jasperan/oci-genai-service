import oci

CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

# Service endpoint
endpoint = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

generative_ai_client = oci.generative_ai.GenerativeAiClient(config=config, service_endpoint=endpoint)

response = generative_ai_client.get_dedicated_ai_cluster("<dac_ocid>")

print("**************************GetDedicatedAiCluster Result**************************")
print(response.data)