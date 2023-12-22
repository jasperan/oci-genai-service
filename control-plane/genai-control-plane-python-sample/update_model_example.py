import oci

CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

# Service endpoint
endpoint = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

generative_ai_client = oci.generative_ai.GenerativeAiClient(config=config, service_endpoint=endpoint)

update_model_details = oci.generative_ai.models.UpdateModelDetails()
update_model_details.display_name = "updated_sample_code_model"
update_model_details.vendor = "updated_genai_testing"
update_model_details.version = "updated_demo_testing"
update_model_details.description = "This can be deleted"

response = generative_ai_client.update_model("<model_ocid>", update_model_details)

print("**************************UpdateModel Result**************************")
print(response.data)
