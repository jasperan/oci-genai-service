import time
import oci

# User's compartment OCID
compartment_id = "<compartment_ocid>"
CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

# Service endpoint
endpoint = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

generative_ai_client = oci.generative_ai.generative_ai_client.GenerativeAiClient(config=config, service_endpoint=endpoint)

create_endpoint_details = oci.generative_ai.models.CreateEndpointDetails()
create_endpoint_details.compartment_id = compartment_id
create_endpoint_details.dedicated_ai_cluster_id = "<dac_ocid>"
create_endpoint_details.model_id = "<model_ocid>"
create_endpoint_details.display_name = "delete me please"

if "<compartment_ocid>" in compartment_id:
    print("ERROR:Please update your compartment id in target python file")
    quit()

if "<dac_ocid>" in create_endpoint_details.dedicated_ai_cluster_id:
    print("ERROR:Please update your dedicated_ai_cluster id in target python file")
    quit()

if "<model_ocid>" in create_endpoint_details.model_id:
    print("ERROR:Please update your model id in target python file")
    quit()

response = generative_ai_client.create_endpoint(create_endpoint_details)

print("**************************CreateEndpoint Result**************************")
print(response.data)

# Make an inference call to validate the newly created endpoint
endpoint_id = response.data.id
endpoint_is_ready = False
while get_endpoint_res.lifecycle_state != "ACTIVE":
    time.sleep(3)
    print("Waiting endpoint " + endpoint_id + " to be active ...")

    get_endpoint_res = generative_ai_client.get_endpoint(endpoint_id)

    if get_endpoint_res.data.lifecycle_state == "ACTIVE":
        endpoint_is_ready = True
        print("Endpoint " + endpoint_id + " is active.")

prompts = ["<enter-the-prompt-you-want-to-test>"]
generate_text_detail = oci.generative_ai.models.GenerateTextDetails()
generate_text_detail.prompts = prompts
generate_text_detail.serving_mode = oci.generative_ai.models.DedicatedServingMode(endpoint_id=endpoint_id)
generate_text_detail.compartment_id = compartment_id
generate_text_detail.max_tokens = 20
generate_text_detail.temperature = 0.75
generate_text_detail.frequency_penalty = 1.0
generate_text_detail.top_p = 0.7

generate_text_response = generative_ai_client.generate_text(generate_text_detail)

# Print result
print("**************************Generate Texts Result**************************")
print(generate_text_response.data)