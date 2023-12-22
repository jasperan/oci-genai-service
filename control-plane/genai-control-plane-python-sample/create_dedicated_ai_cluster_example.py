import oci

# User's compartment OCID
compartment_id = "<compartment_ocid>"
CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

# Service endpoint
endpoint = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

generative_ai_client = oci.generative_ai.GenerativeAiClient(config=config, service_endpoint=endpoint)

create_dedicated_ai_cluster_details = oci.generative_ai.models.CreateDedicatedAiClusterDetails()
create_dedicated_ai_cluster_details.description = "can be deleted"
create_dedicated_ai_cluster_details.display_name = "please delete me"
create_dedicated_ai_cluster_details.type = "FINE_TUNING"
create_dedicated_ai_cluster_details.compartment_id = compartment_id
create_dedicated_ai_cluster_details.unit_count = 1
create_dedicated_ai_cluster_details.unit_shape = "X_LARGE"

if "<compartment_ocid>" in compartment_id:
    print("ERROR:Please update your compartment id in target python file")
    quit()

response = generative_ai_client.create_dedicated_ai_cluster(create_dedicated_ai_cluster_details)

print("**************************CreateDedicatedAiCluster Result**************************")
print(response.data)
