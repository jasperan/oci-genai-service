import oci

CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

# Service endpoint
endpoint = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

generative_ai_client = oci.generative_ai.GenerativeAiClient(config=config, service_endpoint=endpoint)

update_dedicated_ai_cluster_details = oci.generative_ai.models.UpdateDedicatedAiClusterDetails()
update_dedicated_ai_cluster_details.description = "Can be deleted"
update_dedicated_ai_cluster_details.display_name = "Please delete me"

response = generative_ai_client.update_dedicated_ai_cluster("ocid1.service.oc1.ap-osaka-1.amaaaaaapheaj2iaelolhg7475lfwbn6aixtnwfxkizkiy3w54l6bvku6vaa", update_dedicated_ai_cluster_details)

print("**************************UpdateDedicatedAiCluster Result**************************")
print(response.data)