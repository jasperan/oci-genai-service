import oci

# User's compartment OCID
compartment_id = "<compartment_ocid>"
CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

# Service endpoint
endpoint = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

generative_ai_client = oci.generative_ai.GenerativeAiClient(config=config, service_endpoint=endpoint)

if "<compartment_ocid>" in compartment_id:
    print("ERROR:Please update your compartment id in target python file")
    quit()

response = generative_ai_client.list_work_request_logs("<work_request_id>")

print("**************************ListWorkRequestLogs Result**************************")
print(response.data)