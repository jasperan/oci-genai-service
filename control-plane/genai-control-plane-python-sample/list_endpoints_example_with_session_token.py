import oci

# User's compartment OCID
compartment_id = "<compartment-id>"
CONFIG_PROFILE = "<session-token-profile-name>"

config = oci.config.from_file('~/.oci/config', profile_name=CONFIG_PROFILE)
token_file = config['security_token_file']
token = None
with open(token_file, 'r') as f:
     token = f.read()

private_key = oci.signer.load_private_key_from_file(config['key_file'])
signer = oci.auth.signers.SecurityTokenSigner(token, private_key)

# Service endpoint
endpoint = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

generative_ai_client = oci.generative_ai.GenerativeAiClient(config=config, signer=signer, service_endpoint=endpoint)

response = generative_ai_client.list_endpoints(compartment_id)

print("**************************ListModels Result**************************")
print(response.data)