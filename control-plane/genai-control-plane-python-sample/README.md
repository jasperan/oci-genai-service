# Generative AI Service CP python sample for demo

**How to run \*_demo.py on your laptop :**

## 0. Set up
```
Follow links below to generate a config file and a key pair in your ~/.oci directory
https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm
https://docs.oracle.com/en-us/iaas/Content/API/Concepts/apisigningkey.htm
https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/cliinstall.htm#configfile

After completion, you should have following 2 things in your ~/.oci directory 

a. A config file(where key file point to private key:key_file=~/.oci/oci_api_key.pem)
b. A key pair named oci_api_key.pem and oci_api_key_public.pem

Now make sure you change the reference of key file in config file (where key file point to private key:key_file=/YOUR_DIR_TO_KEY_FILE/oci_api_key.pem)
```
## 1. Make sure you are in the root directory of this project:
```
cd ~/genai-demo
```

## 2. Upload Public Key
```
# Upload your oci_api_key_public.pem to console:
https://docs.oracle.com/en-us/iaas/Content/API/Concepts/apisigningkey.htm#three
```

## 3. Make sure you have python installed on your machine
```
python --version
```
And I see following:
```
Python 3.8.16
```
 
## 4. Install all dependencies(include our beta version SDK):
```
# We suggest you install dependencies in a virtual env to avoid conflicts on your system
cd genai-control-plane-python-sample
python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements-local.txt
```

## 5. make sure you have updated all corresponding fields in target script 
for example  in “list_models_example.py” update compartmentId=“target compartment id "

## 6. Kick off different python sample scripts directly
```
python <script_name>.py
```

## Appendix: Token-based Authentication
Check [Token-based Authentication for the CLI](https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/clitoken.htm#Running_Scripts_on_a_Computer_without_a_Browser)
if need to run testing with session token via BOAT. Here is the sample config to setup client with session token in the test scripts:
```
config = oci.config.from_file('~/.oci/config', profile_name="oc1")

def make_security_token_signer(oci_config):
    pk = oci.signer.load_private_key_from_file(oci_config.get("key_file"), None)
    with open(oci_config.get("security_token_file")) as f:
        st_string = f.read()
    return oci.auth.signers.SecurityTokenSigner(st_string, pk)

signer = make_security_token_signer(oci_config=config)
# Service endpoint
endpoint = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

generative_ai_client = oci.generative_ai.generative_ai_client.GenerativeAiClient(config=config, service_endpoint=endpoint, retry_strategy=oci.retry.NoneRetryStrategy(), signer=signer)
```
