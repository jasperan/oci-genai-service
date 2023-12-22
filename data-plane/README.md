# Generative AI Service DP python sample for demo

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
cd ~/generative-ai-demo
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
'''
Python 3.8.3
'''
 
## 4. Install all dependencies(include our beta version SDK):
```
# We suggest you install dependencies in a virtual env to avoid conflicts on your system
cd generative-ai-service-python-sample
python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements-local.txt
```
## 5. Make sure you have updated all corresponding fields in target script 
for example  in “embed_text_demo.py” update compartmentId=“target compartment id "

## 6. Kick off different python sample scripts
```
python generate_text_demo.py
python summarize_text_demo.py
python embed_text_demo.py
```
 
## 6. Example Results
generate_text_demo.py
```
**************************Generate Texts Result**************************
{
  "generated_texts": [
    [
      {
        "finish_reason": null,
        "id": "2d042434-6392-4ad8-83f0-d6e000c08c82",
        "likelihood": null,
        "text": " Here are a few facts about Earth:\n\n- The Earth is the third planet from the sun",
        "token_likelihoods": null
      }
    ]
  ],
  "id": "2d042434-6392-4ad8-83f0-d6e000c08c82",
  "model_id": "cohere.command",
  "model_version": "v14.2",
  "prompts": null,
  "time_created": "2023-08-31T20:01:39.646000+00:00"
}
```
summarize_text_demo.py
```
**************************Summarize Texts Result**************************
{
  "id": "a8f01d58-bda7-456b-b119-be96442df113",
  "input": null,
  "model_id": "cohere.command",
  "model_version": "v14.2",
  "summary": "Quantum dots (QDs) are semiconductor particles of a few nanometers in size, having unique optical and electronic properties resulting from quantum mechanics, compared to larger particles. When the dots are illuminated, an electron can be excited to a higher energy state. The excited electron can drop back into the valence band, releasing its energy as light, as part of a process called photoluminescence. The colour of the light depends upon the difference between the conductance band and the valence band, or between discrete energy states in the case of poorly defined band structures."
}
```
embed_text-demo.py
```
**************************Embed Texts Result**************************
{
  "embeddings": [
    [
      -0.20739746,
      -0.34399414,
      -1.6728516,
      -0.71484375,
      -0.5629883,
      ...
      -0.6972656,
      1.3525391
    ]
  ],
  "id": "dbf613ab-1d3d-416b-8fa4-bd2b01b63f31",
  "inputs": null,
  "model_id": "ocid1.generativeaiworkrequest.oc1.us-chicago-1.amaaaaaapi24rzaajd7mtlse6whk26l6yvt3ajkzkjr5bctzfcjd4zjrfbrq",
  "model_version": "2.0"
}
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

