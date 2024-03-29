# coding: utf-8
# Copyright (c) 2023, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.

##########################################################################
# embed_text_demo.py
# Supports Python 3
##########################################################################
# Info:
# Get texts from LLM model for given prompts using OCI Generative AI Service.
##########################################################################
# Application Command line(no parameter needed)
# python embed_text_demo.py
##########################################################################

import oci
import logging
import yaml

# Enable debug logging
logging.getLogger('oci').setLevel(logging.DEBUG)


with open('config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

compartment_id = config_data['compartment_id']
CONFIG_PROFILE = config_data['config_profile']
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

# Service endpoint
endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config, service_endpoint=endpoint, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))

inputs = ["Hello", "World"]
embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
embed_text_detail.inputs = inputs
embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id="cohere.embed-english-light-v2.0")
embed_text_detail.compartment_id = compartment_id

if "<compartment_ocid>" in compartment_id:
    print("ERROR:Please update your compartment id in target python file")
    quit()
    
embed_text_response = generative_ai_inference_client.embed_text(embed_text_detail)

# Print result
print("**************************Embed Texts Result**************************")
print(embed_text_response.data)

"""
https://docs.cohere.com/reference/embed
Available models and corresponding embedding dimensions:

    embed-english-v3.0 1024

    embed-multilingual-v3.0 1024

    embed-english-light-v3.0 384

    embed-multilingual-light-v3.0 384

    embed-english-v2.0 4096

    embed-english-light-v2.0 1024

    embed-multilingual-v2.0 768

"""
