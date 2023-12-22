# coding: utf-8
# Copyright (c) 2023, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.

##########################################################################
# summarize_text_demo.py
# Supports Python 3
##########################################################################
# Info:
# Get texts from LLM model for given prompts using OCI Generative AI Service.
##########################################################################
# Application Command line(no parameter needed)
# python summarize_text_demo.py
##########################################################################

import oci

# Setup basic variables
# Auth Config
# TODO: Please update config profile name and use the compartmentId that has policies grant permissions for using Generative AI Service
compartment_id = "<compartment_ocid>"
CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

# Service endpoint
endpoint = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

generative_ai_client = oci.generative_ai.GenerativeAiClient(config=config, service_endpoint=endpoint, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))

text_to_summarize = "Quantum dots (QDs) - also called semiconductor nanocrystals, are semiconductor particles a few nanometres in size, having optical and electronic properties that differ from those of larger particles as a result of quantum mechanics. They are a central topic in nanotechnology and materials science. When the quantum dots are illuminated by UV light, an electron in the quantum dot can be excited to a state of higher energy. In the case of a semiconducting quantum dot, this process corresponds to the transition of an electron from the valence band to the conductance band. The excited electron can drop back into the valence band releasing its energy as light. This light emission (photoluminescence) is illustrated in the figure on the right. The color of that light depends on the energy difference between the conductance band and the valence band, or the transition between discrete energy states when the band structure is no longer well-defined in QDs."
summarize_text_detail = oci.generative_ai.models.SummarizeTextDetails()
summarize_text_detail.serving_mode = oci.generative_ai.models.OnDemandServingMode(model_id="cohere.command")
summarize_text_detail.compartment_id = compartment_id
summarize_text_detail.input = text_to_summarize

if "<compartment_ocid>" in compartment_id:
    print("ERROR:Please update your compartment id in target python file")
    quit()

summarize_text_response = generative_ai_client.summarize_text(summarize_text_detail)

# Print result
print("**************************Summarize Texts Result**************************")
print(summarize_text_response.data)