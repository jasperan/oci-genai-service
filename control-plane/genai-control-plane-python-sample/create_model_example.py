import oci

# User's compartment OCID
compartment_id = "<compartment_ocid>"
CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

# Service endpoint
endpoint = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

generative_ai_client = oci.generative_ai.GenerativeAiClient(config=config, service_endpoint=endpoint)

training_config = oci.generative_ai.models.TFewTrainingConfig()
training_config.training_config_type = "TFEW_TRAINING_CONFIG"
training_config.total_training_epochs = 50
training_config.learning_rate = 0.001
training_config.training_batch_size = 10
training_config.early_stopping_patience = 10
training_config.early_stopping_threshold = 0.01
training_config.log_model_metrics_interval_in_steps = 0

training_dataset = oci.generative_ai.models.ObjectStorageDataset()
training_dataset.dataset_type = "OBJECT_STORAGE"
training_dataset.namespace_name = "axk4z7krhqfx"
training_dataset.bucket_name = "genai-test"
training_dataset.object_name = "sales_pitch_generation_train.jsonl"

fine_tune_details = oci.generative_ai.models.FineTuneDetails()
fine_tune_details.dedicated_ai_cluster_id = "<dac_ocid>"
fine_tune_details.training_config = training_config
fine_tune_details.training_dataset = training_dataset

create_model_details = oci.generative_ai.models.CreateModelDetails()
create_model_details.compartment_id = compartment_id
create_model_details.fine_tune_details = fine_tune_details
create_model_details.display_name = "please delete me"
create_model_details.vendor = "genai-testing"
create_model_details.version = "demo.testing"
create_model_details.description = "can be deleted"
create_model_details.base_model_id = "<basemodel_ocid>"

if "<compartment_ocid>" in compartment_id:
    print("ERROR:Please update your compartment id in target python file")
    quit()

if "<dac_ocid>" in fine_tune_details.dedicated_ai_cluster_id:
    print("ERROR:Please update your dedicated_ai_cluster id in target python file")
    quit()

if "<basemodel_ocid>" in create_model_details.base_model_id:
    print("ERROR:Please update your base_model id in target python file")
    quit()

response = generative_ai_client.create_model(create_model_details)

print("**************************CreateModel Result**************************")
print(response.data)
