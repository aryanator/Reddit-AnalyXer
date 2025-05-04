import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# Define SageMaker role
role = sagemaker.get_execution_role()

# Create Hugging Face Model
huggingface_model = HuggingFaceModel(
    model_data="s3://reddit-summarizer-models/fine-tuned-model.tar.gz",
    role=role,
    transformers_version="4.48.3",
    pytorch_version="2.6.0",
    py_version="3.12.4",
)

# Deploy the model
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
)