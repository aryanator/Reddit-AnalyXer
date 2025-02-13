import boto3
import json

# Initialize SageMaker runtime client
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name="your-region")

# SageMaker endpoint name
ENDPOINT_NAME = "your-sagemaker-endpoint-name"

def summarize_post(post):
    """
    Summarize a post's title and body using the SageMaker endpoint.
    """
    text = f"{post['title']}. {post['body']}"
    
    # Prepare the input data for the endpoint
    input_data = {
        "inputs": text,
        "parameters": {
            "max_length": 50,
            "min_length": 25,
            "do_sample": False
        }
    }

    # Send the request to the SageMaker endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(input_data)
    )

    # Parse the response
    result = json.loads(response["Body"].read().decode())
    return result[0]["summary_text"]