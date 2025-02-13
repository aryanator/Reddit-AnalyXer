import boto3

s3 = boto3.client("s3")
s3.upload_file("model.tar.gz", "reddit-summarizer-models", "model.tar.gz")