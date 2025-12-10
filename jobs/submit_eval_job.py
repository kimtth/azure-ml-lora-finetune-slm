"""
Submit evaluation job to Azure Machine Learning.

This script submits an evaluation job to test the fine-tuned model on Azure ML.
"""

import os
import warnings
import logging
from pathlib import Path
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
from dotenv import load_dotenv

# Suppress experimental class warnings and SDK debug logging
warnings.filterwarnings("ignore", message=".*experimental class.*")
logging.getLogger("azure.ai.ml").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)

load_dotenv()


def get_ml_client():
    """Create and return MLClient instance."""
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
    
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    
    print(f"Connected to workspace: {workspace_name}")
    return ml_client


def create_or_update_environment(ml_client: MLClient):
    """Create or update the evaluation environment."""
    env_name = "lora-finetune-env"
    
    # Path to conda.yaml
    conda_file_path = Path(__file__).parent.parent / "environment" / "conda.yaml"
    
    env = Environment(
        name=env_name,
        description="Environment for LoRA fine-tuning and evaluation",
        image="mcr.microsoft.com/azureml/curated/acft-hf-nlp-gpu:latest",
        conda_file=str(conda_file_path),
    )
    
    env = ml_client.environments.create_or_update(env)
    print(f"Environment: {env.name}:{env.version}")
    
    return f"{env.name}:{env.version}"


def get_data_inputs(ml_client: MLClient):
    """Get data inputs from blob storage via direct URIs."""
    
    # Get datastore info
    default_datastore = ml_client.datastores.get_default()
    
    # Construct paths to uploaded data
    blob_folder = "LocalUpload/lora-training-data"
    
    # Create Azure ML URIs that reference the blob storage paths
    val_uri = f"azureml://subscriptions/{ml_client.subscription_id}/resourcegroups/{ml_client.resource_group_name}/workspaces/{ml_client.workspace_name}/datastores/{default_datastore.name}/paths/{blob_folder}/validation.jsonl"
    
    val_data = Input(
        type=AssetTypes.URI_FILE,
        path=val_uri,
    )
    
    print("Evaluation data configured from blob storage")
    print(f"  Test data: {blob_folder}/validation.jsonl")
    return val_data


def submit_evaluation_job(
    ml_client: MLClient,
    environment: str,
    model_path: str,
    test_data: Input,
    compute_name: str = None,
    experiment_name: str = "lora-evaluation",
):
    """Submit the evaluation job to Azure ML."""
    
    if compute_name is None:
        compute_name = os.getenv("AZURE_COMPUTE_NAME", "gpu-cluster")
    
    # Path to evaluation script
    code_dir = Path(__file__).parent.parent / "src"
    
    # Load training config to get model name
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    base_model = config['model']['name']
    
    # Configure evaluation command
    job = command(
        code=str(code_dir),
        command=(
            "python evaluate.py "
            f"--model_path {model_path} "
            f"--base_model {base_model} "
            "--test_data ${{inputs.test_data}} "
            "--output_dir ${{outputs.eval_output}} "
            "--max_new_tokens 256 "
            "--temperature 0.7 "
            "--top_p 0.9 "
            "--batch_size 1 "
            "--mlflow_tracking"
        ),
        environment=environment,
        compute=compute_name,
        inputs={
            "test_data": test_data,
        },
        outputs={
            "eval_output": {
                "type": AssetTypes.URI_FOLDER,
            }
        },
        experiment_name=experiment_name,
        display_name=f"lora-eval-{base_model.split('/')[-1]}",
        description=f"Evaluate fine-tuned {base_model.split('/')[-1]} model",
    )
    
    # Submit the job
    returned_job = ml_client.jobs.create_or_update(job)
    
    print("\n" + "=" * 80)
    print("Evaluation Job Submitted!")
    print("=" * 80)
    print(f"Job name: {returned_job.name}")
    print(f"Job status: {returned_job.status}")
    print(f"Studio URL: {returned_job.studio_url}")
    print("=" * 80)
    
    return returned_job


def main():
    """Main function to submit evaluation job."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Submit evaluation job to Azure ML")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model (e.g., azureml://jobs/<job-id>/outputs/model_output)",
    )
    parser.add_argument(
        "--compute_name",
        type=str,
        default=None,
        help="Name of compute cluster (default: from .env AZURE_COMPUTE_NAME)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="lora-evaluation",
        help="Experiment name",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Azure ML - Model Evaluation Job Submission")
    print("=" * 80)
    
    # Create ML client
    print("\n[1/4] Connecting to Azure ML workspace...")
    ml_client = get_ml_client()
    
    # Create/update environment
    print("\n[2/4] Creating/updating environment...")
    environment = create_or_update_environment(ml_client)
    
    # Get evaluation data
    print("\n[3/4] Configuring evaluation data...")
    test_data = get_data_inputs(ml_client)
    
    # Submit job
    print("\n[4/4] Submitting evaluation job...")
    job = submit_evaluation_job(
        ml_client,
        environment,
        args.model_path,
        test_data,
        args.compute_name,
        args.experiment_name,
    )
    
    print("\nTo monitor the job:")
    print(f"  1. Visit: {job.studio_url}")
    print(f"  2. Or run: az ml job show --name {job.name}")
    
    print("\nJob submission complete!")


if __name__ == "__main__":
    main()
