"""
Submit evaluation job to Azure Machine Learning.

This script submits an evaluation job to test the fine-tuned model on Azure ML.
"""

import os
import argparse
from pathlib import Path
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Submit evaluation job to Azure ML")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Azure ML path to fine-tuned model (e.g., azureml://jobs/<job-id>/outputs/model_output/final_model)",
    )
    parser.add_argument(
        "--compute_name",
        type=str,
        default="gpu-cluster",
        help="Name of compute cluster",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="lora-evaluation",
        help="Experiment name",
    )
    
    return parser.parse_args()


def get_ml_client():
    """Create and return MLClient instance."""
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID", "<your-subscription-id>")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP", "<your-resource-group>")
    workspace_name = os.getenv("AZURE_WORKSPACE_NAME", "<your-workspace-name>")
    
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    
    print(f"Connected to workspace: {workspace_name}")
    return ml_client


def get_environment(ml_client: MLClient):
    """Get the existing environment or create if needed."""
    env_name = "lora-finetune-env"
    
    try:
        # Try to get latest version
        env = ml_client.environments.get(name=env_name, label="latest")
        print(f"Using existing environment: {env.name}:{env.version}")
        return f"{env.name}:{env.version}"
    except Exception:
        # Create if doesn't exist
        conda_file_path = Path(__file__).parent.parent / "environment" / "conda.yaml"
        
        env = Environment(
            name=env_name,
            description="Environment for LoRA evaluation",
            image="mcr.microsoft.com/azureml/curated/acft-hf-nlp-gpu:latest",
            conda_file=str(conda_file_path),
        )
        
        env = ml_client.environments.create_or_update(env)
        print(f"Environment created: {env.name}:{env.version}")
        return f"{env.name}:{env.version}"


def prepare_test_data():
    """Prepare test data input."""
    data_dir = Path(__file__).parent.parent / "data"
    
    test_data = Input(
        type=AssetTypes.URI_FILE,
        path=str(data_dir / "validation.jsonl"),
    )
    
    print("Test data prepared")
    return test_data


def submit_evaluation_job(
    ml_client: MLClient,
    environment: str,
    model_path: str,
    test_data: Input,
    compute_name: str,
    experiment_name: str,
):
    """Submit the evaluation job to Azure ML."""
    
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
            "--model_path ${{inputs.model_path}} "
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
            "model_path": Input(
                type=AssetTypes.URI_FOLDER,
                path=model_path,
            ),
            "test_data": test_data,
        },
        outputs={
            "eval_output": {
                "type": AssetTypes.URI_FOLDER,
                "mode": "rw_mount",
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
    args = parse_args()
    
    print("=" * 80)
    print("Azure ML - Model Evaluation Job Submission")
    print("=" * 80)
    
    # Create ML client
    print("\n[1/4] Connecting to Azure ML workspace...")
    ml_client = get_ml_client()
    
    # Get environment
    print("\n[2/4] Getting environment...")
    environment = get_environment(ml_client)
    
    # Prepare test data
    print("\n[3/4] Preparing test data...")
    test_data = prepare_test_data()
    
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
