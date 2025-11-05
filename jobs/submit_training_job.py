"""
Submit fine-tuning job to Azure Machine Learning.

This script submits a LoRA fine-tuning job to Azure ML workspace.
"""

import os
from pathlib import Path
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes


def get_ml_client():
    """Create and return MLClient instance."""
    # Get configuration from environment variables or modify directly
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


def create_or_update_environment(ml_client: MLClient):
    """Create or update the training environment."""
    env_name = "lora-finetune-env"
    
    # Path to conda.yaml
    conda_file_path = Path(__file__).parent.parent / "environment" / "conda.yaml"
    
    env = Environment(
        name=env_name,
        description="Environment for LoRA fine-tuning with PEFT and bitsandbytes",
        image="mcr.microsoft.com/azureml/curated/acft-hf-nlp-gpu:latest",
        conda_file=str(conda_file_path),
    )
    
    env = ml_client.environments.create_or_update(env)
    print(f"Environment created/updated: {env.name}:{env.version}")
    
    return f"{env.name}:{env.version}"


def upload_data():
    """Upload training and validation data to workspace."""
    data_dir = Path(__file__).parent.parent / "data"
    
    # Upload training data
    train_data = Input(
        type=AssetTypes.URI_FILE,
        path=str(data_dir / "train.jsonl"),
    )
    
    # Upload validation data
    val_data = Input(
        type=AssetTypes.URI_FILE,
        path=str(data_dir / "validation.jsonl"),
    )
    
    print("Data inputs prepared")
    return train_data, val_data


def submit_training_job(
    ml_client: MLClient,
    environment: str,
    train_data: Input,
    val_data: Input,
    compute_name: str = "gpu-cluster",
    experiment_name: str = "lora-finetuning",
):
    """Submit the training job to Azure ML."""
    
    # Load training config to get model name
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model_name = config['model']['name']
    dataset_name = config['data']['dataset_name']
    
    # Path to training script
    code_dir = Path(__file__).parent.parent / "src"
    
    # Configure training command
    job = command(
        code=str(code_dir),
        command=(
            "python train.py "
            f"--model_name {model_name} "
            "--train_data ${{inputs.train_data}} "
            "--validation_data ${{inputs.validation_data}} "
            "--output_dir ${{outputs.model_output}} "
            "--num_epochs 3 "
            "--batch_size 4 "
            "--learning_rate 2e-4 "
            "--lora_r 16 "
            "--lora_alpha 32 "
            "--lora_dropout 0.05 "
            "--max_seq_length 512 "
            "--gradient_accumulation_steps 4 "
            "--gradient_checkpointing "
            "--load_in_4bit "
            "--mlflow_tracking"
        ),
        environment=environment,
        compute=compute_name,
        inputs={
            "train_data": train_data,
            "validation_data": val_data,
        },
        outputs={
            "model_output": {
                "type": AssetTypes.URI_FOLDER,
                "mode": "rw_mount",
            }
        },
        experiment_name=experiment_name,
        display_name=f"lora-finetune-{model_name.split('/')[-1]}",
        description=f"Fine-tune {model_name.split('/')[-1]} with LoRA on {dataset_name.split('/')[-1]}",
    )
    
    # Submit the job
    returned_job = ml_client.jobs.create_or_update(job)
    
    print("\n" + "=" * 80)
    print("Training Job Submitted!")
    print("=" * 80)
    print(f"Job name: {returned_job.name}")
    print(f"Job status: {returned_job.status}")
    print(f"Studio URL: {returned_job.studio_url}")
    print("=" * 80)
    
    return returned_job


def main():
    """Main function to submit training job."""
    print("=" * 80)
    print("Azure ML - LoRA Fine-tuning Job Submission")
    print("=" * 80)
    
    # Create ML client
    print("\n[1/4] Connecting to Azure ML workspace...")
    ml_client = get_ml_client()
    
    # Create/update environment
    print("\n[2/4] Creating/updating environment...")
    environment = create_or_update_environment(ml_client)
    
    # Upload data
    print("\n[3/4] Preparing data inputs...")
    train_data, val_data = upload_data()
    
    # Submit job
    print("\n[4/4] Submitting training job...")
    
    # Load config for dynamic naming
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_name = config['model']['name'].split('/')[-1]
    
    job = submit_training_job(
        ml_client,
        environment,
        train_data,
        val_data,
        compute_name="gpu-cluster",  # Change to your compute cluster name
        experiment_name=f"lora-finetuning-{model_name}",
    )
    
    print("\nTo monitor the job:")
    print(f"  1. Visit: {job.studio_url}")
    print(f"  2. Or run: az ml job show --name {job.name}")
    
    print("\nJob submission complete!")


if __name__ == "__main__":
    main()
