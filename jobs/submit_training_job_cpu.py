"""
Submit CPU-optimized training job to Azure ML.
Uses Standard_E4ds_v4 (4 cores, 32GB RAM).
"""

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, AmlCompute
from azure.identity import DefaultAzureCredential
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_or_get_compute(ml_client: MLClient, compute_config: dict):
    """Create or get CPU compute cluster."""
    compute_name = compute_config['compute']['name']
    
    try:
        compute = ml_client.compute.get(compute_name)
        print(f"Found existing compute: {compute_name}")
        return compute
    except Exception:
        print(f"Creating new CPU compute: {compute_name}")
        compute = AmlCompute(
            name=compute_name,
            type=compute_config['compute']['type'],
            size=compute_config['compute']['size'],
            min_instances=compute_config['compute']['min_instances'],
            max_instances=compute_config['compute']['max_instances'],
            idle_time_before_scale_down=compute_config['compute']['idle_time_before_scale_down'],
        )
        return ml_client.compute.begin_create_or_update(compute).result()


def main():
    # Load configuration
    training_config = load_config("../config/training_config_cpu.yaml")
    compute_config = load_config("../config/compute_config_cpu.yaml")
    
    # Initialize ML Client
    ml_client = MLClient.from_config(DefaultAzureCredential())
    
    print("=" * 80)
    print("Azure ML CPU Training Job Submission")
    print("=" * 80)
    
    # Create or get compute
    compute = create_or_get_compute(ml_client, compute_config)
    print(f"Compute: {compute.name} ({compute.size})")
    
    # Create environment
    env = Environment(
        name="lora-cpu-env",
        description="CPU-optimized environment for LoRA fine-tuning",
        conda_file="../environment/conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )
    
    # Create training command
    job = command(
        code="../src",
        command="python train_cpu.py \
            --model_name ${{inputs.model_name}} \
            --dataset_name ${{inputs.dataset_name}} \
            --max_samples ${{inputs.max_samples}} \
            --lora_r ${{inputs.lora_r}} \
            --lora_alpha ${{inputs.lora_alpha}} \
            --num_epochs ${{inputs.num_epochs}} \
            --batch_size ${{inputs.batch_size}} \
            --gradient_accumulation_steps ${{inputs.gradient_accumulation_steps}} \
            --learning_rate ${{inputs.learning_rate}} \
            --max_seq_length ${{inputs.max_seq_length}} \
            --output_dir ${{outputs.model_output}}",
        inputs={
            "model_name": training_config['model']['name'],
            "dataset_name": training_config['data']['dataset_name'],
            "max_samples": training_config['data']['max_samples'],
            "lora_r": training_config['lora']['r'],
            "lora_alpha": training_config['lora']['lora_alpha'],
            "num_epochs": training_config['training']['num_epochs'],
            "batch_size": training_config['training']['batch_size'],
            "gradient_accumulation_steps": training_config['training']['gradient_accumulation_steps'],
            "learning_rate": training_config['training']['learning_rate'],
            "max_seq_length": training_config['training']['max_seq_length'],
        },
        outputs={"model_output": "./outputs"},
        environment=env,
        compute=compute_config['compute']['name'],
        display_name="lora-cpu-training",
        description="CPU-optimized LoRA fine-tuning on Databricks Dolly 15K",
        experiment_name="lora-finetuning-cpu",
    )
    
    # Submit job
    print("\nSubmitting training job...")
    returned_job = ml_client.jobs.create_or_update(job)
    
    print("=" * 80)
    print("Job submitted successfully!")
    print(f"Job name: {returned_job.name}")
    print(f"Job status: {returned_job.status}")
    print(f"Studio URL: {returned_job.studio_url}")
    print("=" * 80)
    print("\nNOTE: CPU training will take longer than GPU training.")
    print("Expected time: 2-4 hours for 1000 samples on Standard_E4ds_v4")
    print("=" * 80)


if __name__ == "__main__":
    main()
