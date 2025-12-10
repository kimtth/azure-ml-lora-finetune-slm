"""
Upload training data directly to Azure Blob Storage using identity-based authentication.

This script uploads data files directly to the workspace storage account,
bypassing the Azure ML SDK's datastore upload mechanism.
"""

import os
import warnings
import logging
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from dotenv import load_dotenv

# Suppress experimental class warnings and SDK debug logging
warnings.filterwarnings("ignore", message=".*experimental class.*")
logging.getLogger("azure.ai.ml").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)

load_dotenv()


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


def upload_file_to_blob(blob_service_client, container_name, local_path, blob_path):
    """Upload a file to blob storage using identity-based authentication."""
    
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_path)
    
    print(f"  Uploading {local_path.name} to {blob_path}...")
    
    with open(local_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    
    print("  [OK] Upload complete")
    
    return blob_client.url


def main():
    """Main function to upload data to blob storage."""
    print("=" * 80)
    print("Azure ML - Upload Data to Blob Storage (Identity-Based)")
    print("=" * 80)
    
    # Get ML client to retrieve storage info
    print("\n[1/3] Connecting to Azure ML workspace...")
    ml_client = get_ml_client()
    
    # Get default datastore info
    print("\n[2/3] Getting storage account information...")
    default_datastore = ml_client.datastores.get_default()
    
    account_name = default_datastore.account_name
    container_name = default_datastore.container_name
    
    print(f"  Storage account: {account_name}")
    print(f"  Container: {container_name}")
    
    # Create blob service client with identity-based authentication
    account_url = f"https://{account_name}.blob.core.windows.net"
    credential = DefaultAzureCredential()
    
    blob_service_client = BlobServiceClient(
        account_url=account_url,
        credential=credential
    )
    
    print("  Authentication: Identity-based (DefaultAzureCredential)")
    
    # Upload data files
    print("\n[3/3] Uploading data files...")
    
    data_dir = Path(__file__).parent.parent / "data"
    
    # Create a folder in the blob storage for our training data
    blob_folder = "LocalUpload/lora-training-data"
    
    # Upload training data
    train_blob_path = f"{blob_folder}/train.jsonl"
    upload_file_to_blob(
        blob_service_client,
        container_name,
        data_dir / "train.jsonl",
        train_blob_path
    )
    
    # Upload validation data
    val_blob_path = f"{blob_folder}/validation.jsonl"
    upload_file_to_blob(
        blob_service_client,
        container_name,
        data_dir / "validation.jsonl",
        val_blob_path
    )
    
    # Construct Azure ML URIs
    train_uri = f"azureml://subscriptions/{ml_client.subscription_id}/resourcegroups/{ml_client.resource_group_name}/workspaces/{ml_client.workspace_name}/datastores/{default_datastore.name}/paths/{train_blob_path}"
    val_uri = f"azureml://subscriptions/{ml_client.subscription_id}/resourcegroups/{ml_client.resource_group_name}/workspaces/{ml_client.workspace_name}/datastores/{default_datastore.name}/paths/{val_blob_path}"
    
    print("\n" + "=" * 80)
    print("Upload Complete!")
    print("=" * 80)
    print("\nData uploaded to blob storage:")
    print(f"  Training:   {train_blob_path}")
    print(f"  Validation: {val_blob_path}")
    print("\nAzure ML URIs:")
    print(f"  Training:   {train_uri}")
    print(f"  Validation: {val_uri}")
    print("\nYou can now submit your training job!")
    print("  python submit_training_job.py")
    print("=" * 80)
    
    return train_uri, val_uri


if __name__ == "__main__":
    main()
