"""
Script to upload trained adapters to HuggingFace.
Loads adapter models from the expected paths based on current directory structure,
and uploads them to the sae-ts-effects dataset repo.
"""
import os
import torch
from huggingface_hub import HfApi

def upload_adapter(filepath, model_size, layer=12):
    """Upload adapter to HuggingFace."""
    repo_id = "schalnev/sae-ts-effects"
    remote_name = f"adapter_{model_size}_layer_{layer}.pt"
    
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=remote_name,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Successfully uploaded {filepath} to {repo_id} as {remote_name}")
    except Exception as e:
        print(f"Error uploading {filepath}: {e}")

if __name__ == "__main__":
    # Upload 2B model adapter
    adapter_2b_path = "adapter_2b_layer_12.pt"  # current path
    if os.path.exists(adapter_2b_path):
        print(f"\nUploading 2B adapter from {adapter_2b_path}...")
        upload_adapter(adapter_2b_path, "2b")
    else:
        print(f"Could not find 2B adapter at {adapter_2b_path}")

    # Upload 9B model adapter
    adapter_9b_path = "adapter_9b_layer_12.pt"  # current path
    if os.path.exists(adapter_9b_path):
        print(f"\nUploading 9B adapter from {adapter_9b_path}...")
        upload_adapter(adapter_9b_path, "9b")
    else:
        print(f"Could not find 9B adapter at {adapter_9b_path}")
        