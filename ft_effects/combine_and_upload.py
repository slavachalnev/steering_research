import os
import torch
from huggingface_hub import create_repo, HfApi

def combine_effects(model_size='2b'):
    """Combine multiple effects files into a single file."""
    if model_size == '2b':
        paths = [
            "effects/G2_2B_L12/65k_from_0",
            "effects/G2_2B_L12/65k_from_10k",
            "effects/G2_2B_L12/65k_from_20k",
            "effects/G2_2B_L12/65k_from_30k",
            "effects/G2_2B_L12/65k_from_40k",
        ]
    else:  # 9b
        paths = [
            "effects/G2_9B_L12/131k_from_0",
            "effects/G2_9B_L12/131k_from_16k",
            "effects/G2_9B_L12/131k_from_32k",
        ]

    features = []
    effects = []

    for path in paths:
        print(f"Loading from {path}")
        features.append(torch.load(os.path.join(path, "used_features.pt")))
        effects.append(torch.load(os.path.join(path, "all_effects.pt")))

    features = torch.cat(features)
    effects = torch.cat(effects)

    print(f"Combined shapes - Features: {features.shape}, Effects: {effects.shape}")
    
    combined_data = {
        'features': features,
        'effects': effects
    }
    
    output_file = f"effects_{model_size}.pt"
    torch.save(combined_data, output_file)
    print(f"Saved combined data to {output_file}")
    return output_file

def upload_to_hf(filepath, model_size):
    """Upload the combined file to HuggingFace."""
    # Note: Changed username to schalnev
    repo_id = "schalnev/sae-ts-effects"
    
    try:
        # Create the repo if it doesn't exist
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        
        # Upload the file
        api = HfApi()
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=f"effects_{model_size}.pt",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Successfully uploaded {filepath} to {repo_id}")
    except Exception as e:
        print(f"Error uploading to HuggingFace: {e}")
        print("Make sure you're logged in with 'huggingface-cli login'")

def upload_readme():
    """Upload README to the HuggingFace repo."""
    readme_content = """# SAE-TS Effects Dataset

This dataset contains pre-computed feature effects for the SAE-TS (SAE-Targeted Steering) method described in our paper [Improving Steering Vectors by Targeting Sparse Autoencoder Features](https://github.com/slavachalnev/sae-ts).

## Contents

The dataset contains two files:
- `effects_2b.pt`: Pre-computed effects for Gemma-2B model
- `effects_9b.pt`: Pre-computed effects for Gemma-9B model

Each file is a PyTorch saved dictionary containing:
- `features`: The steering vectors used (shape: [num_features, d_model])
- `effects`: The measured effects of each steering vector (shape: [num_features, d_sae])

## Usage

These files are used by the SAE-TS training script. To use them:

```python
import torch
from huggingface_hub import hf_hub_download

# Download the data
path = hf_hub_download(repo_id="schalnev/sae-ts-effects", 
                      filename="effects_2b.pt")  # or effects_9b.pt
                      
# Load the data
data = torch.load(path)
features = data['features']  
effects = data['effects']
```

## License

This dataset is released under the MIT license. See LICENSE for more information.
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    api = HfApi()
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id="schalnev/sae-ts-effects",
        repo_type="dataset"
    )
    
    os.remove("README.md")  # Clean up

if __name__ == "__main__":
    # Process 2B model effects
    print("\nProcessing 2B model effects...")
    file_2b = combine_effects('2b')
    upload_to_hf(file_2b, '2b')
    
    # Process 9B model effects
    print("\nProcessing 9B model effects...")
    file_9b = combine_effects('9b')
    upload_to_hf(file_9b, '9b')
    
    # Upload README
    print("\nUploading README...")
    upload_readme()