import torch

# Option 2: Using torch.load() (if it's a PyTorch tensor file)
try:
    tensor_data_indices = torch.load('all_indices.pt')
    print(tensor_data_indices[0])
except Exception as e:
    print(f"Error loading PyTorch file: {e}")

try:
    tensor_data_values = torch.load('all_values.pt')
    print(tensor_data_values[0])
except Exception as e:
    print(f"Error loading PyTorch file: {e}")

