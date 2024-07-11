import torch
import json


# Load all tensors
try:
    indices = torch.load("top_is_8000_16000.pt", map_location=torch.device("cpu"))
    values = torch.load("top_vs_8000_16000.pt", map_location=torch.device("cpu"))
except Exception as e:
    print(f"Error loading PyTorch files: {e}")
    exit(1)

print(indices)
print(indices.shape)
print(values)
print(values.shape)

feature = 8000

shifted_feature = feature - 8000

if shifted_feature < 0 or shifted_feature >= 16000:
    print("Feature out of range. Returning empty tensors.")
    print(torch.tensor([]))
    print(torch.tensor([]))
else:
    print(indices[shifted_feature])
    print(values[shifted_feature])
