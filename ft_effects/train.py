# %%
import os
import sys
sys.path.append(os.path.abspath('..'))

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import plotly.express as px
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformer_lens import HookedTransformer
from huggingface_hub import hf_hub_download

from steering.sae import JumpReLUSAE
# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

paths = [
    "effects/G2_2B_L12/65k_from_0",
    "effects/G2_2B_L12/65k_from_10k",
    "effects/G2_2B_L12/65k_from_20k",
    "effects/G2_2B_L12/65k_from_30k",
    "effects/G2_2B_L12/65k_from_40k",
]

features = []
effects = []

for path in paths:
    features.append(torch.load(os.path.join(path, "used_features.pt")))
    effects.append(torch.load(os.path.join(path, "all_effects.pt")))

features = torch.cat(features)
effects = torch.cat(effects)

val_features = features[-100:]
val_effects = effects[-100:]
features = features[:-100]
effects = effects[:-100]

# %%
print(features.shape)
print(effects.shape)
print(val_features.shape)
print(val_effects.shape)
# %%

@torch.no_grad()
def get_sae():
    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-res",
        filename="layer_12/width_16k/average_l0_82/params.npz",
        force_download=False)
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)
    sae.cpu()
    return sae

sae = get_sae()

# %%

class LinearAdapter(nn.Module):
    def __init__(self, sae):
        super().__init__()
        d_model, d_sae = sae.W_enc.shape 
        # self.W = nn.Parameter(sae.W_enc.detach().clone())
        self.W = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_model, d_sae)))
        self.b = nn.Parameter(torch.zeros(d_sae))

    def forward(self, x):
        return x @ self.W + self.b


# %%
dataset = TensorDataset(features, effects)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
val_dataset = TensorDataset(val_features, val_effects)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

def train(num_epochs, lr=1e-4):
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)

    scheduler = CosineAnnealingLR(opt, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        adapter.train()
        total_loss = 0
        num_batches = 0
        
        for batch_features, batch_effects in dataloader:
            opt.zero_grad()
            batch_features = batch_features.to(device)
            batch_effects = batch_effects.to(device)
            pred = adapter(batch_features)
            loss = F.mse_loss(pred, batch_effects)
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        
        # Validation
        adapter.eval()
        val_total_loss = 0
        val_num_batches = 0
        
        with torch.no_grad():
            for val_features, val_effects in val_dataloader:
                val_features = val_features.to(device)
                val_effects = val_effects.to(device)
                val_pred = adapter(val_features)
                val_loss = F.mse_loss(val_pred, val_effects)
                val_total_loss += val_loss.item()
                val_num_batches += 1
        
        scheduler.step()
        avg_val_loss = val_total_loss / val_num_batches
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")


# %%
adapter = LinearAdapter(sae)
adapter.to(device)
train(10)
# %%
# save the adapter
torch.save(adapter.state_dict(), "adapter.pt")
# %%

model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
hp = "blocks.12.hook_resid_post"

