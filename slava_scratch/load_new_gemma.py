# %%
import os
import sys
sys.path.append(os.path.abspath('..'))

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
import numpy as np
from huggingface_hub import hf_hub_download, HfFileSystem

torch.set_grad_enabled(False)

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)

# %%

def get_gemma2_2b_SAE_path(layer, width=16, closest_l0=100):
    fs = HfFileSystem()
    all_paths = fs.glob("google/gemma-scope-2b-pt-res/**/params.npz")
    candidate_paths = [p for p in all_paths if f'layer_{layer}/width_{width}k/average_l0_' in p]
    # get the l0 value from the path
    l0_values = [int(p.split('average_l0_')[1].split('/')[0]) for p in candidate_paths]
    # find the one closest to closest_l0
    idx = np.argmin(np.abs(np.array(l0_values) - closest_l0))
    desire_l0 = l0_values[idx]
    desire_path = candidate_paths[idx]
    return desire_l0, desire_path

for layer in range(model.cfg.n_layers):
    l0, path = get_gemma2_2b_SAE_path(layer)
    print(layer, l0, path)

# %%

path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-2b-pt-res",
    filename="layer_12/width_16k/average_l0_82/params.npz",
    force_download=False,
)

params = np.load(path_to_params)
pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

# %%
class JumpReLUSAE(nn.Module):
  def __init__(self, d_model, d_sae):
    # Note that we initialise these to zeros because we're loading in pre-trained weights.
    # If you want to train your own SAEs then we recommend using blah
    super().__init__()
    self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
    self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
    self.threshold = nn.Parameter(torch.zeros(d_sae))
    self.b_enc = nn.Parameter(torch.zeros(d_sae))
    self.b_dec = nn.Parameter(torch.zeros(d_model))

  def encode(self, input_acts):
    pre_acts = input_acts @ self.W_enc + self.b_enc
    mask = (pre_acts > self.threshold)
    acts = mask * torch.nn.functional.relu(pre_acts)
    return acts

  def decode(self, acts):
    return acts @ self.W_dec + self.b_dec

  def forward(self, acts):
    acts = self.encode(acts)
    recon = self.decode(acts)
    return recon

# %%

sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
sae.load_state_dict(pt_params)

# %%



# %%
# %%
