# Computes scale at which loss is ==8 for each feature. 
# %%
import os
import sys
sys.path.append(os.path.abspath('..'))

import json
from steering.patch import patch_resid, generate
from steering.sae import JumpReLUSAE
from steering.evals_utils import evaluate_completions

from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from transformer_lens import HookedTransformer
from transformer_lens import utils as tutils
import numpy as np
from huggingface_hub import hf_hub_download, HfFileSystem

torch.set_grad_enabled(False)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)

# %%
hp = "blocks.12.hook_resid_post"
path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-2b-pt-res",
    filename="layer_12/width_16k/average_l0_82/params.npz",
    force_download=False)

params = np.load(path_to_params)
pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
sae.load_state_dict(pt_params)
sae.to(device)
# %%

batch_size = 64
data = tutils.load_dataset("NeelNanda/c4-code-20k", split="train")
tokenized_data = tutils.tokenize_and_concatenate(data, model.tokenizer, max_length=32)
tokenized_data = tokenized_data.shuffle(42)
loader = DataLoader(tokenized_data, batch_size=batch_size)

def compute_loss(steer, scales, n_batches=2):
    losses = []
    for scale in scales:
        total_loss = 0
        for batch_idx, batch in enumerate(loader):
            with model.hooks(fwd_hooks=[(hp, partial(patch_resid, steering=steer, scale=scale))]):
                loss = model(batch['tokens'], return_type='loss')
                total_loss += loss
            if batch_idx >= n_batches:
                break
        losses.append(total_loss.item() / n_batches)
        if total_loss.item()/n_batches > 8:
            break
    return losses

def compute_all(scales, from_i, to_i):
    all_losses = []
    try:
        for i in tqdm(range(from_i, to_i)):
            steer = sae.W_dec[i]
            losses = compute_loss(steer, scales, 2)
            all_losses.append(losses)
    except KeyboardInterrupt:
        print("\nInterrupt received. Saving partial results...")
    finally:
        if all_losses:
            # filename = f'losses_0_{len(all_losses)}.json'
            filename = f'losses_{from_i}_{len(all_losses)+from_i}.json'
            with open(filename, 'w') as f:
                json.dump(all_losses, f)
            print(f"Results saved to {filename}")
    return all_losses


# %%

all_losses = compute_all(list(range(20, 220, 20)), from_i=0, to_i=sae.W_dec.shape[0])


# %%