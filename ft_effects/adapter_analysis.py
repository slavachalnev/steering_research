# %%
import os
import sys
sys.path.append(os.path.abspath('..'))

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
import json
import einops
import scipy.linalg as linalg

from transformer_lens import HookedTransformer
from huggingface_hub import hf_hub_download

from steering.sae import JumpReLUSAE
from steering.patch import patch_resid

from baselines.analysis import steer_model
from steering.evals_utils import multi_criterion_evaluation

from ft_effects.train import LinearAdapter
from ft_effects.utils import get_sae

torch.set_grad_enabled(False)
# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sae = get_sae()
# %%

linear_adapter = LinearAdapter(sae.W_enc.shape[0], sae.W_enc.shape[1])
linear_adapter.load_state_dict(torch.load("linear_adapter.pt"))

# %%
normed_adapter = linear_adapter.W / torch.norm(linear_adapter.W, dim=0)
normed_encoder = sae.W_enc / torch.norm(sae.W_enc, dim=0)
normed_decoder = sae.W_dec / torch.norm(sae.W_dec, dim=1, keepdim=True)

# %%

px.histogram(torch.norm(linear_adapter.W, dim=0).cpu().numpy()).show()

# %%

print(linear_adapter.W.shape) # [d_model, d_sae]
print(sae.W_enc.shape) # [d_model, d_sae]
print(sae.W_dec.shape) # [d_sae, d_model]

encoder_sims = einops.einsum(normed_adapter, normed_encoder, "d_model d_sae, d_model d_sae -> d_sae")

fig = px.histogram(encoder_sims.cpu().numpy(), 
                   title="Similarity of Predictor Column to Corresponding Encoder",
                   labels={'value': 'Similarity', 'count': 'Frequency'},
                   marginal='box')  # Adding a box plot for additional insight
fig.update_layout(xaxis_title="Similarity", yaxis_title="Frequency")
fig.show()


# %%

def analyze_transformations(X, Y, linear_threshold=0.01, rotation_threshold=0.01):
    # Ensure X and Y are numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    
    # Check shapes
    assert X.shape == Y.shape, "X and Y must have the same shape"
    d, n = X.shape
    print(f"Input shape: {X.shape}")
    
    # Linear transformation
    # Solve AX = Y for A
    A = Y @ np.linalg.pinv(X)
    Y_pred_linear = A @ X
    mse_linear = np.mean((Y - Y_pred_linear) ** 2)
    
    print(f"Linear transformation MSE: {mse_linear}")
    print(f"Linear transformation is good: {mse_linear < linear_threshold}")

    # Rotation (Orthogonal Procrustes problem)
    M = Y @ X.T
    U, _, Vt = linalg.svd(M)
    R = U @ Vt
    Y_pred_rotation = R @ X
    mse_rotation = np.mean((Y - Y_pred_rotation) ** 2)
    
    print(f"Rotation MSE: {mse_rotation}")
    print(f"Rotation is good: {mse_rotation < rotation_threshold}")

    return A, R, mse_linear, mse_rotation

A, R, mse_linear, mse_rotation = analyze_transformations(normed_adapter, normed_encoder)

# %%

A_dec, R_dec, mse_linear_dec, mse_rotation_dec = analyze_transformations(normed_adapter, normed_decoder.T)

# %%

# save A_dec as torch tensor
torch.save(torch.tensor(A_dec), "A_dec.pt")


# %%
