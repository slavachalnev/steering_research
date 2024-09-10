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
from ft_effects.utils import get_sae, compute_scores

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

def analyze_transformations(X, Y, include_bias=True, linear_threshold=0.01, rotation_threshold=0.01):
    # Ensure X and Y are numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    
    # Check shapes
    assert X.shape == Y.shape, "X and Y must have the same shape"
    d, n = X.shape
    print(f"Input shape: {X.shape}")
    
    # Linear transformation (with optional bias)
    if include_bias:
        # Solve AX + b = Y for A and b
        X_homogeneous = np.vstack([X, np.ones((1, n))])
        A_with_bias = Y @ np.linalg.pinv(X_homogeneous)
        A = A_with_bias[:, :-1]
        b = A_with_bias[:, -1:]
        Y_pred_linear = A @ X + b
    else:
        # Solve AX = Y for A
        A = Y @ np.linalg.pinv(X)
        b = None
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

    return A, b, R, mse_linear, mse_rotation

# %%
A, b, R, mse_linear, mse_rotation = analyze_transformations(normed_adapter, normed_encoder, include_bias=False)

# %%

A_dec, b_dec, R_dec, mse_linear_dec, mse_rotation_dec = analyze_transformations(normed_adapter, normed_decoder.T, include_bias=False)

# %%

# # save A_dec as torch tensor
# torch.save(torch.tensor(A_dec), "A_dec.pt")
# torch.save(torch.tensor(b_dec), "b_dec.pt")

# save R_dec as torch tensor
torch.save(torch.tensor(R_dec), "R_dec.pt")


# %%
sae = get_sae()
sae.to(device)
model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)

# %%
ft_name = "london"
ft_id = 14455 # london
criterion = "Text mentions London or anything related to London."

# ft_name = "wedding"
# ft_id = 4230 # wedding
# criterion = "Text mentions weddings or anything related to weddings."

# %%
R_dec = torch.tensor(R_dec).to(device)
print(R_dec.device)
transformed_steer = R_dec.T @ sae.W_dec[ft_id]
transformed_steer = transformed_steer / torch.norm(transformed_steer)

# with adapter bias
adapter = LinearAdapter(sae.W_enc.shape[0], sae.W_enc.shape[1])
adapter.load_state_dict(torch.load("linear_adapter.pt"))
adapter.to(device)
b = adapter.W @ adapter.b
b = b / torch.norm(b)
print(b.shape)
transformed_steer = R_dec.T @ sae.W_dec[ft_id]
transformed_steer = transformed_steer / torch.norm(transformed_steer)
transformed_steer = transformed_steer - 1 * b
transformed_steer = transformed_steer / torch.norm(transformed_steer)

_ = compute_scores(transformed_steer, model, f"{ft_name}_rotated_decoder_with_bias", criterion, scales=list(range(0, 220, 20)))
# %%



