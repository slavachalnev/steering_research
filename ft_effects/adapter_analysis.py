# %%
import os
import sys
sys.path.append(os.path.abspath('..'))

import torch
from torch import nn
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from tqdm import tqdm
import einops
import scipy.linalg as linalg

from transformer_lens import HookedTransformer

from ft_effects.utils import get_sae, compute_scores, LinearAdapter

torch.set_grad_enabled(False)
# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sae = get_sae()
# %%

linear_adapter = LinearAdapter(sae.W_enc.shape[0], sae.W_enc.shape[1])
linear_adapter.load_state_dict(torch.load("linear_adapter.pt"))

# # %%
# ### save correction bias
# b = linear_adapter.W @ linear_adapter.b
# b = b / torch.norm(b)
# torch.save(b, "correction_bias_layer_12.pt")

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

    return torch.tensor(R).to(device)

# remove ft_id 14455 and 4230 from normed_adapter
censored_adapter = torch.cat([normed_adapter[:, :14455], normed_adapter[:, 14456:4230], normed_adapter[:, 4231:]], dim=1)
censored_encoder = torch.cat([normed_encoder[:, :14455], normed_encoder[:, 14456:4230], normed_encoder[:, 4231:]], dim=1)
censored_decoder = torch.cat([normed_decoder[:14455], normed_decoder[14456:4230], normed_decoder[4231:]], dim=0)

# %%
R = analyze_transformations(censored_adapter, censored_encoder, include_bias=False)

# %%

R_dec = analyze_transformations(censored_adapter, censored_decoder.T, include_bias=False)

# %%

px.imshow(R_dec[:200, :200].cpu(), title="Rotation Matrix", color_continuous_scale="RdBu", color_continuous_midpoint=0).show()
px.imshow(R_dec[-200:, -200:].cpu(), title="Rotation Matrix", color_continuous_scale="RdBu", color_continuous_midpoint=0).show()

# %%
# histogram of diagonal values of R_dec
px.histogram(torch.diag(R_dec).cpu().numpy(), title="Diagonal of Rotation Matrix", labels={'value': 'Value', 'count': 'Frequency'}, marginal='box').show()

# %%
# check that R_dec is a rotation matrix
print(torch.norm(R_dec @ R_dec.T - torch.eye(R_dec.shape[0])))
px.imshow((R_dec @ R_dec.T)[:30, :30], title="R_dec @ R_dec.T", color_continuous_scale="RdBu", color_continuous_midpoint=0).show()

# %%
model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)

# %%
# ft_name = "london"
# ft_id = 14455 # london
# criterion = "Text mentions London or anything related to London."

ft_name = "wedding"
ft_id = 4230 # wedding
criterion = "Text mentions weddings or anything related to weddings."

# ft_name = "recipe"
# ft_id = 1
# criterion = "The text is a recipe or a description of a scientific method. Specifically, it mentions serving in a dish or pouring into a beaker etc."


# %%
sae.to(device)
# transformed_steer = R_dec.T @ R_dec.T @ sae.W_dec[ft_id] # two rotations
# transformed_steer = R_dec.T @ sae.W_dec[ft_id] # one rotation
transformed_steer = R_dec @ sae.W_dec[ft_id]  # rotate in opposite direction

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

# random steer with adapter bias
random_steer = torch.randn(sae.W_enc.shape[0]).to(device)
random_steer = random_steer / torch.norm(random_steer)
random_steer_with_bias = random_steer - 1 * b
random_steer_with_bias = random_steer_with_bias / torch.norm(random_steer_with_bias)
_ = compute_scores(random_steer_with_bias, model, f"random_steer_with_bias", criterion, scales=list(range(0, 220, 20)))

# compared to no adapter bias
_ = compute_scores(random_steer, model, f"random_steer_no_bias", criterion, scales=list(range(0, 220, 20)))

# %%


# %%

# %%



