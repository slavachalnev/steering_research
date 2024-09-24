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

from ft_effects.utils import get_sae, compute_scores, LinearAdapter, approximate_adapter

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

px.histogram(linear_adapter.b.cpu().numpy(), title="Bias of Linear Adapter").show()
# %%

topk_indices = linear_adapter.b.cpu().numpy().argsort()[-10:][::-1]
topk_values = linear_adapter.b.cpu().numpy()[topk_indices]

bottomk_indices = linear_adapter.b.cpu().numpy().argsort()[:10]
bottomk_values = linear_adapter.b.cpu().numpy()[bottomk_indices]

print("| Rank | Top Indices | Top Values | Bottom Indices | Bottom Values |")
print("|------|-------------|------------|----------------|---------------|")
for i in range(10):
    print(f"| {i+1:<4} | {topk_indices[i]:<11} | {topk_values[i]:>10.4f} | {bottomk_indices[i]:<14} | {bottomk_values[i]:>13.4f} |")

# %%
# find the linear adapter columns with norm less than 0.05
small_idxs = torch.where(torch.norm(linear_adapter.W, dim=0) < 0.05)[0]
print(small_idxs[:10])

# %%
# # remove the normed adapter columns with norm less than 0.05
# large_idxs = torch.where(torch.norm(linear_adapter.W, dim=0) >= 0.05)[0]
# normed_adapter = normed_adapter[:, large_idxs].cpu()
# normed_encoder = normed_encoder[:, large_idxs].cpu()
# normed_decoder = normed_decoder[large_idxs].cpu()


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

# %%
R = analyze_transformations(normed_adapter, normed_encoder, include_bias=False)

# %%

R_dec = analyze_transformations(normed_adapter, normed_decoder.T, include_bias=False)

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
hp = "blocks.12.hook_resid_post"

ft_name = "london"
ft_id = 14455 # london
criterion = "Text mentions London or anything related to London."

# ft_name = "wedding"
# ft_id = 4230 # wedding
# criterion = "Text mentions weddings or anything related to weddings."

# ft_name = "recipe"
# ft_id = 1
# criterion = "The text is a recipe or a description of a scientific method. Specifically, it mentions serving in a dish or pouring into a beaker etc."

# ft_name = "citations"
# ft_id = 115
# criterion = "Text contains academic citations or references."

# %%
sae.to(device)

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

_ = compute_scores(transformed_steer, model, f"{ft_name}_rotated_decoder_with_bias", criterion, hp, scales=list(range(0, 220, 20)))

# %%

# random steer with adapter bias
random_steer = torch.randn(sae.W_enc.shape[0]).to(device)
random_steer = random_steer / torch.norm(random_steer)
random_steer_with_bias = random_steer - 1 * b
random_steer_with_bias = random_steer_with_bias / torch.norm(random_steer_with_bias)
_ = compute_scores(random_steer_with_bias, model, f"random_steer_with_bias", criterion, hp, scales=list(range(0, 220, 20)))

# compared to no adapter bias
_ = compute_scores(random_steer, model, f"random_steer_no_bias", criterion, hp, scales=list(range(0, 220, 20)))


# %%
##################################################################################
sparse_adapter_b = adapter.b.clone()
sparse_adapter_b[7507] = 0
sparse_adapter_b[1041] = 0
        
sparse_b = adapter.W @ sparse_adapter_b
sparse_b = sparse_b / torch.norm(sparse_b)

# compare optim steer with and without bias sparsity
steer = linear_adapter.W[:, ft_id].to(device)
steer = steer / torch.norm(steer)
steer_with_bias = steer - 1 * b
steer_with_bias = steer_with_bias / torch.norm(steer_with_bias)
steer_with_sparse_bias = steer - 1 * sparse_b
steer_with_sparse_bias = steer_with_sparse_bias / torch.norm(steer_with_sparse_bias)

# _ = compute_scores(steer, model, f"{ft_name}_optim", criterion, hp, scales=list(range(0, 220, 20)))
_ = compute_scores(steer_with_bias, model, f"{ft_name}_optim_with_bias", criterion, hp, scales=list(range(0, 220, 20)))
# _ = compute_scores(steer_with_sparse_bias, model, f"{ft_name}_optim_with_sparse_bias", criterion, hp, scales=list(range(0, 220, 20)))


# %%



class Approximator(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.W1 = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_model, d_hidden)))
        self.W2 = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_hidden, d_model)))
    
    def forward(self, x):
        return x + x @ self.W1 @ self.W2


def approximate_adapter(feats, adapter) -> Approximator:
    feats = feats.detach()
    out_feats = []
    b_to_use = adapter.W @ adapter.b
    for i, f in enumerate(feats):
        out_f = adapter.W[:, i] / torch.norm(adapter.W[:, i]) - b_to_use / torch.norm(b_to_use)
        out_f = out_f / torch.norm(out_f)
        out_feats.append(out_f)
    out_feats = torch.stack(out_feats, dim=0).detach()

    # only keep large values
    large_idxs = torch.where(torch.norm(out_feats, dim=0) >= 0.05)[0]
    out_feats = out_feats[:, large_idxs]

    # train approximator
    approx = Approximator(feats.shape[1], 64)
    approx.to(feats.device)
    optimizer = torch.optim.Adam(approx.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for i in range(int(5e3)):
        optimizer.zero_grad()
        out = approx(feats)
        loss = criterion(out, out_feats)
        loss.backward()
        if i % 100 == 0:
            print(loss.item())
        optimizer.step()
    return approx



# %%
linear_adapter.to(device)
# torch.set_grad_enabled(True)
with torch.enable_grad():
    approx = approximate_adapter(sae.W_dec.to(device), linear_adapter)

approx = approx.cpu()

# %%

approx_steer = approx(sae.W_dec[ft_id].unsqueeze(0))
print(approx_steer.shape)
print(approx_steer.device)
print(torch.norm(approx_steer))

approx_steer = approx_steer / torch.norm(approx_steer)

_ = compute_scores(approx_steer, model, f"{ft_name}_approx_adapter", criterion, hp, scales=list(range(0, 220, 20)))
# %%



