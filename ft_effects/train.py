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

from transformer_lens import HookedTransformer
from huggingface_hub import hf_hub_download

from steering.sae import JumpReLUSAE
from steering.patch import patch_resid

from baselines.analysis import steer_model
from steering.evals_utils import multi_criterion_evaluation

from ft_effects.utils import get_sae, LinearAdapter, compute_scores
# %%

BIG_MODEL = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sae = get_sae(big_model=BIG_MODEL)

hp = "blocks.12.hook_resid_post"

# %%



class HybridAdapter(nn.Module):
    def __init__(self, W_enc, d_model, d_sae):
        super().__init__()
        # self.W = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_model, d_model)))

        self.W1 = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_model, 4*d_model)))
        self.W2 = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(4*d_model, d_model)))
        self.b1 = nn.Parameter(torch.zeros(4*d_model))
        self.b2 = nn.Parameter(torch.zeros(d_model))

        self.b_final = nn.Parameter(torch.zeros(d_sae))

        self.W_enc = W_enc.clone().detach()
        self.W_enc.requires_grad_(False)
        self.W_enc = self.W_enc.to(device)

    def forward(self, x):
        h = F.relu(x @ self.W1 + self.b1)
        h = h @ self.W2 + self.b2
        x = h # no residual connection
        x = x @ self.W_enc
        return x + self.b_final

    # def forward(self, x):
    #     x = x@self.W
    #     x = x @ self.W_enc
    #     return x + self.b_final
        


def train(num_epochs, lr=1e-4, weight_decay=1e-5):
    if BIG_MODEL:
        paths = [
            "effects/G2_9B_L12/131k_from_0",
            "effects/G2_9B_L12/131k_from_64k",
        ]
    else:
        paths = [
            "effects/G2_2B_L12/65k_from_0",
            "effects/G2_2B_L12/65k_from_10k",
            "effects/G2_2B_L12/65k_from_20k",
            "effects/G2_2B_L12/65k_from_30k",
            "effects/G2_2B_L12/65k_from_40k",

            # "effects/G2_2B_L12/16k_from_0",
            # "effects/G2_2B_L12/sample_and_combine_16k",

            # "effects/G2_2B_L12/random",
            # "effects/G2_2B_L12/random_2",
            # "effects/G2_2B_L12/random_3",
            # "effects/G2_2B_L12/random_4",
            # "effects/G2_2B_L12/random_5",
        ]

    features = []
    effects = []

    for path in paths:
        features.append(torch.load(os.path.join(path, "used_features.pt")))
        effects.append(torch.load(os.path.join(path, "all_effects.pt")))

    features = torch.cat(features)
    effects = torch.cat(effects)

    # normalise features to have norm 1
    features = features / torch.norm(features, dim=-1, keepdim=True)
    n_val = 100

    val_features = features[-n_val:]
    val_effects = effects[-n_val:]
    features = features[:-n_val]
    effects = effects[:-n_val]

    dataset = TensorDataset(features, effects)
    val_dataset = TensorDataset(val_features, val_effects)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)

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
if __name__ == "__main__":
    adapter = LinearAdapter(sae.W_enc.shape[0], sae.W_enc.shape[1])
    # adapter = HybridAdapter(sae.W_enc, sae.W_enc.shape[0], sae.W_enc.shape[1])
    # adapter = HybridAdapter(sae.W_dec.T, sae.W_enc.shape[0], sae.W_enc.shape[1])

    adapter.to(device)
    train(15, lr=2e-4)

# # %%

    # if BIG_MODEL:
    #     torch.save(adapter.state_dict(), "linear_adapter_9B.pt")
    # else:
    #     torch.save(adapter.state_dict(), "linear_adapter_2B.pt")
# %%

def find_optimal_steer(adapter, target, d_model,
                       n_steps=int(1e4),
                       return_intermediate=False,
                       target_scale=1,
                       init_steer=None,
                       ):
    # steer = torch.zeros(d_model, requires_grad=True, device=device)
    if init_steer is None:
        steer = nn.Parameter(torch.zeros(d_model, requires_grad=True, device=device))
    else:
        steer = nn.Parameter(init_steer.clone().detach().to(device))

    steer.data = steer.data / torch.norm(steer.data)

    steer = steer.to(device)
    target = target.to(device)
    target = target * target_scale
    optim = torch.optim.Adam([steer], lr=1e-3)
    intermediates = []
    pbar = tqdm(range(n_steps), desc="Optimizing")
    for step in pbar:
        optim.zero_grad()
        pred = adapter(steer) ###
        loss = F.mse_loss(pred, target)
        loss.backward()
        optim.step()
        if step % 1000 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")
        steer.data = steer.data / torch.norm(steer.data)
        if step % 10 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            if return_intermediate:
                intermediates.append(steer.detach().cpu())
    if return_intermediate:
        return steer.detach(), intermediates
    else:
        return steer.detach()

def single_step_steer(adapter, target, bias_scale=1):
    steer_vec = adapter.W @ target.to(device)
    steer_vec = steer_vec / torch.norm(steer_vec)
    bias_vec = adapter.W @ adapter.b
    bias_vec = bias_vec / torch.norm(bias_vec)
    bias_vec = bias_vec * bias_scale
    return steer_vec - bias_vec

# %%
target = torch.zeros(sae.W_enc.shape[1])
target_value = 1


# ft_name = "london"
# ft_id = 14455 # london
# target[ft_id] = target_value # london
# criterion = "Text mentions London or anything related to London."

# ft_name = "wedding"
# ft_id = 4230 # wedding
# target[ft_id] = target_value # wedding
# criterion = "Text mentions weddings or anything related to weddings."

# ft_name = "bridge"
# ft_id = 7272
# target[ft_id] = target_value
# criterion = "Text mentions bridges or anything related to bridges."

# ft_name = "bird"
# ft_id = 1842
# target[ft_id] = target_value
# criterion = "Text mentions birds or anything related to birds."

# ft_name = "london_bridge"
# ft_id = 14455
# target[ft_id] = target_value
# target[7272] = target_value
# criterion = "Text mentions bridges in London."

# ft_name = "recipe"
# ft_id = 1
# target[ft_id] = target_value
# criterion = "The text is a recipe or a description of a scientific method. Specifically, it mentions serving in a dish or pouring into a beaker etc."


##########
# big model

# ft_name = "london"
# ft_id = 6915
# target[ft_id] = target_value
# criterion = "Text mentions London or anything related to London."

ft_name = "wedding"
ft_id = 2822
target[ft_id] = target_value
criterion = "Text mentions weddings or anything related to weddings."


# %%
# optimal_steer = find_optimal_steer(adapter, target, sae.W_enc.shape[0], n_steps=int(1), target_scale=100)
# optimal_steer = adapter.W[:, ft_id]

# optimal_steer = find_optimal_steer(adapter, torch.zeros_like(target), sae.W_enc.shape[0], n_steps=int(1))
# optimal_steer = adapter.compute_optimal_input(torch.zeros_like(target))

if __name__ == "__main__":
    
    # optimal_steer = adapter.compute_optimal_input(target*1)
    optimal_steer = single_step_steer(adapter, target, bias_scale=1).cpu()
    # optimal_steer = find_optimal_steer(adapter,
    #                                    target,
    #                                    sae.W_enc.shape[0],
    #                                    n_steps=int(1e4),
    #                                    target_scale=10,
    #                                    init_steer=torch.randn(sae.W_enc.shape[0]),
    #                                 #    init_steer=torch.zeros(sae.W_enc.shape[0]),
    #                                    ).cpu()

    with torch.no_grad():
        optimal_steer = optimal_steer / torch.norm(optimal_steer)
    
    # cosine sims between optimal steer and all vector of sae.W_dec
    sims = (sae.W_dec @ optimal_steer) / (torch.norm(sae.W_dec, dim=-1) * torch.norm(optimal_steer))
    # top sims
    top_sims, top_indices = torch.topk(sims, 10)
    print("sae_16k top_sims:", top_sims)
    print("sae_16k top_indices:", top_indices)

    # def get_sae_dec():
    #     path_to_params = hf_hub_download(
    #         repo_id="google/gemma-scope-2b-pt-res",
    #         filename="layer_12/width_65k/average_l0_72/params.npz",
    #         force_download=False)
    #     params = np.load(path_to_params)
    #     pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
    #     sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    #     sae.load_state_dict(pt_params)
    #     return sae.W_dec

    # sae_65k = get_sae_dec()
    # sims = (sae_65k @ optimal_steer) / (torch.norm(sae_65k, dim=-1) * torch.norm(optimal_steer))
    # top_sims, top_indices = torch.topk(sims, 10)
    # print("sae_65k top_sims:", top_sims)
    # print("sae_65k top_indices:", top_indices)


# # %%
# if __name__ == "__main__":
#     old_steer = find_optimal_steer(adapter, target, sae.W_enc.shape[0], n_steps=1, target_scale=100)
#     old_steer = old_steer / torch.norm(old_steer)
#     old_steer = old_steer.cpu()
#     print((optimal_steer @ old_steer) / (torch.norm(optimal_steer) * torch.norm(old_steer)))

# %%
####### steer ######

if __name__ == "__main__":
    if BIG_MODEL:
        model = HookedTransformer.from_pretrained("google/gemma-2-9b", device=device, dtype=torch.float16)
        batch_size = 16
    else:
        model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
        batch_size = 64
    hp = "blocks.12.hook_resid_post"


# %%
if __name__ == "__main__":
    # compute_scores(optimal_steer, f"{ft_name}_optimised", criterion)
    _ = compute_scores(optimal_steer, model, f"{ft_name}_optimised", criterion, hp, scales=list(range(0, 240, 30)), batch_size=batch_size)
    # _ = compute_scores(optimal_steer, f"{ft_name}_optimised", criterion, scales=list(range(100, 300, 20)))

# %%
if __name__ == "__main__":
    _ = compute_scores(sae.W_dec[ft_id], model, f"{ft_name}_decoder", criterion, hp, scales=list(range(0, 240, 30)), batch_size=batch_size)



# %%
# step_options = [1e3, 2e3, 4e3, 6e3, 8e3, 1e4, 2e4, 5e4, 1e5]
step_options = [50, 80, 100, 150, 200]
def step_sweep():
    # sweep n_steps
    scales = list(range(0, 220, 20))
    scores = []
    coherences = []
    products = []
    for step in step_options:
        optimal_steer = find_optimal_steer(adapter, target, sae.W_enc.shape[0], n_steps=int(step))
        with torch.no_grad():
            optimal_steer = optimal_steer / torch.norm(optimal_steer)
        s, c, p = compute_scores(optimal_steer, f"{ft_name}_optimised", criterion, make_plot=False, scales=scales)
        scores.append(s)
        coherences.append(c)
        products.append(p)
        print(p)

    # plot products for every step option
    fig = go.Figure()
    for i, steps in enumerate(step_options):
        fig.add_trace(go.Scatter(
            x=scales,
            y=products[i],
            mode='lines',
            name=f'{int(steps)} steps'
        ))
    fig.update_layout(
        title='Score Product vs Scale for Different Step Options',
        xaxis_title='Scale',
        yaxis_title='Score Product',
        legend_title='Number of Steps',
        yaxis=dict(range=[0, 1])  # Set y-axis range from 0 to 1
    )
    fig.show()
    fig.write_image(f"analysis_out/{ft_name}_step_options_analysis.png")
    return scores, coherences, products

if __name__ == "__main__":
    scores, coherences, products = step_sweep()

# %%
bias_scales = [0, 0.2, 0.5, 0.8, 1, 1.2, 1.5, 1.8, 2, 2.5]
def bias_sweep():
    scales = list(range(0, 220, 20))
    scores = []
    coherences = []
    products = []
    for bias_scale in bias_scales:
        optimal_steer = single_step_steer(adapter, target, bias_scale=bias_scale)
        with torch.no_grad():
            optimal_steer = optimal_steer / torch.norm(optimal_steer)
        s, c, p = compute_scores(optimal_steer, f"{ft_name}_optimised", criterion, make_plot=False, scales=scales)
        scores.append(s)
        coherences.append(c)
        products.append(p)
        print(p)

    fig = go.Figure()
    for i, bias_scale in enumerate(bias_scales):
        fig.add_trace(go.Scatter(
            x=scales,
            y=products[i],
            mode='lines',
            name=f'{bias_scale} bias scale'
        ))
    fig.update_layout(
        title='Score Product vs Scale for Different Bias Scales',
        xaxis_title='Scale',
        yaxis_title='Score Product',
        legend_title='Bias Scale',
        yaxis=dict(range=[0, 1])  # Set y-axis range from 0 to 1
    )
    fig.show()
    fig.write_image(f"analysis_out/{ft_name}_bias_options_analysis.png")
    return scores, coherences, products

if __name__ == "__main__":
    scores, coherences, products = bias_sweep()


# %%

if __name__ == "__main__":

    max_products = [max(p) for p in products]
    # Create a figure for max product vs step
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=step_options,
        y=max_products,
        mode='lines+markers',
        name='Max Product'
    ))
    fig.update_layout(
        title='Max Score Product vs Number of Steps',
        xaxis_title='Number of Steps',
        yaxis_title='Max Score Product',
        xaxis=dict(type='log'),  # Use log scale for x-axis
        yaxis=dict(range=[0, 1])  # Set y-axis range from 0 to 1
    )
    fig.show()
    fig.write_image(f"analysis_out/{ft_name}_max_product_vs_steps.png")



# %%
if __name__ == "__main__":
    # bias values histogram
    b = adapter.b.detach().cpu()
    # fig = px.histogram(b, nbins=100)
    # fig.show()

    top_v, top_i = torch.topk(b, 10)
    print(top_v)
    print(top_i)

    bottom_v, bottom_i = torch.topk(-b, 10)
    print(bottom_v)
    print(bottom_i)

    



# %%

def find_rotation_matrix_svd(v1, v2):
    """
    Find the rotation matrix that rotates v1 to v2 using SVD.
    v1 and v2 are assumed to be 1000-dimensional unit vectors.
    """
    # Ensure input vectors are unit vectors
    v1 = v1 / torch.norm(v1)
    v2 = v2 / torch.norm(v2)

    # Compute the covariance matrix
    H = torch.outer(v2, v1)

    # Perform SVD
    U, _, Vt = torch.linalg.svd(H)

    # Compute the rotation matrix
    R = torch.matmul(U, Vt)

    # Ensure proper rotation (det(R) = 1)
    det = torch.det(R)
    if det < 0:
        Vt[-1, :] *= -1
        R = torch.matmul(U, Vt)

    return R



# %%
if __name__ == "__main__":
    b_to_subtract = adapter.W @ adapter.b
    b_to_subtract = b_to_subtract / torch.norm(b_to_subtract)

    R = find_rotation_matrix_svd(sae.W_dec[ft_id], optimal_steer)
# %%

if __name__ == "__main__":
    rotated_steer = R @ sae.W_dec[ft_id]
    rotated_steer = rotated_steer / torch.norm(rotated_steer)
    rotated_steer = rotated_steer.to(device)

    rotated_steer = rotated_steer - b_to_subtract
    rotated_steer = rotated_steer / torch.norm(rotated_steer)

    _ = compute_scores(rotated_steer, f"{ft_name}_rotated", criterion, scales=list(range(0, 240, 30)))
# %%
if __name__ == "__main__":
    opposite_rotated_steer = R.T @ R.T @ sae.W_dec[ft_id]
    opposite_rotated_steer = opposite_rotated_steer / torch.norm(opposite_rotated_steer)
    opposite_rotated_steer = opposite_rotated_steer.to(device)
    opposite_rotated_steer = opposite_rotated_steer - b_to_subtract
    opposite_rotated_steer = opposite_rotated_steer / torch.norm(opposite_rotated_steer)
    _ = compute_scores(opposite_rotated_steer, f"{ft_name}_opposite_rotated", criterion, scales=list(range(0, 240, 30)))

# %%

