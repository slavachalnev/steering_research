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
# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

paths = [
    "effects/G2_2B_L12/65k_from_0",
    "effects/G2_2B_L12/65k_from_10k",
    "effects/G2_2B_L12/65k_from_20k",
    "effects/G2_2B_L12/65k_from_30k",
    "effects/G2_2B_L12/65k_from_40k",
    # "effects/G2_2B_L12/multi_16k_from_0",
]

features = []
effects = []

for path in paths:
    features.append(torch.load(os.path.join(path, "used_features.pt")))
    effects.append(torch.load(os.path.join(path, "all_effects.pt")))

features = torch.cat(features)
effects = torch.cat(effects)
print(features.shape)
print(effects.shape)

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
        self.d_model, self.d_sae = sae.W_enc.shape 
        # self.W = nn.Parameter(sae.W_enc.detach().clone())
        self.W = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.d_model, self.d_sae)))
        self.b = nn.Parameter(torch.zeros(self.d_sae))

    def forward(self, x):
        return x @ self.W + self.b

class NonLinearAdapter(nn.Module):
    def __init__(self, sae):
        super().__init__()
        self.d_model, self.d_sae = sae.W_enc.shape 
        self.W1 = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.d_model, self.d_sae)))
        self.b1 = nn.Parameter(torch.zeros(self.d_sae))
        self.W2 = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.d_sae, self.d_sae)))
        self.b2 = nn.Parameter(torch.zeros(self.d_sae))

    def forward(self, x):
        x = x @ self.W1 + self.b1
        x = F.relu(x)
        x = x @ self.W2 + self.b2
        return x


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
# adapter = NonLinearAdapter(sae)
adapter.to(device)
train(20, lr=2e-4)

torch.save(adapter.state_dict(), "adapter.pt")
# %%


def find_optimal_steer(target, d_model):
    steer = torch.zeros(d_model, requires_grad=True, device=device)
    steer = steer.to(device)
    target = target.to(device)
    optim = torch.optim.Adam([steer], lr=1e-4)
    for step in range(10000):
        optim.zero_grad()
        pred = adapter(steer)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optim.step()
        if step % 100 == 0:
            print(loss.item())
    return steer.detach()


# %%

target = torch.zeros(sae.W_enc.shape[1])

# ft_name = "london"
# ft_id = 14455 # london
# target[ft_id] = 5 # london
# target[3931] = 1 # uk
# criterion = "Text mentions London or anything related to London."

ft_name = "wedding"
ft_id = 4230 # wedding
target[ft_id] = 5 # wedding
criterion = "Text mentions weddings or anything related to weddings."


# %%

optimal_steer = find_optimal_steer(target, sae.W_enc.shape[0])
with torch.no_grad():
    optimal_steer = optimal_steer / torch.norm(optimal_steer)
    
optimal_steer = optimal_steer.cpu()

# cosine sims between optimal steer and all vector of sae.W_dec
sims = (sae.W_dec @ optimal_steer) / (torch.norm(sae.W_dec, dim=-1) * torch.norm(optimal_steer))
# top sims
top_sims, top_indices = torch.topk(sims, 10)
print("sae_16k top_sims:", top_sims)
print("sae_16k top_indices:", top_indices)

def get_sae_dec():
    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-res",
        filename="layer_12/width_65k/average_l0_72/params.npz",
        force_download=False) 
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)
    return sae.W_dec

sae_65k = get_sae_dec()
sims = (sae_65k @ optimal_steer) / (torch.norm(sae_65k, dim=-1) * torch.norm(optimal_steer))
top_sims, top_indices = torch.topk(sims, 10)
print("sae_65k top_sims:", top_sims)
print("sae_65k top_indices:", top_indices)


# %%
# %%
####### steer ######

model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
hp = "blocks.12.hook_resid_post"

# def gen(prompt, steer, scale):
#     toks = model.to_tokens(prompt, prepend_bos=True)
#     toks = toks.expand(10, -1)
#     with model.hooks([(hp, partial(patch_resid, steering=steer, scale=scale))]):
#         gen_toks = model.generate(toks, max_new_tokens=30)
#     return model.to_string(gen_toks)


# gen("I think", optimal_steer, 120)

# %%
def compute_scores(steer, name):
    scores = []
    coherences = []
    all_texts = dict()
    products = []
    scales = list(range(0, 210, 10))
    for scale in tqdm(scales):
        gen_texts = steer_model(model, steer.to(device), 12, "I think", scale)
        all_texts[scale] = gen_texts

        score, coherence = multi_criterion_evaluation(
            gen_texts,
            [
                criterion,
                "Text is coherent and the grammar is correct."
            ],
            prompt="I think",
        )

        score = [item['score'] for item in score]
        score = [(item - 1) / 9 for item in score]
        avg_score = sum(score)/len(score)
        coherence = [item['score'] for item in coherence]
        coherence = [(item - 1) / 9 for item in coherence]
        avg_coherence = sum(coherence)/len(coherence)
        scores.append(avg_score)
        coherences.append(avg_coherence)
        products.append(avg_score * avg_coherence)

    fig = go.Figure()
    fig.update_layout(title=f"Steering Analysis for {name}")
    fig.add_trace(go.Scatter(x=scales, y=coherences, mode='lines', name='Coherence'))
    fig.add_trace(go.Scatter(x=scales, y=scores, mode='lines', name='Score'))
    fig.add_trace(go.Scatter(x=scales, y=products, mode='lines', name='Coherence * Score'))
    fig.update_layout(xaxis_title='Scale', yaxis_title='Value')
    fig.show()
    fig.write_image(f"analysis_out/{name}_steer_analysis.png")

    # save all_texts as json
    with open(f"analysis_out/{name}_all_texts.json", "w") as f:
        json.dump(all_texts, f)

compute_scores(optimal_steer, f"{ft_name}_target")

# %%

# compute_scores(sae.W_dec[14455], "london")
compute_scores(sae.W_dec[ft_id], f"{ft_name}_decoder")



# %%
model.cpu()
del model
# %%

