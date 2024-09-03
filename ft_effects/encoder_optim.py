# optimise steering vector wrt encoder predicition on a bunch of data
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
from transformer_lens import utils as tutils
from huggingface_hub import hf_hub_download

from steering.sae import JumpReLUSAE
from steering.patch import patch_resid

from baselines.analysis import steer_model
from steering.evals_utils import multi_criterion_evaluation
# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
sae.to(device)
sae.requires_grad_(False)

# %%

class ActBuffer:
    def __init__(self, loader, n_samples=int(1e5), batch_size=128):
        self.acts = torch.zeros(0, sae.W_enc.shape[0]).to(device)
        self.batch_size = batch_size
        self.pos = 0
        self.loader = loader
        self.n_samples = n_samples
        self.refill()

    @torch.no_grad()
    def refill(self):
        print("Refilling buffer")
        # cut up to pos
        self.acts = self.acts[self.pos:]

        to_add = []
        n_added = self.acts.shape[0]
        for batch in self.loader:
            _, resid = model.run_with_cache(batch["tokens"], names_filter=hp, stop_at_layer=13)
            resid = resid[hp]
            resid = resid.reshape(-1, resid.shape[-1])
            to_add.append(resid)
            n_added += resid.shape[0]
            if n_added >= self.n_samples:
                break
        self.acts = torch.cat([self.acts, torch.cat(to_add, dim=0)], dim=0)

        ps = torch.randperm(self.acts.shape[0])
        self.acts = self.acts[ps]
        self.pos = 0

    def get_batch(self):
        if self.pos + self.batch_size >= self.acts.shape[0]:
            self.refill()
        batch = self.acts[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        return batch


# %%

def optimise_wrt_enc(ft_id, target, sae, buffer, n_iters):
    scale = 50
    target = target.to(device)
    steer = sae.W_dec[ft_id].clone()
    steer = steer / torch.norm(steer)
    steer = steer * scale
    steer.requires_grad_(True)
    optimizer = torch.optim.Adam([steer], lr=1e-3)

    for batch_idx in range(n_iters):
        resid = buffer.get_batch()

        # sample 124 random positions
        pos = torch.randint(0, resid.shape[0], (124,))
        resid = resid[pos]

        enc_resid = sae.encode(resid)
        steer_resid = resid + steer
        enc_steer_resid = sae.encode(steer_resid)
        pred_diffs = ((enc_steer_resid - enc_resid)**2).mean(dim=0)

        loss = F.mse_loss(pred_diffs, target)
        if batch_idx % 100 == 0:
            print(batch_idx, loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # renormalise
        steer.data = steer.data / torch.norm(steer.data)
        steer.data = steer.data * scale

    return steer.detach()

# %%
target = torch.zeros(sae.W_enc.shape[1])
target_value = 50

# ft_name = "london"
# ft_id = 14455 # london
# target[ft_id] = target_value # london
# criterion = "Text mentions London or anything related to London."

ft_name = "wedding"
ft_id = 4230 # wedding
target[ft_id] = target_value # wedding
criterion = "Text mentions weddings or anything related to weddings."


# %%

if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
    hp = "blocks.12.hook_resid_post"

    data = tutils.load_dataset("NeelNanda/c4-code-20k", split="train")
    tokenized_data = tutils.tokenize_and_concatenate(data, model.tokenizer, max_length=32)
    tokenized_data = tokenized_data.shuffle(42)
    loader = DataLoader(tokenized_data, batch_size=32)

# %%
if __name__ == "__main__":
    buffer = ActBuffer(loader)
    optim_steer = optimise_wrt_enc(ft_id, target, sae, buffer, n_iters=int(1e4))
    optim_steer = optim_steer / torch.norm(optim_steer)

# %%
sim = (optim_steer @ sae.W_dec[ft_id]) / (torch.norm(optim_steer) * torch.norm(sae.W_dec[ft_id]))
print(sim)
print(torch.norm(optim_steer))

# %%
@torch.no_grad()
def compute_scores(steer, name, criterion, make_plot=True, scales=None):
    if scales is None:
        scales = list(range(0, 210, 10))
    scores = []
    coherences = []
    all_texts = dict()
    products = []
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
    fig.update_layout(
        title=f"Steering Analysis for {name}",
        xaxis_title='Scale',
        yaxis_title='Value',
        yaxis=dict(range=[0, 1])  # Set y-axis range from 0 to 1
    )
    fig.add_trace(go.Scatter(x=scales, y=coherences, mode='lines', name='Coherence'))
    fig.add_trace(go.Scatter(x=scales, y=scores, mode='lines', name='Score'))
    fig.add_trace(go.Scatter(x=scales, y=products, mode='lines', name='Coherence * Score'))
    if make_plot:
        fig.show()
        fig.write_image(f"analysis_out/{name}_steer_analysis.png")
    # save all_texts as json
    if make_plot:
        with open(f"analysis_out/{name}_all_texts.json", "w") as f:
            json.dump(all_texts, f)
    return scores, coherences, products

# %%
if __name__ == "__main__":
    _ = compute_scores(optim_steer, f"{ft_name}_optimised", criterion, scales=list(range(0, 220, 20)))

# %%
if __name__ == "__main__":
    _ = compute_scores(sae.W_dec[ft_id], f"{ft_name}_decoder", criterion, scales=list(range(0, 220, 20)))
# %%
