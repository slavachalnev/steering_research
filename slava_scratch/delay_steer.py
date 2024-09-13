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

from steering.evals_utils import multi_criterion_evaluation

from ft_effects.train import LinearAdapter
from ft_effects.utils import get_sae

torch.set_grad_enabled(False)
# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sae = get_sae()
sae.to(device)

model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)

# %%

def patch_resid(resid, hook, steering, scale=1): # <--- only patch up to the last two tokens
    if resid.shape[1] <= 2:
        return resid
    resid[:, 1:-1, :] = resid[:, 1:-1, :] + steering * scale
    resid[:, -1, :] = resid[:, -1, :] + steering * scale * 0.5
    return resid


# def patch_resid(resid, hook, steering, scale=1):
#     resid[:, :, :] = resid[:, :, :] + steering * scale
#     return resid


def steer_model(model, steer, layer, text, scale=5, batch_size=32, n_samples=64):
    toks = model.to_tokens(text, prepend_bos=True)
    toks = toks.expand(batch_size, -1)
    hp = f"blocks.{layer}.hook_resid_post"
    all_gen = []
    for i in range(0, n_samples, batch_size):
        with model.hooks([(hp, partial(patch_resid, steering=steer, scale=scale))]):
            gen_toks = model.generate(
                toks,
                max_new_tokens=30,
                use_past_kv_cache=False, ### <--- no past kv cache
                top_k = 50,
                top_p = 0.3,
                verbose=False,
            )
            all_gen.extend(model.to_string(gen_toks))
    return all_gen

@torch.no_grad()
def compute_scores(steer, model, name, criterion, make_plot=True, scales=None):
    if scales is None:
        scales = list(range(0, 210, 10))
    scores = []
    coherences = []
    all_texts = dict()
    products = []
    for scale in tqdm(scales):
        gen_texts = steer_model(model, steer.to(model.W_E.device), 12, "I think", scale)
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
    return scores, coherences, products


# %%
ft_name = "london"
ft_id = 14455 # london
criterion = "Text mentions London or anything related to London."

# ft_name = "wedding"
# ft_id = 4230 # wedding
# criterion = "Text mentions weddings or anything related to weddings."

# ft_name = "recipe"
# ft_id = 1
# criterion = "The text is a recipe or a description of a scientific method. Specifically, it mentions serving in a dish or pouring into a beaker etc."

# %%

_ = compute_scores(sae.W_dec[ft_id], model, f"{ft_name}_delayed_decoder", criterion, scales=list(range(0, 220, 20)))





# %%
# %%





