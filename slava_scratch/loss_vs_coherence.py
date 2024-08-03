# %%
import os
import sys
sys.path.append(os.path.abspath('..'))

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

scales = list(range(0, 110, 10))
print(scales)

def coherence(steer, scales):
    avg_coherences = []
    for scale in scales:
        gen_texts = generate(
            model,
            hooks=[(hp, partial(patch_resid, steering=steer, scale=scale))],
            prompt="",
            n_samples=128,
            batch_size=128,
            max_new_tokens=30,
        )
        scores = evaluate_completions(gen_texts,
                                      criterion="The text is coherent and makes sense. The grammar is correct.",
                                      prompt="",
                                      )
        scores = [score['score'] for score in scores if "error" not in score]
        avg_coherences.append(sum(scores) / len(scores))
    return avg_coherences

# %%

batch_size = 128
data = tutils.load_dataset("NeelNanda/c4-code-20k", split="train")
tokenized_data = tutils.tokenize_and_concatenate(data, model.tokenizer, max_length=32)
tokenized_data = tokenized_data.shuffle(42)
loader = DataLoader(tokenized_data, batch_size=batch_size)


def compute_loss(steer, scales, n_batches):
    losses = []
    for scale in scales:
        total_loss = 0
        for batch_idx, batch in enumerate(loader):
            with model.hooks(fwd_hooks=[(hp, partial(patch_resid,
                                                        steering=steer,
                                                        scale=scale,
                                                        ))]):
                loss = model(batch['tokens'], return_type='loss')
                total_loss += loss
            if batch_idx >= n_batches:
                break
        losses.append(total_loss.item() / n_batches)
    return losses

# %%

def compute_all(n=100):
    all_coherences: list[list[float]] = []
    all_losses: list[list[float]] = []

    for i in tqdm(range(n)):
        steer = sae.W_dec[i]

        cohs: list[float] = coherence(steer, scales)
        losses = compute_loss(steer, scales, 1)

        all_coherences.append(cohs)
        all_losses.append(losses)
    
    return all_coherences, all_losses


# %%

all_c, all_l = compute_all(50)


# %%


flat_c = [item for sublist in all_c for item in sublist]
flat_l = [item for sublist in all_l for item in sublist]
# %%
fig = px.scatter(x=flat_l, y=flat_c)
fig.show()

# %%



fig = px.scatter(x=[s[5] for s in all_l], y=[s[5] for s in all_c])
fig.show()


# %%

# Prepare the data for plotting
data = []
for i in range(5):  # For each of the 100 lines
    for j in range(10):  # For each of the 10 data points
        data.append({
            'Line': f'Line {i+1}',
            'Coherence': all_c[i][j],
            'Loss': all_l[i][j]
        })

# Convert to DataFrame
df = pd.DataFrame(data)

# Create the line plot
fig = px.line(df, x='Loss', y='Coherence', color='Line', 
              title='Coherence vs Loss for 100 Lines',
              labels={'Loss': 'Loss', 'Coherence': 'Coherence'},
              hover_data=['Line'])

# Update layout for better readability
fig.update_layout(
    xaxis_title='Loss',
    yaxis_title='Coherence',
    legend_title='Lines',
    showlegend=False  # Hide legend due to large number of lines
)

fig.show()

# %%
print('scales:', scales)
# %%

# Prepare the data for plotting
data = []
for i in range(50):  # For each of the 100 lines
    for j in range(10):  # For each of the 10 data points
        data.append({
            'Line': f'Line {i+1}',
            'Scale': scales[j],
            'Loss': all_l[i][j]
        })

# Convert to DataFrame
df = pd.DataFrame(data)

# Create the line plot
fig = px.line(df, x='Scale', y='Loss', color='Line', 
              title='Loss vs Scale for 100 Lines',
              labels={'Scale': 'Scale', 'Loss': 'Loss'},
              hover_data=['Line'])

# Update layout for better readability
fig.update_layout(
    xaxis_title='Scale',
    yaxis_title='Loss',
    legend_title='Lines',
    showlegend=False  # Hide legend due to large number of lines
)

fig.show()


# %%