# %%
import os
import sys
sys.path.append(os.path.abspath('..'))
from steering.patch import patch_resid
import torch
from transformer_lens import HookedTransformer
from functools import partial
import json
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from steering.evals_utils import multi_criterion_evaluation
from steering.utils import normalise_decoder

from datasets import load_dataset

from baselines.activation_steering import get_activation_steering, load_act_steer
from baselines.caa import compute_diffs_for_dataset, all_layer_effects
from sae_lens import SAE

from steering.sae import JumpReLUSAE

from ft_effects.utils import LinearAdapter, steer_model

import numpy as np
from huggingface_hub import hf_hub_download
from steering.utils import normalise_decoder

import transformer_lens.utils as tutils

from heapq import heappush, heappushpop

from IPython.display import HTML, display
import html

# %%
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)

# %%

SEQ_LEN = 128

data = load_dataset("NeelNanda/c4-code-20k", split="train")

tokenized_data = tutils.tokenize_and_concatenate(data, model.tokenizer, max_length=SEQ_LEN) # type: ignore
tokenized_data = tokenized_data.shuffle(42)

all_tokens = tokenized_data["tokens"]
assert isinstance(all_tokens, torch.Tensor)

print(all_tokens.shape)

# %%

# load 65k sae
path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-2b-pt-res",
    filename="layer_12/width_65k/average_l0_72/params.npz",
    force_download=False
)
params = np.load(path_to_params)
pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
sae.load_state_dict(pt_params)
sae.to(device)

hp = "blocks.12.hook_resid_post"

# %%
text = "San Francisco's Golden Gate Bridge"

toks = model.to_tokens(text)
toks = toks.to(device)

_, acts = model.run_with_cache(toks, names_filter=hp)
acts = acts[hp]

print("acts shape", acts.shape)

all_sae_acts = []
for batch in acts:
    sae_acts = sae.encode(batch)
    all_sae_acts.append(sae_acts)

all_sae_acts = torch.stack(all_sae_acts, dim=0)

top_v, top_i = torch.topk(all_sae_acts, 10, dim=-1)

print(top_v)
print(top_i)

# %%


def get_top_samples(tokens, top_k: int = 10, batch_size: int = 4, sae_layer: int = 12):
    n_samples, seq_len = tokens.shape
    n_features = None  # We'll set this when we get the first batch of results
    
    # Initialize lists to store the top k values and indices for each feature
    top_k_values = []
    top_k_indices = []
    
    for i in tqdm(range(0, n_samples, batch_size)):
        batch = tokens[i:i+batch_size]
        _, acts = model.run_with_cache(batch, names_filter=hp, stop_at_layer=sae_layer+1)
        acts = acts[hp]
        max_sae_acts = []
        for sample in acts:
            sae_acts = sae.encode(sample)
            max_sae_acts.append(sae_acts.max(dim=0).values)
        
        max_sae_acts = torch.stack(max_sae_acts, dim=0)
        
        if n_features is None:
            n_features = max_sae_acts.shape[1]
            top_k_values = [[] for _ in range(n_features)]
            top_k_indices = [[] for _ in range(n_features)]
        
        # Update top k for each feature
        for feature_idx in range(n_features):
            feature_values = max_sae_acts[:, feature_idx]
            for sample_idx, value in enumerate(feature_values):
                global_idx = i + sample_idx
                if len(top_k_values[feature_idx]) < top_k:
                    heappush(top_k_values[feature_idx], (value.item(), global_idx))
                    top_k_indices[feature_idx] = [idx for _, idx in top_k_values[feature_idx]]
                elif value > top_k_values[feature_idx][0][0]:
                    heappushpop(top_k_values[feature_idx], (value.item(), global_idx))
                    top_k_indices[feature_idx] = [idx for _, idx in top_k_values[feature_idx]]
    
    # Sort the results for each feature
    for feature_idx in range(n_features):
        sorted_results = sorted(zip(top_k_values[feature_idx], top_k_indices[feature_idx]), reverse=True)
        top_k_values[feature_idx] = [value for value, _ in sorted_results]
        top_k_indices[feature_idx] = [idx for _, idx in sorted_results]
    
    return top_k_values, top_k_indices


# top_k_values, top_k_indices = get_top_samples(all_tokens[:10000], top_k=10, batch_size=16, sae_layer=12)
# top_k_values, top_k_indices = get_top_samples(all_tokens[:200], top_k=10, batch_size=16, sae_layer=12)


# %%

# # save top_k_values and top_k_indices
# with open("top_k_values_65k_12.json", "w") as f:
#     json.dump(top_k_values, f)
# with open("top_k_indices_65k_12.json", "w") as f:
#     json.dump(top_k_indices, f)
# torch.save(all_tokens, "all_tokens.pt")

# load top_k_values and top_k_indices
with open("top_k_values_65k_12.json", "r") as f:
    top_k_values = json.load(f)
with open("top_k_indices_65k_12.json", "r") as f:
    top_k_indices = json.load(f)



# %%

def acts_for_feature(tokens, feature_idx: int, batch_size: int = 4, sae_layer: int = 12):
    n_samples, seq_len = tokens.shape
    all_acts = []
    for i in tqdm(range(0, n_samples, batch_size)):
        batch = tokens[i:i+batch_size]
        _, acts = model.run_with_cache(batch, names_filter=hp, stop_at_layer=sae_layer+1)
        acts = acts[hp]
        batch_sae_acts = []
        for sample in acts:
            sae_acts = sae.encode(sample)
            batch_sae_acts.append(sae_acts[:, feature_idx])
        
        batch_sae_acts = torch.stack(batch_sae_acts, dim=0)
        all_acts.append(batch_sae_acts)
    
    return torch.cat(all_acts, dim=0)

acts_for_selected = acts_for_feature(all_tokens[:10], 2)
acts_for_selected

# %%
def display_highlighted_tokens(tokens, values, tokenizer):
    # Ensure tensors are on CPU
    tokens = tokens.cpu()
    values = values.cpu().to(torch.float32)
    
    original_values = values.clone()
    
    # Normalize values to range [0, 1] for coloring
    values = (values - values.min()) / (values.max() - values.min())
    
    # Function to convert value to RGB color
    def value_to_color(value):
        # Using a gradient from white to red
        color_value = int(255 * (1 - value.item()))
        return f"rgb(255, {color_value}, {color_value})"
    
    html_output = ""
    
    for batch_idx in range(tokens.shape[0]):
        # Use the original values for max activation
        max_activation = original_values[batch_idx].max().item()
        html_output += f"<p>Max Activation: {max_activation:.4f}:</p>"
        html_output += "<p>"
        
        for token, value in zip(tokens[batch_idx], values[batch_idx]):
            word = tokenizer.decode([token.item()])
            # Escape special characters to prevent HTML conflicts
            word_escaped = html.escape(word)
            color = value_to_color(value)
            html_output += f'<span style="background-color: {color};">{word_escaped}</span>'
        
        html_output += "</p>"
    
    display(HTML(html_output))


# %%


# ft_id = 23915 # <-- London
# ft_id = 13843 # <-- SF

ft_id = 52734 # <-- Bridge


samples = all_tokens[top_k_indices[ft_id]]

acts = acts_for_feature(samples, ft_id)
acts

display_highlighted_tokens(samples[:, 3:], acts[:, 3:], model.tokenizer)



# %%



