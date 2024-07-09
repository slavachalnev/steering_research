import os
import sys
sys.path.append(os.path.abspath('..'))
import json

import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens import utils as tutils
from transformer_lens.evals import make_pile_data_loader, evaluate_on_dataset

from functools import partial
from datasets import load_dataset
from tqdm import tqdm

import einops

from sae_lens import SAE
# from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
# from sae_lens import SparseAutoencoder, ActivationsStore

# from steering.eval_utils import evaluate_completions
from steering.utils import normalise_decoder
from steering.patch import generate, scores_2d, patch_resid

# from sae_vis.data_config_classes import SaeVisConfig
# from sae_vis.data_storing_fns import SaeVisData

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

torch.set_grad_enabled(False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("gemma-2b", device=device)

hp6 = "blocks.6.hook_resid_post"
sae6, _, _ = SAE.from_pretrained(
    release = "gemma-2b-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = hp6, # won't always be a hook point
    device = 'cpu'
)
sae6 = sae6.to(device)
normalise_decoder(sae6)

prompts = [
    "",
    "I think",
    "Breaking news",
    "Last night",
    "For sale",
    "The weather",
    "Dear Sir/Madam",
    "Preheat the oven",
    "It's interesting that"
    "Assistant:",
    "I went up to",
    "New study suggests",
    ]


def gen(ft_id=None, scale=60, batch_size=64, max_toks=32, n_batches=1):
    if ft_id is not None:
        steer = sae6.W_dec[ft_id]
        hooks = [(hp6, partial(patch_resid, hook=sae6, steering=steer, scale=scale))]
    else:
        hooks = []
    generated_tokens = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        prompt_batch = tokens.expand(batch_size, -1)
        for _ in range(n_batches):
            with model.hooks(hooks):
                gen_batch = model.generate(
                    prompt_batch,
                    max_new_tokens=max_toks - tokens.shape[-1],
                    top_k=50,
                    top_p=0.3,
                    verbose=False,
                )
            generated_tokens.append(gen_batch)
    return torch.cat(generated_tokens, dim=0)
            

def get_feature_acts(tokens, batch_size):
    assert tokens.shape[1] == 32

    all_sae_acts = torch.zeros(sae6.cfg.d_sae, device=device)
    count = 0

    for i in range(0, tokens.shape[0], batch_size):
        batch = tokens[i:i+batch_size]
        _, acts = model.run_with_cache(batch, names_filter=hp6, stop_at_layer=7)
        acts = acts[hp6] # shape (batch_size, len, d_model)
        acts = acts.reshape(-1, acts.shape[-1]) # shape (batch_size * len, d_model)
        sae_acts = sae6.encode(acts)
        all_sae_acts += sae_acts.sum(dim=0)
        count += sae_acts.shape[0]
    return all_sae_acts / count


with open("../interface/live_neurons.json", "r") as f:
    live_neurons = torch.tensor(json.load(f))

baseline_dist = get_feature_acts(gen(n_batches=10), 64)

top_vs = []
top_is = []

from_idx = 0
to_idx = 8000

for i in tqdm(range(from_idx, to_idx)):
    if i not in live_neurons:
        top_vs.append(torch.zeros(20, dtype=torch.float32))
        top_is.append(torch.zeros(20, dtype=torch.int64))
        continue
    ft_dist = get_feature_acts(gen(i), 64)
    diff = ft_dist - baseline_dist
    top_v, top_i = torch.topk(diff, 20, dim=-1)
    top_vs.append(top_v.cpu())
    top_is.append(top_i.cpu())

all_top_vs = torch.stack(top_vs)
all_top_is = torch.stack(top_is)
torch.save(all_top_vs, f"top_vs_{from_idx}_{to_idx}.pt")
torch.save(all_top_is, f"top_is_{from_idx}_{to_idx}.pt")

# acts_baseline = get_feature_acts(gen(), 64)
# acts_london = get_feature_acts(gen(10138), 64)

# london_diff = acts_london - acts_baseline
# top_v, top_i = torch.topk(london_diff, 10, dim=-1)
# print(top_i)
# print(top_v)

