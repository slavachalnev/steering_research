import os
import sys
sys.path.append(os.path.abspath('..'))
import json
from steering.sae import JumpReLUSAE

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens import utils as tutils
from transformer_lens.evals import make_pile_data_loader, evaluate_on_dataset


from functools import partial
from datasets import load_dataset
from tqdm import tqdm

from steering.patch import generate, scores_2d, patch_resid

import numpy as np
from huggingface_hub import hf_hub_download

torch.set_grad_enabled(False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)


################# load sae ##################
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
#############################################


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


def gen(steer=None, scale=60, batch_size=64, max_toks=32, n_batches=1, verbose=False):
    if steer is not None:
        hooks = [(hp, partial(patch_resid, steering=steer, scale=scale))]
    else:
        hooks = []
    generated_tokens = []
    for prompt in tqdm(prompts, disable=not verbose):
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
    all_sae_acts = torch.zeros(sae.W_dec.shape[0], device=device)
    count = 0
    for i in range(0, tokens.shape[0], batch_size):
        batch = tokens[i:i+batch_size]
        _, acts = model.run_with_cache(batch, names_filter=hp, stop_at_layer=13)
        acts = acts[hp] # shape (batch_size, len, d_model)
        acts = acts.reshape(-1, acts.shape[-1]) # shape (batch_size * len, d_model)
        sae_acts = sae.encode(acts)
        all_sae_acts += sae_acts.sum(dim=0)
        count += sae_acts.shape[0]
    return all_sae_acts / count


print("Getting baseline feature acts")
baseline_dist = get_feature_acts(gen(n_batches=10, verbose=True), 64) ### n_batches=10
print("done baseline")

top_vs = []
top_is = []

from_idx = 0
to_idx = 8000

early_stop = False

try:
    for i in tqdm(range(from_idx, to_idx)):
        feature = sae.W_dec[i]
        ft_dist = get_feature_acts(gen(feature), 64)
        diff = ft_dist - baseline_dist
        top_v, top_i = torch.topk(diff, 20, dim=-1)
        top_vs.append(top_v.cpu())
        top_is.append(top_i.cpu())
except KeyboardInterrupt:
    # save the results so far
    print("Saving results so far")
    all_top_vs = torch.stack(top_vs)
    all_top_is = torch.stack(top_is)
    up_to_idx = from_idx + all_top_vs.shape[0]
    torch.save(all_top_vs, f"top_vs_{from_idx}_{up_to_idx}.pt")
    torch.save(all_top_is, f"top_is_{from_idx}_{up_to_idx}.pt")
    early_stop = True

if not early_stop:
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

