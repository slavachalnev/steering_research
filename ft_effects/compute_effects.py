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


def get_sae_dec():
    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-res",
        filename="layer_12/width_65k/average_l0_72/params.npz",
        force_download=False) 
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)
    return sae.W_dec.to(device)


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


def get_scale(steer, loader, scales, n_batches=2, target_loss=6):
    assert torch.allclose(torch.norm(steer), torch.tensor(1.0))
    losses = []
    for scale in scales:
        total_loss = 0
        for batch_idx, batch in enumerate(loader):
            with model.hooks(fwd_hooks=[(hp, partial(patch_resid, steering=steer, scale=scale))]):
                loss = model(batch['tokens'], return_type='loss')
                total_loss += loss
            if batch_idx >= n_batches:
                break
        losses.append(total_loss.item() / n_batches)
        if total_loss.item()/n_batches > target_loss:
            break
    scales = scales[:len(losses)]
    # linear interpolation
    x1, x2 = scales[-2], scales[-1]
    y1, y2 = losses[-2], losses[-1]
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    target_scale = (target_loss - b) / m
    return target_scale


def all_effects(features, save_to: str, scale=None):
    """
    Args:
        features: steering vectors of shape (n_features, d_model)
        save_to: directory to save results
        scale: if not None, use this scale for all features. If None,
            then compute scale for each feature automatically.
    """
    baseline_dist = get_feature_acts(gen(n_batches=10, verbose=True), 64)

    # prep data
    data = tutils.load_dataset("NeelNanda/c4-code-20k", split="train")
    tokenized_data = tutils.tokenize_and_concatenate(data, model.tokenizer, max_length=32)
    tokenized_data = tokenized_data.shuffle(42)
    loader = DataLoader(tokenized_data, batch_size=64)

    all_effects = []
    used_features = []
    try:
        for feature in tqdm(features):
            if scale is None:
                scale = get_scale(feature, loader, scales=list(range(0, 220, 20)), n_batches=2)
            ft_dist = get_feature_acts(gen(feature, scale=scale), 64)
            diff = ft_dist - baseline_dist
            all_effects.append(diff.to("cpu"))
            used_features.append((feature * scale).to("cpu"))
    finally:
        print("Saving results")
        all_effects = torch.stack(all_effects)
        used_features = torch.stack(used_features)
        torch.save(all_effects, os.path.join(save_to, "all_effects.pt"))
        torch.save(used_features, os.path.join(save_to, "used_features.pt"))
        with open(os.path.join(save_to, "about.txt"), "w") as f:
            f.write(f"all_effects shape: {all_effects.shape}\n")
            f.write(f"used_features shape: {used_features.shape}\n")


# save_dir = "effects/G2_2B_L12/16k_from_0"
# os.makedirs(save_dir)
# all_effects(sae.W_dec[:3], save_dir)


# save_dir = "effects/G2_2B_L12/65k_from_0"
# os.makedirs(save_dir)
# all_effects(get_sae_dec()[:3], save_dir)



### interpolation ###
wedding = sae.W_dec[4230]
serve_dish = sae.W_dec[1]
interp = []
for alpha in np.linspace(0, 1, 21):
    interp.append(alpha * wedding + (1-alpha) * serve_dish)
interp = torch.stack(interp)
interp = interp / torch.norm(interp, dim=-1, keepdim=True)
save_dir = "effects/G2_2B_L12/interp_wedding_dish"
os.makedirs(save_dir)
all_effects(interp, save_dir)

