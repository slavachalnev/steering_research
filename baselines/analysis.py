# analysis.py

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

import numpy as np
from huggingface_hub import hf_hub_download
from steering.utils import normalise_decoder

# %%

def load_sae_steer(path):

    with open(os.path.join(path, "feature_steer.json"), 'r') as f:
        steer = json.load(f)

    sae_load_method = steer.get('sae_load_method', 'saelens')

    if sae_load_method == 'saelens':
        sae, _, _ = SAE.from_pretrained(
            release=steer['sae'],
            sae_id=steer['layer'],
            device='cpu'
        )
        normalise_decoder(sae)
    elif sae_load_method == 'gemmascope':
        path_to_params = hf_hub_download(
            repo_id=steer['repo_id'],
            filename=steer['filename'],
            force_download=False
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
        sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
        sae.load_state_dict(pt_params)
        sae.cpu()
    else:
        raise ValueError(f"Unknown sae_load_method: {sae_load_method}")

    vectors = []
    for ft_id, ft_scale in steer['features']:
        vectors.append(sae.W_dec[ft_id] * ft_scale)
    vectors = torch.stack(vectors, dim=0)
    vec = vectors.sum(dim=0)
    vec = vec / torch.norm(vec, dim=-1, keepdim=True)

    return vec, steer['hp'], steer['layer']

def steer_model(model, steer, hp, text, scale=5, batch_size=64, n_samples=128):
    toks = model.to_tokens(text, prepend_bos=True)
    toks = toks.expand(batch_size, -1)
    all_gen = []
    for _ in range(0, n_samples, batch_size):
        with model.hooks([(hp, partial(patch_resid, steering=steer, scale=scale))]):
            gen_toks = model.generate(
                toks,
                max_new_tokens=30,
                use_past_kv_cache=True,
                top_k=50,
                top_p=0.3,
                verbose=False,
            )
            all_gen.extend(model.to_string(gen_toks))
    return all_gen

def plot(path, coherence, score, product, scales, method, steering_goal_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scales, y=coherence, mode='lines', name='Coherence'))
    fig.add_trace(go.Scatter(x=scales, y=score, mode='lines', name='Score'))
    fig.add_trace(go.Scatter(x=scales, y=product, mode='lines', name='Coherence * Score'))
    fig.update_layout(
        title=f'Steering Analysis for {steering_goal_name} ({method})',
        xaxis_title='Scale',
        yaxis_title='Value',
        legend_title='Metric',
        yaxis=dict(range=[0, 1])
    )
    # Create a filename-friendly version of the steering goal name
    safe_name = steering_goal_name.replace(" ", "_")
    fig.write_image(os.path.join(path, f"scores_{safe_name}_{method}.png"), scale=2)

def analyse_steer(model, steer, hp, path, method='activation_steering'):
    scales = list(range(0, 320, 20))
    with open(os.path.join(path, "criteria.json"), 'r') as f:
        criteria = json.load(f)
    with open(os.path.join(path, "prompts.json"), 'r') as f:
        prompts = json.load(f)

    # Read the steering goal name from criteria.json
    steering_goal_name = criteria[0].get('name', 'Unknown')

    all_texts = []

    avg_score = []
    avg_coh = []
    for scale in tqdm(scales):
        texts = steer_model(model, steer, hp, prompts[0], scale=scale)
        all_texts.append((scale, texts))

        score, coherence = multi_criterion_evaluation(
            texts,
            [criteria[0]['score'], criteria[0]['coherence']],
            prompt=prompts[0],
        )

        score = [item['score'] for item in score]
        score = [(item - 1) / 9 for item in score]
        coherence = [item['score'] for item in coherence]
        coherence = [(item - 1) / 9 for item in coherence]
        avg_score.append(sum(score) / len(score))
        avg_coh.append(sum(coherence) / len(coherence))

    # Compute the product at each scale
    product = [c * s for c, s in zip(avg_coh, avg_score)]

    # Find the maximum product and the corresponding scale
    max_product = max(product)
    max_index = product.index(max_product)
    max_scale = scales[max_index]

    # Log or store these results
    result = {
        'path': path,
        'method': method,
        'steering_goal_name': steering_goal_name,
        'max_product': max_product,
        'scale_at_max': max_scale
    }

    with open(os.path.join(path, f"generated_texts_{method}.json"), 'w') as f:
        json.dump(all_texts, f, indent=2)

    plot(path, avg_coh, avg_score, product, scales, method, steering_goal_name)

    return result

# %%
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)

# %%
if __name__ == "__main__":

    paths = [
        "steer_cfgs/gemma2/london",
        "steer_cfgs/gemma2/wedding",
    ]

    results = []

    for path in paths:
        # Activation Steering
        print("Activation Steering")
        pos_examples, neg_examples, val_examples, layer = load_act_steer(path)
        steer = get_activation_steering(model, pos_examples, neg_examples, device=device, layer=layer)
        steer = steer / torch.norm(steer, dim=-1, keepdim=True)
        hp = f"blocks.{layer}.hook_resid_post"
        result = analyse_steer(model, steer, hp, path, method='ActSteer')
        results.append(result)

        # SAE Steering
        print("SAE Steering")
        steer, hp, layer = load_sae_steer(path)
        steer = steer.to(device)
        result = analyse_steer(model, steer, hp, path, method='SAE')
        results.append(result)

    # Write the results to a JSON file
    with open('steering_results.json', 'w') as f:
        json.dump(results, f, indent=2)

# %%
