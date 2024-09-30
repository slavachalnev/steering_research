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

# %%


def load_sae_model(config):
    sae_load_method = config.get('sae_load_method', 'saelens')

    if sae_load_method == 'saelens':
        sae, _, _ = SAE.from_pretrained(
            release=config['sae'],
            sae_id=config['layer'],
            device='cpu'
        )
        normalise_decoder(sae)
    elif sae_load_method == 'gemmascope':
        path_to_params = hf_hub_download(
            repo_id=config['repo_id'],
            filename=config['filename'],
            force_download=False
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
        sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
        sae.load_state_dict(pt_params)
        sae.cpu()
    else:
        raise ValueError(f"Unknown sae_load_method: {sae_load_method}")

    return sae


def load_sae_steer(path):
    # Read the configuration for SAE steering
    with open(os.path.join(path, "feature_steer.json"), 'r') as f:
        config = json.load(f)

    # Load SAE model
    sae = load_sae_model(config)

    # Get steering vector
    vectors = []
    for ft_id, ft_scale in config['features']:
        vectors.append(sae.W_dec[ft_id] * ft_scale)
    vectors = torch.stack(vectors, dim=0)
    vec = vectors.sum(dim=0)
    vec = vec / torch.norm(vec, dim=-1, keepdim=True)
    hp = config['hp']
    layer = config['layer']

    return vec, hp, layer


def single_step_steer(adapter, target, bias_scale=1):
    # used for optimised steering
    steer_vec = adapter.W @ target.to(device)
    steer_vec = steer_vec / torch.norm(steer_vec)
    bias_vec = adapter.W @ adapter.b
    bias_vec = bias_vec / torch.norm(bias_vec)
    bias_vec = bias_vec * bias_scale
    steer = steer_vec - bias_vec
    steer = steer / torch.norm(steer, dim=-1, keepdim=True)
    return steer


def load_optimised_steer(path):
    with open(os.path.join(path, "optimised_steer.json"), 'r') as f:
        config = json.load(f)
    
    layer = config['layer']

    sae = load_sae_model(config)

    adapter = LinearAdapter(sae.W_enc.shape[0], sae.W_enc.shape[1])
    adapter.load_state_dict(torch.load(f"adapter_layer_{layer}.pt"))
    adapter.to(device)

    target = torch.zeros(adapter.W.shape[1]).to(device)
    for ft_id, ft_scale in config['features']:
        target[ft_id] = ft_scale

    vec = single_step_steer(adapter, target, bias_scale=1)

    hp = config['hp']
    return vec, hp, layer


def load_rotation_steer(path):
    # Like optimised steer, same config, but use rotation matrix
    with open(os.path.join(path, "optimised_steer.json"), 'r') as f:
        config = json.load(f)
    
    layer = config['layer']
    hp = config['hp']

    sae = load_sae_model(config)
    sae.to(device)
    
    R = torch.load(f"R_dec_layer_{layer}.pt")
    R = R.to(device)
    b = torch.load(f"correction_bias_layer_{layer}.pt")
    b = b.to(device)

    vectors = []
    for ft_id, ft_scale in config['features']:
        vectors.append(sae.W_dec[ft_id] * ft_scale)
    vectors = torch.stack(vectors, dim=0)
    vec = vectors.sum(dim=0)
    vec = vec / torch.norm(vec, dim=-1, keepdim=True)

    steer = R.T @ vec
    steer = steer / torch.norm(steer)
    steer = steer - b
    steer = steer / torch.norm(steer)

    return steer, hp, layer


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

    # Save data used to make the graphs
    graph_data = {
        'path': path,
        'method': method,
        'steering_goal_name': steering_goal_name,
        'scales': scales,
        'avg_coherence': avg_coh,
        'avg_score': avg_score,
        'product': product
    }

    return result, graph_data

# %%
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    big_model = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if big_model:
        model = HookedTransformer.from_pretrained("google/gemma-2-9b", device=device, dtype=torch.float16)
        model_name = "gemma-2-9b"
    else:
        model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device, dtype=torch.float16)
        model_name = "gemma-2-2b"
# %%
if __name__ == "__main__":

    paths = [
        "steer_cfgs/gemma2/anger",
        "steer_cfgs/gemma2/christian_evangelist",
        "steer_cfgs/gemma2/conspiracy",
        "steer_cfgs/gemma2/french",
        "steer_cfgs/gemma2/london",
        "steer_cfgs/gemma2/love",
        "steer_cfgs/gemma2/praise",
        "steer_cfgs/gemma2/want_to_die",
        "steer_cfgs/gemma2/wedding",

        # "steer_cfgs/gemma2/london_65k",
        # "steer_cfgs/gemma2/GGB_65k",

        # "steer_cfgs/gemma2/citations",
    ]

    results = []
    graph_data_list = []

    for path in paths:
        print(path)
        # Activation Steering
        print("Activation Steering")
        pos_examples, neg_examples, val_examples, layer = load_act_steer(path)
        steer = get_activation_steering(model, pos_examples, neg_examples, device=device, layer=layer)
        steer = steer / torch.norm(steer, dim=-1, keepdim=True)
        hp = f"blocks.{layer}.hook_resid_post"
        result, graph_data = analyse_steer(model, steer, hp, path, method='ActSteer')
        results.append(result)
        graph_data_list.append(graph_data)

        # SAE Steering
        print("SAE Steering")
        steer, hp, layer = load_sae_steer(path)
        steer = steer.to(device)
        result, graph_data = analyse_steer(model, steer, hp, path, method='SAE')
        results.append(result)
        graph_data_list.append(graph_data)

        # Optimized Steering
        print("Optimized Steering")
        steer, hp, layer = load_optimised_steer(path)
        steer = steer.to(device)
        result, graph_data = analyse_steer(model, steer, hp, path, method='OptimisedSteer')
        results.append(result)
        graph_data_list.append(graph_data)

        # Rotation Steering
        print("Rotation Steering")
        steer, hp, layer = load_rotation_steer(path)
        steer = steer.to(device)
        result, graph_data = analyse_steer(model, steer, hp, path, method='RotationSteer')
        results.append(result)
        graph_data_list.append(graph_data)

    # Write the results to a JSON file
    with open(f'steering_results_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Write the graph data to a JSON file
    with open(f'graph_data_all_methods_{model_name}.json', 'w') as f:
        json.dump(graph_data_list, f, indent=2)

# # %%
# if __name__ == "__main__":
#     paths = [
#         "steer_cfgs/london_g2/layer_1",
#         "steer_cfgs/london_g2/layer_2",
#         # "steer_cfgs/london_g2/layer_3",
#         "steer_cfgs/london_g2/layer_4",
#         # "steer_cfgs/london_g2/layer_5",
#         "steer_cfgs/london_g2/layer_8",
#         "steer_cfgs/london_g2/layer_12",
#         "steer_cfgs/london_g2/layer_13",
#         "steer_cfgs/london_g2/layer_16",
#         "steer_cfgs/london_g2/layer_22",
#         # "steer_cfgs/london_g2/layer_24",
#     ]

#     results = []
#     graph_data_list = []

#     for path in paths:
#         # Activation Steering
#         print("Activation Steering")
#         pos_examples, neg_examples, val_examples, layer = load_act_steer(path)
#         steer = get_activation_steering(model, pos_examples, neg_examples, device=device, layer=layer)
#         steer = steer / torch.norm(steer, dim=-1, keepdim=True)
#         hp = f"blocks.{layer}.hook_resid_post"
#         result, graph_data = analyse_steer(model, steer, hp, path, method='ActSteer')
#         results.append(result)
#         graph_data_list.append(graph_data)

#         # SAE Steering
#         print("SAE Steering")
#         steer, hp, layer = load_sae_steer(path)
#         steer = steer.to(device)
#         result, graph_data = analyse_steer(model, steer, hp, path, method='SAE')
#         results.append(result)
#         graph_data_list.append(graph_data) 

#     # Write the results to a JSON file
#     with open('steering_results_london.json', 'w') as f:
#         json.dump(results, f, indent=2)

#     # Write the graph data to a JSON file
#     with open('graph_data_all_methods_london.json', 'w') as f:
#         json.dump(graph_data_list, f, indent=2)

# %%
