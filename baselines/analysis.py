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
from tqdm import tqdm
from steering.evals_utils import multi_criterion_evaluation
from steering.utils import normalise_decoder

from datasets import load_dataset

from baselines.activation_steering import get_activation_steering, load_act_steer
from baselines.caa import compute_diffs_for_dataset, all_layer_effects
from sae_lens import SAE


# %%

def load_sae_steer(path):
    with open(os.path.join(path, "sae_steer.json"), 'r') as f:
        steer = json.load(f)
    # steer is {
    #    "sae": "gemma-2b-res-jb",
    #    "layer": 6,
    #    "hp": "blocks.6.hook_resid_post",
    #    "features": [[155, .5], [23333, .5]] }
    sae, _, _ = SAE.from_pretrained(
        release = steer['sae'],
        sae_id = steer['hp'],
        device = 'cpu')
    normalise_decoder(sae)
    vectors = []
    for ft_id, ft_scale in steer['features']:
        vectors.append(sae.W_dec[ft_id] * ft_scale)
    vectors = torch.stack(vectors, dim=0)
    vec = vectors.sum(dim=0)
    vec = vec / torch.norm(vec, dim=-1, keepdim=True)
    return vec, steer['hp'], steer['layer']

def steer_model(model, steer, layer, text, scale=5, batch_size=64, n_samples=128):
    toks = model.to_tokens(text, prepend_bos=True)
    toks = toks.expand(batch_size, -1)
    hp = f"blocks.{layer}.hook_resid_post"
    all_gen = []
    for i in range(0, n_samples, batch_size):
        with model.hooks([(hp, partial(patch_resid, steering=steer, scale=scale))]):
            gen_toks = model.generate(
                toks,
                max_new_tokens=30,
                use_past_kv_cache=True,
                top_k = 50,
                top_p = 0.3,
                verbose=False,
            )
            all_gen.extend(model.to_string(gen_toks))
    return all_gen

def plot(path, coherence, score, scales, method):
    fig = px.line(
        x=scales,
        y=[coherence, score],
        labels={'x': 'Scale', 'y': 'Score'},
        title=f'Coherence and Score vs Scale for {path} ({method})',
        line_shape='linear'
    )
    fig.data[0].name = 'Coherence'
    fig.data[1].name = 'Score'
    fig.update_layout(legend_title_text='Metric')
    fig.write_image(os.path.join(path, f"scores_{method}.png"))

def analyse_steer(model, steer, layer, path, method='activation_steering'):
    scales = list(range(0, 200, 20))
    with open(os.path.join(path, "criteria.json"), 'r') as f:
        criteria = json.load(f)
    with open(os.path.join(path, "prompts.json"), 'r') as f:
        prompts = json.load(f)
    
    avg_score = []
    avg_coh = []
    for scale in tqdm(scales):
        texts = steer_model(model, steer, layer, prompts[0], scale=scale)
        
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

    plot(path, avg_coh, avg_score, scales, method)



# %%
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("google/gemma-2b", device=device)

# %%
if __name__ == "__main__":

    paths = [
        # "steer_cfgs/golden_gate",
        # "steer_cfgs/love_hate",
        # "steer_cfgs/christianity",
        "steer_cfgs/london",
    ]
    act_steer_layers = [10, 10, 10]

    for path, layer in zip(paths, act_steer_layers):
        # # Activation Steering
        # print("activation steering")
        # pos_examples, neg_examples, val_examples = load_act_steer(path)
        # steer = get_activation_steering(model, pos_examples, neg_examples, device=device, layer=layer)
        # steer = steer / torch.norm(steer, dim=-1, keepdim=True)
        # analyse_steer(model, steer, layer, path, method='activation_steering')

        # # CAA
        # dataset = load_dataset("Anthropic/model-written-evals", data_files="persona/anti-immigration.jsonl")['train']
        # train_dataset = dataset[:-50]
        # test_dataset = dataset[-50:]
        
        # diffs = compute_diffs_for_dataset(train_dataset, model=model, device=device)
        # effects = all_layer_effects(diffs, model=model)
        
        # # Normalize the diffs
        # diffs = diffs / torch.norm(diffs, dim=-1, keepdim=True)
        
        # analyse_steer(model, diffs[layer], layer, path, method='caa')

        # sae steering
        print("sae steering")
        steer, hp, layer = load_sae_steer(path)
        steer = steer.to(device)
        analyse_steer(model, steer, layer, path, method='sae')
    
    


# %%