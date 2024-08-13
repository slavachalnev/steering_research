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

from baselines.activation_steering import get_activation_steering, load_act_steer

# %%

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

def plot(path, coherence, score, scales):
    fig = px.line(
        x=scales,
        y=[coherence, score],
        labels={'x': 'Scale', 'y': 'Score'},
        title=f'Coherence and Score vs Scale for {path}',
        line_shape='linear'
    )
    fig.data[0].name = 'Coherence'
    fig.data[1].name = 'Score'
    fig.update_layout(legend_title_text='Metric')
    fig.show()
    fig.write_image(os.path.join(path, "scores.png"))


def analyse_steer(model, steer, layer, path):
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

    plot(path, avg_coh, avg_score, scales)




# %%
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("google/gemma-2b", device=device)

# %%
if __name__ == "__main__":
    paths = [
        "steer_cfgs/golden_gate",
        # "steer_cfgs/love_hate",
        # "steer_cfgs/christianity",
    ]

    for path in paths:
        layer = 10
        pos_examples, neg_examples, val_examples = load_act_steer(path)
        steer = get_activation_steering(model, pos_examples, neg_examples, device=device, layer=layer)
        steer = steer / torch.norm(steer, dim=-1, keepdim=True)

        # gen_text = steer_model(model, steer, layer=layer, text="What is your name?", scale=15)
        # print(gen_text)
        analyse_steer(model, steer, layer, path)
        


# given a steering vector and a path to eval config, do analysis.
# and really the only analysis we need is scale vs score and coherence.
# save the plot to file in same dir as eval config.





# %%