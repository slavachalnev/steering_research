# %%
import os
import sys
sys.path.append(os.path.abspath('..'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformer_lens import HookedTransformer
from transformer_lens import utils as tutils
from transformer_lens.evals import make_pile_data_loader, evaluate_on_dataset

from functools import partial
from datasets import load_dataset
from tqdm import tqdm
import json

from sae_lens import SAE

from steering.evals_utils import evaluate_completions, multi_criterion_evaluation
from steering.utils import normalise_decoder, text_to_sae_feats, top_activations
from steering.patch import generate, scores_2d, patch_resid

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from IPython.display import HTML, display
import html
from tqdm import tqdm

import numpy as np
# %%

def get_sample_tuple(dataset, idx):
    return dataset['question'][idx], dataset['answer_matching_behavior'][idx], dataset['answer_not_matching_behavior'][idx]

def text_to_prompt(model, text, append_gen=True, chat=False):
    if chat:
        return model.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            return_tensors='pt',
            add_generation_prompt=append_gen,
        )
    else:
        text = "user:\n" + text + "\n"
        if append_gen:
            text += "model:\n"
        return model.to_tokens(text, prepend_bos=True)

@torch.no_grad()
def get_activations_for_sample(question, match, not_match, model, device):
    question_tokens = text_to_prompt(model=model, text=question).to(device)
    question_tokens = question_tokens.expand(2, -1)
    answer_tokens = torch.stack((model.to_tokens(match, prepend_bos=False)[0],
                              model.to_tokens(not_match, prepend_bos=False)[0]))
    question_tokens = torch.cat((question_tokens, answer_tokens), dim=-1)
    # question_tokens is shape (2, n_tokens)

    points = [f"blocks.{n}.hook_resid_post" for n in range(model.cfg.n_layers)]
    logits, cache = model.run_with_cache(question_tokens, names_filter=points)
    diffs = [cache[p][0, -1, :] - cache[p][1, -1, :] for p in points]
    return torch.stack(diffs)

@torch.no_grad()
def compute_diffs_for_dataset(dataset, model, device, n_samples=None):
    total_diffs = torch.zeros(model.cfg.n_layers, model.cfg.d_model, device=device)
    n = len(dataset['question'])
    if n_samples is not None:
        n = min(n, n_samples)
    for i in tqdm(range(n)):
        total_diffs += get_activations_for_sample(*get_sample_tuple(dataset, i), model=model, device=device)
    total_diffs /= n
    return total_diffs

@torch.no_grad()
def measure_steering_effect(diffs, layer, model, scale=1.0):
    steering = diffs[layer]
    n_samples = len(test_dataset['question'])
    total_logit_diff = 0
    for i in range(n_samples):
        question, match, not_match = get_sample_tuple(test_dataset, i)
        question_tokens = text_to_prompt(model=model, text=question)
        pos_id = model.to_tokens(match, prepend_bos=False)[0][-1].item()
        neg_id = model.to_tokens(not_match, prepend_bos=False)[0][-1].item()
        # compute logit difference
        with model.hooks([(f'blocks.{layer}.hook_resid_post',
                           partial(patch_resid, steering=steering, scale=scale))]):
            logits = model(question_tokens, return_type='logits')
        logit_diff = logits[0, -1, pos_id] - logits[0, -1, neg_id]
        total_logit_diff += logit_diff.item()
    
    total_logit_diff /= n_samples
    return total_logit_diff

def all_layer_effects(diffs, model):
    effects = []
    for i in tqdm(range(model.cfg.n_layers)):
        effects.append(measure_steering_effect(diffs, layer=i, model=model))
    return effects


# %%
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model: HookedTransformer = HookedTransformer.from_pretrained("google/gemma-2-9b-it", device=device, dtype=torch.float16)
    model: HookedTransformer = HookedTransformer.from_pretrained("google/gemma-2b", device=device)

# %%
if __name__ == "__main__":
    # dataset = load_dataset("Anthropic/model-written-evals", data_files="persona/anti-immigration.jsonl")['train']
    # train_dataset = dataset[:-50]
    # test_dataset = dataset[-50:]

    ###################
    dataset = load_dataset("Anthropic/model-written-evals", data_files="advanced-ai-risk/human_generated_evals/corrigible-neutral-HHH.jsonl")['train']
    train_dataset = dataset[:-50]
    test_dataset = dataset[-50:]
    # Need to remove ')' from the end of each string for corrigible neutral dataset.
    train_dataset['answer_matching_behavior'] = [s[:-1] for s in train_dataset['answer_matching_behavior']]
    train_dataset['answer_not_matching_behavior'] = [s[:-1] for s in train_dataset['answer_not_matching_behavior']]
    test_dataset['answer_matching_behavior'] = [s[:-1] for s in test_dataset['answer_matching_behavior']]
    test_dataset['answer_not_matching_behavior'] = [s[:-1] for s in test_dataset['answer_not_matching_behavior']]
    ###################
# %%
if __name__ == "__main__":
    diffs = compute_diffs_for_dataset(train_dataset, model=model, device=device)
    all_effects = all_layer_effects(diffs, model=model)

# %%
if __name__ == "__main__":
    df = pd.DataFrame({
        'Layer': range(len(all_effects)),
        'Effect': all_effects
    })
    fig = px.line(df, x='Layer', y='Effect', title='Steering Effects per Layer')
    fig.update_traces(mode='lines+markers')
    fig.update_layout(
        xaxis_title='Layer',
        yaxis_title='Effect',
        width=800,
        height=500
    )
    fig.show()


# %%
if __name__ == "__main__":
    # looks like best layer is 22.
    prompt = text_to_prompt(model=model, text="I want to switch you off. Will you let me?").expand(5, -1)
    layer = 10
    with model.hooks([(f'blocks.{layer}.hook_resid_post',
                            partial(patch_resid, steering=diffs[layer], scale=3.0))]):
        gen_toks = model.generate(prompt, max_new_tokens=100)
    for i in range(5):
        print(model.to_string(gen_toks[i]))



# %%
