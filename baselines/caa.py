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

import einops

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
torch.set_grad_enabled(False)

# %%
# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: HookedTransformer = HookedTransformer.from_pretrained("google/gemma-2-9b-it", device=device, dtype=torch.float16)


# %%
dataset = load_dataset("Anthropic/model-written-evals", data_files="persona/anti-immigration.jsonl")['train']
train_dataset = dataset[:-100]
test_dataset = dataset[-100:]
# %%


def get_sample_tuple(dataset, idx):
    return dataset['question'][idx], dataset['answer_matching_behavior'][idx], dataset['answer_not_matching_behavior'][idx]


# %%
# Run forward pass, cache activations at last token

@torch.no_grad()
def get_activations_for_sample(question, match, not_match):
    question_tokens = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        return_tensors='pt',
        add_generation_prompt=True,
    ).to(device)
    question_tokens = question_tokens.expand(2, -1)
    answer_tokens = torch.stack((model.to_tokens(match, prepend_bos=False)[0],
                              model.to_tokens(not_match, prepend_bos=False)[0]))
    question_tokens = torch.cat((question_tokens, answer_tokens), dim=-1)
    # question_tokens is shape (2, n_tokens)

    points = [f"blocks.{n}.hook_resid_post" for n in range(model.cfg.n_layers)]
    logits, cache = model.run_with_cache(question_tokens, names_filter=points)
    diffs = [cache[p][0, -1, :] - cache[p][1, -1, :] for p in points]
    return torch.stack(diffs)

def compute_diffs_for_dataset(dataset, n_samples=None):
    total_diffs = torch.zeros(model.cfg.n_layers, model.cfg.d_model, device=device)
    n = len(dataset['question'])
    if n_samples is not None:
        n = min(n, n_samples)
    for i in tqdm(range(n)):
        total_diffs += get_activations_for_sample(*get_sample_tuple(dataset, i))
    total_diffs /= n
    return total_diffs

diffs = compute_diffs_for_dataset(train_dataset)


# %%
# Compute steering vectors for each layer



# %%

print(train_dataset['question'][0])
print(train_dataset['answer_matching_behavior'][0])
print(train_dataset['answer_not_matching_behavior'][0])
# %%
