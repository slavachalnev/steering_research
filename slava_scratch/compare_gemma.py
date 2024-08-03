# %%
import os
import sys
sys.path.append(os.path.abspath('..'))
import json

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

from transformers import AutoModelForCausalLM

torch.set_grad_enabled(False)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_tl = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)


# %%
model_hf = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map='auto',
)

# %%





# %%
# %%
