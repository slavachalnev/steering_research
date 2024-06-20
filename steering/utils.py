from typing import List, Callable, Union, Optional, Literal

from functools import partial
from tqdm import tqdm
import json

import torch
from torch.utils.data import DataLoader
import einops

from datasets import load_dataset

from transformer_lens import HookedTransformer
from transformer_lens.utilities import devices
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
import transformer_lens.utils as tutils


@torch.no_grad()
def normalise_decoder(sae, scale_input=False):
    """
    Normalises the decoder weights of the SAE to have unit norm.
    
    Use this when loading for gemma-2b saes.

    Args:
        sae (SparseAutoencoder): The sparse autoencoder.
        scale_input (bool): Use this when loading layer 12 model.
    """
    norms = torch.norm(sae.W_dec, dim=1)
    sae.W_dec /= norms[:, None]
    sae.W_enc *= norms[None, :]
    sae.b_enc *= norms

    if scale_input:
        raise NotImplementedError()
        sae.W_enc *= 0.2175 # computed in slava_scratch/scale_sae.ipynb
