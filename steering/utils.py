from typing import List, Callable, Union, Optional, Literal

from functools import partial

import torch
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download, HfFileSystem

import numpy as np

from sae_lens import SAE

from transformer_lens import HookedTransformer
from transformer_lens.utilities import devices
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
import transformer_lens.utils as tutils


@torch.no_grad()
def text_to_sae_feats(
        model: HookedTransformer,
        sae: SAE,
        hook_point: str,
        text: str,
    ):
    """
    Converts text to SAE features.

    Returns:
        torch.Tensor: SAE activations. Shape: [batch_size, sequence_len, d_sae]
    """

    _, acts = model.run_with_cache(text, names_filter=hook_point)
    acts = acts[hook_point]

    all_sae_acts = []
    for batch in acts:
        sae_acts = sae.encode(batch.to(sae.device))
        all_sae_acts.append(sae_acts)
    
    return torch.stack(all_sae_acts, dim=0)


@torch.no_grad()
def top_activations(activations: torch.Tensor, top_k: int=10):
    """
    Returns the top_k activations for each position in the sequence.
    """
    top_v, top_i = torch.topk(activations, top_k, dim=-1)

    return top_v, top_i


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

        
def get_gemma2_2b_SAE_path(layer, width=16, closest_l0=100):
    fs = HfFileSystem()
    all_paths = fs.glob("google/gemma-scope-2b-pt-res/**/params.npz")
    candidate_paths = [p for p in all_paths if f'layer_{layer}/width_{width}k/average_l0_' in p]
    # get the l0 value from the path
    l0_values = [int(p.split('average_l0_')[1].split('/')[0]) for p in candidate_paths]
    # find the one closest to closest_l0
    idx = np.argmin(np.abs(np.array(l0_values) - closest_l0))
    desire_l0 = l0_values[idx]
    desire_path = candidate_paths[idx]
    return desire_l0, desire_path
