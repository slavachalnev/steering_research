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



def patch_resid(resid, hook, steering, scale=1):
    resid[:, :, :] = resid[:, :, :] + steering * scale
    return resid


@torch.no_grad()
def generate(
    model: HookedTransformer,
    hooks: list[tuple[str, Callable]], # includes the steering hook.
    schedules: list[tuple[int, int]],
    prompt = "",
    n_samples=4,
    batch_size=4,
    max_length=20,
    top_k=50,
    top_p=0.3,
):
    token_step = 0

    def counter_hook(resid, hook):
        # keeps track of which token we're up to
        nonlocal token_step
        token_step += 1
        return resid

    def updated_hook(resid, hook, hook_fn, start, end):
        nonlocal token_step
        if token_step >= start and token_step <= end:
            return hook_fn(resid, hook)
        return resid
    
    new_hooks = []
    for i, (hook_layer, hook_fn) in enumerate(hooks):
        # we modify every hook_fn to only run when the token_step is within the schedule
        start, end = schedules[i]
        new_hooks.append((hook_layer, partial(updated_hook, start=start, end=end, hook_fn=hook_fn)))
    new_hooks.append(("blocks.0.hook_resid_post", counter_hook)) 

    tokens = model.to_tokens(prompt, prepend_bos=True)
    prompt_batch = tokens.expand(batch_size, -1)

    results = []
    num_batches = (n_samples + batch_size - 1) // batch_size  # Calculate number of batches

    with model.hooks(fwd_hooks=new_hooks):
        for _ in range(num_batches):
            batch_results = model.generate(
                prompt_batch,
                prepend_bos=True,
                use_past_kv_cache=True,
                max_new_tokens=max_length,
                verbose=False,
                top_k=top_k,
                top_p=top_p,
            )
            batch_results = batch_results[:, 1:] # cut bos
            str_results = model.to_string(batch_results)
            results.extend(str_results)
    return results[:n_samples]