import sys
sys.path.append('..')
from steering.patch import patch_resid

import torch
from tqdm import tqdm
from functools import partial



PROMPTS = [
    "",
    "",
    "",
    "I think",
    "Breaking news",
    "Last night",
    "For sale",
    "The weather",
    "Dear Sir/Madam",
    "Preheat the oven",
    "It's interesting that"
    "Assistant:",
    "I went up to",
    "New study suggests",
    ]


def gen(
        model,
        steer=None,
        scale=60,
        batch_size=64,
        max_toks=32,
        n_batches=1,
        verbose=False,
        hp="blocks.12.hook_resid_post",
    ):
    if steer is not None:
        hooks = [(hp, partial(patch_resid, steering=steer, scale=scale))]
    else:
        hooks = []
    generated_tokens = []
    for prompt in tqdm(PROMPTS, disable=not verbose):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        prompt_batch = tokens.expand(batch_size, -1)
        for _ in range(n_batches):
            with model.hooks(hooks):
                gen_batch = model.generate(
                    prompt_batch,
                    max_new_tokens=max_toks - tokens.shape[-1],
                    top_k=50,
                    top_p=0.3,
                    verbose=False,
                )
            generated_tokens.append(gen_batch)
    return torch.cat(generated_tokens, dim=0)
            

def get_feature_acts(model, sae, tokens, batch_size, hp="blocks.12.hook_resid_post"):
    assert tokens.shape[1] == 32
    all_sae_acts = torch.zeros(sae.W_dec.shape[0], device=sae.W_dec.device)
    count = 0
    for i in range(0, tokens.shape[0], batch_size):
        batch = tokens[i:i+batch_size]
        _, acts = model.run_with_cache(batch, names_filter=hp, stop_at_layer=13)
        acts = acts[hp] # shape (batch_size, len, d_model)
        acts = acts.reshape(-1, acts.shape[-1]) # shape (batch_size * len, d_model)
        sae_acts = sae.encode(acts)
        all_sae_acts += sae_acts.sum(dim=0)
        count += sae_acts.shape[0]
    return all_sae_acts / count


def get_scale(
        model,
        steer,
        loader,
        scales,
        n_batches=2,
        target_loss=6,
        hp="blocks.12.hook_resid_post",
    ):
    assert torch.allclose(torch.norm(steer), torch.tensor(1.0))
    losses = []
    for scale in scales:
        total_loss = 0
        for batch_idx, batch in enumerate(loader):
            with model.hooks(fwd_hooks=[(hp, partial(patch_resid, steering=steer, scale=scale))]):
                loss = model(batch['tokens'], return_type='loss')
                total_loss += loss
            if batch_idx >= n_batches:
                break
        losses.append(total_loss.item() / n_batches)
        if total_loss.item()/n_batches > target_loss:
            break
    scales = scales[:len(losses)]
    print("scales", scales)
    print("losses", losses)

    # linear interpolation
    x1, x2 = scales[-2], scales[-1]
    y1, y2 = losses[-2], losses[-1]
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    target_scale = (target_loss - b) / m
    return target_scale

