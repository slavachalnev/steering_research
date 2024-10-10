# %%
import os
import sys
sys.path.append(os.path.abspath('..'))

from ft_effects.utils import gen, get_feature_acts
from transformer_lens import utils as tutils
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from steering.sae import JumpReLUSAE
from huggingface_hub import hf_hub_download
import numpy as np

# %%

@torch.no_grad()
def load_model_and_sae(rank, big_model=False):
    device = torch.device(f"cuda:{rank}")
    if big_model:
        model = HookedTransformer.from_pretrained_no_processing("google/gemma-2-9b", device=device, dtype=torch.float16)
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-9b-pt-res",
            filename="layer_12/width_16k/average_l0_130/params.npz",
            force_download=False)
        loader_batch_size = 4  # Lower batch size for big models
    else:
        model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device, dtype=torch.float16)
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-res",
            filename="layer_12/width_16k/average_l0_82/params.npz",
            force_download=False)
        loader_batch_size = 64  # Higher batch size for small models
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}
    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)
    sae.to(device)
    # prep data
    data = tutils.load_dataset("NeelNanda/c4-code-20k", split="train")
    tokenized_data = tutils.tokenize_and_concatenate(data, model.tokenizer, max_length=32)
    tokenized_data = tokenized_data.shuffle(42)
    loader = DataLoader(tokenized_data, batch_size=loader_batch_size)
    return model, sae, loader


@torch.no_grad()
def get_base_acts(model, sae, big_model=False):

    if big_model:
        # Lower batch sizes for big models
        baseline_samples = gen(model=model, n_batches=20, batch_size=32)
    else:
        # Higher batch sizes for small models
        baseline_samples = gen(model=model, n_batches=10)
    baseline_dist = get_feature_acts(model=model, sae=sae, tokens=baseline_samples, batch_size=32)

    return baseline_dist

# %%
if __name__ == "__main__":
    model, sae, loader = load_model_and_sae(0, big_model=False)
    base_acts = get_base_acts(model, sae, big_model=False)

# %%
if __name__ == "__main__":

    torch.save(base_acts, "base_acts_2b.pt")
    # torch.save(base_acts, "base_acts_9b.pt")

# %%
if __name__ == "__main__":

    print(base_acts.shape)
    print(base_acts[:10])
    print('min', base_acts.min())
    print('max', base_acts.max())

    # Count and print the number of zero values in base_acts
    zero_count = torch.sum(base_acts == 0).item()
    print(f'Number of zero values in base_acts: {zero_count}')
    print(f'Percentage of zero values: {zero_count / base_acts.numel() * 100:.2f}%')


