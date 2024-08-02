# %%
import os
import sys
sys.path.append(os.path.abspath('..'))

from steering.patch import patch_resid
from steering.sae import JumpReLUSAE

from functools import partial

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
import numpy as np
from huggingface_hub import hf_hub_download, HfFileSystem

torch.set_grad_enabled(False)

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)


# %%

hp = "blocks.12.hook_resid_post"
path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-2b-pt-res",
    filename="layer_12/width_16k/average_l0_82/params.npz",
    force_download=False,
)

params = np.load(path_to_params)
pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
sae.load_state_dict(pt_params)
sae.to(device)

# %%

london = sae.W_dec[14455]
uk = sae.W_dec[3931]
place = sae.W_dec[865]

sf = sae.W_dec[3124]

marriage = sae.W_dec[2421]
wedding = sae.W_dec[4230]


def gen(prompt, steer, scale):
    toks = model.to_tokens(prompt, prepend_bos=True)
    toks = toks.expand(10, -1)
    with model.hooks([(hp, partial(patch_resid, steering=steer, scale=scale))]):
        gen_toks = model.generate(toks, max_new_tokens=30)
    return model.to_string(gen_toks)

# %%
gen("I think", wedding, 80)


# %%

# %%
