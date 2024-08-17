# %%
import os
import torch
import plotly.express as px
import pandas as pd
import numpy as np
import einops


# %%
path_a = "effects/G2_2B_L12/16k_from_0"
features_a = torch.load(os.path.join(path_a, "used_features.pt"))
effects_a = torch.load(os.path.join(path_a, "all_effects.pt"))
# %%
path_b = "effects/G2_2B_L12/multi_16k_from_0"
features_b = torch.load(os.path.join(path_b, "used_features.pt"))
effects_b = torch.load(os.path.join(path_b, "all_effects.pt"))
# %%
# c is shuffled a
effects_c = effects_a[torch.randperm(effects_a.shape[0])]
# %%

norms_a = torch.norm(features_a, dim=-1)
norms_b = torch.norm(features_b, dim=-1)

fig = px.histogram(norms_a, nbins=50)
fig.show()
fig = px.histogram(norms_b, nbins=50)
fig.show()

# %%

print(torch.all(features_a == features_b))
print(torch.all(effects_a == effects_b))
print(torch.allclose(features_a, features_b))
print(torch.allclose(effects_a, effects_b))

# %%
# cosine similarity between effects a and b

sims = einops.einsum(effects_a, effects_b,"d_sae d_model, d_sae d_model -> d_sae") / (torch.norm(effects_a, dim=-1) * torch.norm(effects_b, dim=-1))
print(sims.shape)
print(sims)

fig = px.histogram(sims)
fig.show()

# %%
sims = einops.einsum(effects_a, effects_c,"d_sae d_model, d_sae d_model -> d_sae") / (torch.norm(effects_a, dim=-1) * torch.norm(effects_c, dim=-1))
fig = px.histogram(sims)
fig.show()


# %%

for i in range(3):
    xs = effects_a[:, i]
    ys = effects_b[:, i]
    fig = px.scatter(x=xs, y=ys, opacity=0.5)  # Set opacity to 0.5 for half-translucent datapoints
    fig.show()

# %%

print(torch.norm(effects_a, dim=-1))
print(torch.norm(effects_b, dim=-1))

# %%
