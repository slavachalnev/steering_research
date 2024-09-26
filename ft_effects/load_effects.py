# %%
import os
import torch
import plotly.express as px
import pandas as pd
import numpy as np
import einops


# %%
paths = [ # small model
    # "effects/G2_2B_L12/65k_from_0",
    # "effects/G2_2B_L12/65k_from_10k",
    # "effects/G2_2B_L12/65k_from_20k",
    # "effects/G2_2B_L12/65k_from_30k",
    # "effects/G2_2B_L12/65k_from_40k",

    # "effects/G2_2B_L12/16k_from_0",
    # "effects/G2_2B_L12/sample_and_combine_16k",

    # "effects/G2_2B_L12/random",
    # "effects/G2_2B_L12/random_2",
    # "effects/G2_2B_L12/random_3",
    # "effects/G2_2B_L12/random_4",
    # "effects/G2_2B_L12/random_5",
]

paths = [ # big model
    "effects/G2_9B_L12/131k_from_0",
    "effects/G2_9B_L12/131k_from_64k",
]

features = []
effects = []

for path in paths:
    features.append(torch.load(os.path.join(path, "used_features.pt")))
    effects.append(torch.load(os.path.join(path, "all_effects.pt")))

features = torch.cat(features)
effects = torch.cat(effects)
# %%
print(features.shape)
print(effects.shape)
print(torch.norm(features[4230]))
print(torch.norm(features[14455]))

# %%

# compute sum of absolute values of effects
effects_abs = effects.abs().sum(dim=0)
effects_abs[:20]

# %%

# plot histogram of effects_abs

fig = px.histogram(effects_abs)
fig.show()




# %%
# feature id 115
effects_abs[115]

# %%
# get bottom 10 features
bottom_10 = effects_abs.argsort()[:10]
print(bottom_10)
print(effects_abs[bottom_10])

# %%

