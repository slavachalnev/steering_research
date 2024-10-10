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

    "effects/G2_2B_L12/16k_from_0",
    # "effects/G2_2B_L12/sample_and_combine_16k",

    # "effects/G2_2B_L12/random",
    # "effects/G2_2B_L12/random_2",
    # "effects/G2_2B_L12/random_3",
    # "effects/G2_2B_L12/random_4",
    # "effects/G2_2B_L12/random_5",
]

# paths = [  # big model
#     "../effects/G2_2B_L12/16k",
#     # "effects/G2_9B_L12/131k_from_0",
#     # "effects/G2_9B_L12/131k_from_16k",
#     # "effects/G2_9B_L12/131k_from_32k",
# ]

features = []
effects = []

for path in paths:
    features.append(torch.load(os.path.join(path, "used_features.pt")))
    effects.append(torch.load(os.path.join(path, "all_effects.pt")))

features = torch.cat(features)
effects = torch.cat(effects)


# %%
def check_feature_distributions():
    # Define a threshold value
    threshold = 0.4  # Adjust this value as needed

    zeros = []
    different = []
    same = []
    only_dense = []
    for i in range(effects.shape[0]):
        # Create a boolean mask for significant effects
        significant_mask = effects[i] > threshold

        if significant_mask.nonzero().squeeze().tolist() == []:
            zeros.append([i, []])
            continue
        # Get indices of significant effects from the original list
        indices = significant_mask.nonzero().squeeze().tolist()

        # Ensure indices is always a list
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        # Sort indices by their corresponding effect values in descending order
        indices.sort(key=lambda idx: effects[i][idx], reverse=True)
        filter_features = [7507, 6810, 2620, 2009, 7517, 10957, 2291]
        indices = list(filter(lambda idx: idx not in filter_features, indices))

        if len(indices) == 0:
            only_dense.append([i, []])
        elif i not in indices:
            different.append([i, indices])
        else:
            same.append([i, indices])

    print("zeros:", len(zeros))
    print("different:", len(different))
    print("same:", len(same))
    print("only_dense:", len(only_dense))
    return zeros, different, same, only_dense


# %%

# norm of first 10 features
print(features[:10].norm(dim=-1))

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

# load base acts
base_acts = torch.load("base_acts_2b.pt").to('cpu')
print(base_acts.shape)

# %%

# wedding
wedding_effects = effects[4230] / (base_acts + 1e-2)
print(torch.topk(wedding_effects, k=10))

# %%

# facilitate
facilitate_effects = effects[129] / (base_acts + 1e-2)
print(torch.topk(facilitate_effects, k=10))

# %%

# months
months_effects = effects[12904]# / (base_acts + 1e-2)
print(torch.topk(months_effects, k=10))


# %%
