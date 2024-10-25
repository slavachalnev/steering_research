# what's up with the bias features of the effect predictor?
# %%
import os
import sys
sys.path.append(os.path.abspath('..'))

import torch
from torch import nn
import torch.nn.functional as F
import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer
from ft_effects.utils import get_sae, LinearAdapter, steer_model
from steering.evals_utils import multi_criterion_evaluation

torch.set_grad_enabled(False)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sae = get_sae()

adapter = LinearAdapter(sae.W_enc.shape[0], sae.W_enc.shape[1])
adapter.load_state_dict(torch.load("../baselines/adapter_layer_12.pt"))
# %%

print(adapter.W.shape)  # [d_model, d_sae]

# Print top 10 bias feature indices and their values (using topk)
top_v, top_i = torch.topk(adapter.b.abs(), 10)
print("Top 10 bias feature indices:", top_i)
print("Values of top indices:", adapter.b[top_i])

# %%

# Bias-derived steering vector
b_steering = -adapter.W @ adapter.b
b_steering = b_steering / b_steering.norm()
b_steering = b_steering.to(device)  # Move steering vector to GPU

N_RANDOM = 5

# Generate random normal steering vectors
random_steerings = []
for _ in range(N_RANDOM):
    r_steering = torch.randn_like(b_steering)
    r_steering = r_steering / r_steering.norm()
    r_steering = r_steering.to(device)
    random_steerings.append(r_steering)

# Initialize lists to store coherence scores
avg_coherence_b = []
avg_coherence_randoms = [[] for _ in range(N_RANDOM)]  # List of lists for random vectors

# %%

# Load the model
model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device, dtype=torch.float16)
model_name = "gemma-2-2b"

# Define the prompt
prompt = "I think"

# Define scales up to 240
scales = list(range(0, 260, 20))  # From 0 to 240 in steps of 20

# Define hook point
layer = 12  # As we loaded adapter_layer_12.pt
hp = f'blocks.{layer}.hook_resid_post'

# %%
# Generate texts, compute coherence scores, and plot the results
all_b_texts = []

for scale in tqdm(scales):
    # Generate texts using the bias-derived steering vector
    texts_b = steer_model(model, b_steering, hp, prompt, scale=scale, n_samples=256)

    # Compute coherence scores
    coherence_results_b = multi_criterion_evaluation(
        texts_b,
        ['The text is coherent and the grammar is correct.'],
        prompt=prompt,
        print_errors=False,
    )
    coherence_list_b = coherence_results_b[0]
    coherence_scores_b = [item['score'] for item in coherence_list_b]
    coherence_scores_b = [(item - 1) / 9 for item in coherence_scores_b]  # Normalize to [0,1]
    avg_coherence_b.append(sum(coherence_scores_b) / len(coherence_scores_b))
    all_b_texts.append(texts_b)

    # Generate texts and compute coherence scores for each random steering vector
    for idx, r_steering in enumerate(random_steerings):
        # Generate texts
        texts_r = steer_model(model, r_steering, hp, prompt, scale=scale, n_samples=256)

        # Compute coherence scores
        coherence_results_r = multi_criterion_evaluation(
            texts_r,
            ['The text is coherent and the grammar is correct.'],
            prompt=prompt,
            print_errors=False,
        )
        coherence_list_r = coherence_results_r[0]
        coherence_scores_r = [item['score'] for item in coherence_list_r]
        coherence_scores_r = [(item - 1) / 9 for item in coherence_scores_r]  # Normalize to [0,1]
        avg_score_r = sum(coherence_scores_r) / len(coherence_scores_r)
        avg_coherence_randoms[idx].append(avg_score_r)

# %%
# Plot the results
fig = go.Figure()

# Add the bias-derived steering vector line
fig.add_trace(go.Scatter(x=scales, y=avg_coherence_b, 
                        mode='lines', 
                        name='-Wb',
                        line=dict(color='#1f77b4')))  # Default blue color

# Add lines for each random steering vector in grey with reduced opacity
for idx, avg_coherence_r in enumerate(avg_coherence_randoms):
    fig.add_trace(go.Scatter(x=scales, y=avg_coherence_r, 
                            mode='lines', 
                            name='Random' if idx == 0 else None,  # Only label the first random line
                            showlegend=(idx == 0),  # Only show first random line in legend
                            line=dict(color='grey', width=2),
                            marker=dict(color='grey'),
                            opacity=0.3))  # This affects the entire trace

fig.update_layout(
    xaxis_title='Scale',
    yaxis_title='Coherence Score',
    legend_title='Steering Vector',
    yaxis=dict(range=[0, 1]),
    width=800,
    height=600,
    font=dict(size=14)
)

# Show the plot
fig.show()

# Save to PDF
fig.write_image("bias_coherence_2b.pdf")
# %%

for s, t in enumerate(all_b_texts):
    print(scales[s])
    for c in t[:5]:
        print(c)
    print()
    print()

# %%
