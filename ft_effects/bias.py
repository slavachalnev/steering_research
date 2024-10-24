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

b_steering = - adapter.W @ adapter.b
b_steering = b_steering / b_steering.norm()
b_steering = b_steering.to(device)  # Move steering vector to GPU

# %%

# Load the model
model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device, dtype=torch.float16)
model_name = "gemma-2-2b"

# Define the prompt
prompt = "I think"

# Define scales
scales = list(range(0, 320, 20))  # From 0 to 300 in steps of 20

# Initialize list to store coherence scores
avg_coherence = []

# Define hook point
layer = 12  # As we loaded adapter_layer_12.pt
hp = f'blocks.{layer}.hook_resid_post'


# %%
# Generate texts, compute coherence scores, and plot the results

for scale in tqdm(scales):
    # Generate texts using the steering vector
    texts = steer_model(model, b_steering, hp, prompt, scale=scale, n_samples=256)
    
    # Compute coherence scores using multi_criterion_evaluation
    # Ensure that the evaluation functions are properly defined in steering.evals_utils
    coherence_results = multi_criterion_evaluation(
        texts,
        ['The text is coherent and the grammar is correct.'],
        prompt=prompt,
        print_errors=False,
    )
    
    coherence_list = coherence_results[0]
    
    # Extract and normalize coherence scores
    coherence_scores = [item['score'] for item in coherence_list]
    coherence_scores = [(item - 1) / 9 for item in coherence_scores]  # Normalize to [0,1]
    
    # Compute average coherence score
    avg_coherence.append(sum(coherence_scores) / len(coherence_scores))

# %%
# Plot the results
fig = go.Figure()
fig.add_trace(go.Scatter(x=scales, y=avg_coherence, mode='lines+markers', name='Coherence'))

fig.update_layout(
    xaxis_title='Scale',
    yaxis_title='Coherence Score',
    legend_title='Metric',
    yaxis=dict(range=[0, 1]),
    # Make the figure larger for better PDF quality
    width=800,
    height=600,
    # Improve font sizes for PDF
    font=dict(size=14)
)

# Show the plot
fig.show()

# Save to PDF
fig.write_image("bias_coherence_2b.pdf")
# %%
