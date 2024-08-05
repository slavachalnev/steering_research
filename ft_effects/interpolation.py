# %%
import os
import torch
import plotly.express as px
import pandas as pd
import numpy as np
# %%

path = "effects/G2_2B_L12/interp_wedding_dish"

features = torch.load(os.path.join(path, "used_features.pt"))
effects = torch.load(os.path.join(path, "all_effects.pt"))

# %%

norms = torch.norm(features, dim=-1)
fig = px.line(norms)
fig.show()

# %%

cosine_sims_to_wedding = torch.einsum("fd,d->f", features, features[0])
cosine_sims_to_wedding = cosine_sims_to_wedding / norms[0] / norms
print(cosine_sims_to_wedding)

# %%
print(effects.shape)

# interpolate between effects[0] and effects[-1]
interpolated_effects = []
for alpha in torch.linspace(0, 1, 21):
    interpolated_effects.append(alpha * effects[-1] + (1 - alpha) * effects[0])
interpolated_effects = torch.stack(interpolated_effects)

# compute mse between interpolated effects and effects
sse = torch.sum((interpolated_effects - effects) ** 2, dim=-1)
sse_0 = torch.sum((effects[0] - effects) ** 2, dim=-1)
sse_1 = torch.sum((effects[-1] - effects) ** 2, dim=-1)
sse_z = torch.sum((torch.zeros_like(effects) - effects) ** 2, dim=-1)

# %%

fig = px.line(sse)
fig.show()

# %%

df = pd.DataFrame({
    'Step': range(len(sse)),
    'Interpolated': sse.numpy(),
    'First Effect': sse_0.numpy(),
    'Last Effect': sse_1.numpy(),
    'Zero Effect': sse_z.numpy()  # Add the sse_z data
})
# melt the DataFrame for easier plotting
df_melted = pd.melt(df, id_vars=['Step'], var_name='Type', value_name='SSE')
# create the plot with log scale y-axis
fig = px.line(df_melted, x='Step', y='SSE', color='Type', 
              title='SSE Curves',
              labels={'Step': 'Interpolation Step', 'SSE': 'Sum of Squared Errors'},
            #   log_y=True,
              )
# Customize the layout
fig.update_layout(
    yaxis_title="Sum of Squared Errors",
    legend_title="Effect Type",
    font=dict(size=12)
)
fig.show()

# %%

# Get top 4 indices for wedding and serve_dish
top_wedding = torch.argsort(effects[0], descending=True)[:4]
top_serve_dish = torch.argsort(effects[-1], descending=True)[:4]
# Combine the indices
top_indices = torch.cat([top_wedding, top_serve_dish])
# Create a DataFrame for plotting
df_effects = pd.DataFrame({
    'Step': range(len(effects)),
    **{f'Effect_{i}': effects[:, idx].numpy() for i, idx in enumerate(top_indices)}
})
# Calculate linear interpolation for each effect
for i, idx in enumerate(top_indices):
    start_value = effects[0, idx].item()
    end_value = effects[-1, idx].item()
    linear_interp = np.linspace(start_value, end_value, len(effects))
    df_effects[f'Linear_{i}'] = linear_interp
# Melt the DataFrame for easier plotting
df_effects_melted = pd.melt(df_effects, id_vars=['Step'], var_name='Effect', value_name='Value')
# Create the plot
fig = px.line(df_effects_melted, x='Step', y='Value', color='Effect',
              title='Top 4 Effects for Wedding and Serve Dish with Linear Interpolation',
              labels={'Step': 'Interpolation Step', 'Value': 'Effect Value'})
# Customize the layout
fig.update_layout(
    xaxis_title="Interpolation Step",
    yaxis_title="Effect Value",
    legend_title="Effect",
    font=dict(size=12)
)

color_map = {}
for trace in fig.data:
    if trace.name.startswith('Effect_'):
        color_map[trace.name] = trace.line.color
        trace.name = trace.name.replace('Effect_', 'Actual Effect ')
for trace in fig.data:
    if trace.name.startswith('Linear_'):
        effect_name = 'Effect_' + trace.name.split('_')[1]
        trace.line.color = color_map[effect_name]
        trace.line.dash = 'dot'
        trace.opacity = 0.4  # Make the line lighter
        trace.name = trace.name.replace('Linear_', 'Linear Interp ')
fig.show()

# %%