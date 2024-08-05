# %%
import os
import torch
import plotly.express as px
import pandas as pd
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