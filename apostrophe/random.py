# %%
import os
import sys
sys.path.append(os.path.abspath('..'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens import utils as tutils
from transformer_lens.evals import make_pile_data_loader, evaluate_on_dataset

from functools import partial
from datasets import load_dataset
from tqdm import tqdm
import json

import einops

from sae_lens import SAE
# from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
# from sae_lens import SparseAutoencoder, ActivationsStore

from steering.evals_utils import evaluate_completions, multi_criterion_evaluation
from steering.utils import normalise_decoder, text_to_sae_feats, top_activations
from steering.patch import generate, scores_2d

# from sae_vis.data_config_classes import SaeVisConfig
# from sae_vis.data_storing_fns import SaeVisData

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


import numpy as np
from sklearn.linear_model import LinearRegression

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("gemma-2b", device=device)

hp6 = "blocks.6.hook_resid_post"
sae6, _, _ = SAE.from_pretrained(
    release = "gemma-2b-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = hp6, # won't always be a hook point
    device = 'cpu',
)
sae6 = sae6.to(device)
normalise_decoder(sae6)
sae6.requires_grad_(False)

# %%

sae6.W_dec.shape # (n_fts, d_model)

# %%
# random vectors
rs = torch.randn(10000, model.cfg.d_model).to(device)
rs = rs / rs.norm(dim=-1, keepdim=True)
# %%

def patch_resid(resid, hook, steering, scale=1):
    # resid[:, :, :] = resid[:, :, :] + steering * scale
    resid[:, -1, :] = resid[:, -1, :] + steering * scale

# %%

prompts = ["I think she won'"]
correct_ids = model.to_tokens(['t'], prepend_bos=False)

@torch.no_grad()
def many_ft_eval(ft_ids=None, scale=80, correct_ids=correct_ids, use_logits=False, vecs=None):
    res_probs = []
    assert len(prompts) == 1
    batch_size = 32
    n = len(ft_ids) if vecs is None else len(vecs)
    for i in tqdm(range(0, n, batch_size)):
        batch_in = model.to_tokens(prompts, prepend_bos=True)
        if vecs is None:
            batch_ft_ids = ft_ids[i:i+batch_size]
            st = sae6.W_dec[batch_ft_ids]
        else:
            st = torch.stack(vecs[i:i+batch_size])
            
        # st = st[:, None, :] ##
        # st = st.expand(-1, batch_in.shape[1], -1) ##

        batch_in = batch_in.expand(st.shape[0], -1)
        # with model.hooks([(hp6, partial(patch_final, steering=st, scale=scale))]):
        with model.hooks([(hp6, partial(patch_resid, steering=st, scale=scale))]):
            logits = model(batch_in)[:, -1, :]
            if use_logits:
                probs = logits
            else:
                probs = F.softmax(logits, dim=-1)
            probs = probs[torch.arange(probs.shape[0]), correct_ids.squeeze()]
            for p in probs:
                res_probs.append(p.item())
    return res_probs

# %%
evals = many_ft_eval(vecs=[r for r in rs], scale=80)
# %%
sorted_evals = sorted(enumerate(evals), key=lambda x: x[1], reverse=True)

# %%
df = pd.DataFrame({
    'index': range(len(sorted_evals)),
    'line1': [x[1] for x in sorted_evals],
    # 'line2': pss_sorted,
})
fig = px.line(df, x='index', y=['line1'], title='Prob sorted by p(t)')
fig.update_traces(name='p (t)', selector=dict(name='line1'))
fig.show()
# %%
logit_evals = many_ft_eval(vecs=[r for r in rs], scale=80, use_logits=True)

# %%

X = rs.cpu().numpy()
y = np.array(logit_evals)
print(X.shape, y.shape)

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print('split done')

linear = LinearRegression()
linear.fit(X_train, y_train)
print('fit done')

y_train_pred = linear.predict(X_train)
y_test_pred = linear.predict(X_test)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"Training R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Training Mean Squared Error: {train_mse:.4f}")
print(f"Test Mean Squared Error: {test_mse:.4f}")
print(f"Training Mean Absolute Error: {train_mae:.4f}")
print(f"Test Mean Absolute Error: {test_mae:.4f}")

fig = make_subplots(rows=1, cols=2, subplot_titles=("Training Set", "Test Set"))
fig.add_trace(
    go.Scatter(x=y_train, y=y_train_pred, mode='markers', name='Training Data', 
               marker=dict(color='blue', opacity=0.5)),
    row=1, col=1)
fig.add_trace(
    go.Scatter(x=y_test, y=y_test_pred, mode='markers', name='Test Data', 
               marker=dict(color='red', opacity=0.5)),
    row=1, col=2)

# Add perfect prediction lines
min_val = min(min(y_train.min(), y_test.min()), 0)  # Include 0 in the range
max_val = max(y_train.max(), y_test.max())
fig.add_trace(
    go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
               name='Perfect Prediction', line=dict(color='green', dash='dash')),
    row=1, col=1)
fig.add_trace(
    go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
               name='Perfect Prediction', line=dict(color='green', dash='dash'), showlegend=False),
    row=1, col=2)
fig.update_layout(
    title='Actual vs Predicted Values',
    xaxis_title='Actual Values',
    yaxis_title='Predicted Values',
    height=600,
    width=1000)
fig.show()


# %%
# %%
