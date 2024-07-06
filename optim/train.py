import os
import sys
import json
sys.path.append(os.path.abspath('..'))

from functools import partial

import torch
import torch.nn.functional as F
import wandb

from torch.utils.data import Dataset, DataLoader
from steering.patch import patch_resid
from steering.utils import normalise_decoder

from transformer_lens import HookedTransformer
from sae_lens import SAE


class Comparisons(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # Load the tensors
        self.wins = torch.load(os.path.join(data_dir, 'wins.pt'))
        self.losses = torch.load(os.path.join(data_dir, 'losses.pt'))
        
        # Load the list of vector indices from JSON
        with open(os.path.join(data_dir, 'vector_idxs.json'), 'r') as f:
            self.vector_idxs = json.load(f)

    def __len__(self):
        return self.wins.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            'winner': self.wins[idx],
            'loser': self.losses[idx],
            'vector_idx': self.vector_idxs[idx]
        }

class Adapter(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Adapter, self).__init__()
        self.W_in = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(input_size, hidden_size)))
        self.W_out = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(hidden_size, input_size)))
        self.b_mid = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_out = torch.nn.Parameter(torch.zeros(input_size))

    def forward(self, x):
        # takes in a batch of steering vectors
        # x shape is (batch_size, d_model)
        x = x @ self.W_in + self.b_mid
        x = F.relu(x)
        x = x @ self.W_out + self.b_out
        return x


def dpo_loss(pi_winner_logprobs, pi_loser_logprobs, ref_winner_logprobs, ref_loser_logprobs, beta=0.1):
        # Compute log ratios
        pi_logratios = pi_winner_logprobs - pi_loser_logprobs
        ref_logratios = ref_winner_logprobs - ref_loser_logprobs

        losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
        return losses


def train(model, steering_vectors, adapter, dataloader, device, cfg):
    data_iter = iter(dataloader)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(adapter.parameters(), lr=cfg['lr'])

    loss_sum = 0
    for step in range(cfg['total_steps']):
        try:
            data = next(data_iter)
        except StopIteration:
            # Reset the dataloader when it runs out
            data_iter = iter(dataloader)
            data = next(data_iter)

        winners = data['winner']
        losers = data['loser']
        vector_idxs = data['vector_idx']

        with torch.no_grad():
            selected_vectors = steering_vectors[vector_idxs]
            selected_vectors = selected_vectors.to(device)

            ref_winner_logits = model(winners, return_type='logits')
            ref_loser_logits = model(losers, return_type='logits')
            # Convert logits to log probabilities and sum over sequence length
            ref_winner_logprobs = F.log_softmax(ref_winner_logits, dim=-1).sum(dim=1)
            ref_loser_logprobs = F.log_softmax(ref_loser_logits, dim=-1).sum(dim=1)

        # adapt
        adapted_vectors = adapter(selected_vectors)
        adapted_vectors = adapted_vectors[:, None, :]
        adapted_vectors = adapted_vectors.to(torch.float16)

        with model.hooks(fwd_hooks=[(hp6, partial(patch_resid, steering=adapted_vectors))]):
            policy_winner_logits = model(winners, return_type='logits')
            policy_loser_logits = model(losers, return_type='logits')
        
        # Convert logits to log probabilities and sum over sequence length
        policy_winner_logprobs = F.log_softmax(policy_winner_logits, dim=-1).sum(dim=1)
        policy_loser_logprobs = F.log_softmax(policy_loser_logits, dim=-1).sum(dim=1)

        losses = dpo_loss(
            policy_winner_logprobs,
            policy_loser_logprobs,
            ref_winner_logprobs,
            ref_loser_logprobs,
            beta=cfg['beta'],
        )

        loss = losses.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_sum += loss.item()

        if step % 10 == 0:
            # Log metrics to wandb
            wandb.log({
                "step_loss": loss.item(),
                "step": step,
            })
            print(f'step {step}, Loss: {loss.item()}')

        if (step + 1) % 100 == 0:
            avg_loss = loss_sum / 100
            print(f'Step {step + 1}, Loss: {loss.item():.4f}, Avg Loss (last 100): {avg_loss:.4f}')
            wandb.log({
                "avg_loss_last_100": avg_loss,
                "step": step,
            })
            loss_sum = 0  # Reset the sum after logging

    wandb.finish()




if __name__ == '__main__':
    train_cfg = {
        'batch_size': 4,
        'total_steps': int(1e4),
        'lr': 1e-3,
        'beta': 0.1,
        'model': 'gemma-2b',
        'base_scale': 50,
        'adapter_hidden_size': 2,
    }
    dataset = Comparisons('comparison_data')
    dataloader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(train_cfg['model'], device=device)

    hp6 = "blocks.6.hook_resid_post"
    sae6, _, _ = SAE.from_pretrained(
        release = "gemma-2b-res-jb", # see other options in sae_lens/pretrained_saes.yaml
        sae_id = hp6, # won't always be a hook point
        device = 'cpu')
    sae6 = sae6.to(device)
    normalise_decoder(sae6)

    with open('comparison_data/steering_vectors.json', 'r') as f:
        vector_info = json.load(f)
    
    steering_vectors = [sae6.W_dec[d['ft_id']].clone() for d in vector_info]

    steering_vectors = torch.stack(steering_vectors)
    steering_vectors = steering_vectors.to(device)

    steering_vectors = steering_vectors * train_cfg['base_scale']

    adapter = Adapter(steering_vectors.shape[1],
                      hidden_size=train_cfg['adapter_hidden_size'])
    adapter.to(device)

    # Initialize wandb
    wandb.init(project="steering-adapter", config=train_cfg)

    model.to(torch.float16)
    train(model, steering_vectors, adapter, dataloader, device, train_cfg)


