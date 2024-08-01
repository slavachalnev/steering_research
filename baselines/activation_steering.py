# %%
import os
import sys
sys.path.append(os.path.abspath('..'))
from steering.patch import patch_resid
import torch
from transformer_lens import HookedTransformer
from functools import partial
import json
import plotly.express as px

# %%

def load_data(dir_path):
    data_path = os.path.join(dir_path, "act_steer.json")
    with open(data_path, 'r') as f:
        data = json.load(f)
    pos_examples = data['pos']
    neg_examples = data['neg']
    return pos_examples, neg_examples

def get_acts(examples, model, device):
    total_acts = torch.zeros(model.cfg.n_layers, model.cfg.d_model, device=device)
    for i, pos in enumerate(examples):
        tokens = model.tokenizer.apply_chat_template(pos, return_tensors='pt')
        tokens = tokens[:, :-2] # remove the last two tokens <end_of_turn> and newline.
        points = [f"blocks.{n}.hook_resid_post" for n in range(model.cfg.n_layers)]
        logits, cache = model.run_with_cache(tokens, names_filter=points)
        acts = [cache[p][0].mean(0) for p in points]
        acts = torch.stack(acts, dim=0)
        total_acts += acts
        return total_acts / len(examples)

def steer_model(model, steer, layer, text="How do you feel?", scale=5):
    toks = model.tokenizer.apply_chat_template([{"role": "user", "content": text}], return_tensors='pt', add_generation_prompt=True)
    toks = toks.expand(10, -1)

    hp = f"blocks.{layer}.hook_resid_post"
    with model.hooks([(hp, partial(patch_resid, steering=steer[layer], scale=scale))]):
        gen_toks = model.generate(toks, max_new_tokens=40, use_past_kv_cache=True)
    return model.to_string(gen_toks)

# def evaluate_layers(model, steer, pos_examples, neg_examples, scale=5):
#     losses = []
#     for layer in range(model.cfg.n_layers):
#         loss_total = 0
#         hp = f"blocks.{layer}.hook_resid_post"
#         for pos in pos_examples:
#             tokens = model.tokenizer.apply_chat_template(pos, return_tensors='pt')
#             tokens = tokens[:, :-2]
#             with model.hooks([(hp, partial(patch_resid, steering=steer[layer], scale=scale))]):
#                 loss = model.forward(tokens, return_type="loss")
#             loss_total -= loss.item()
#         for neg in neg_examples:
#             tokens = model.tokenizer.apply_chat_template(neg, return_tensors='pt')
#             tokens = tokens[:, :-2]
#             with model.hooks([(hp, partial(patch_resid, steering=steer[layer], scale=scale))]):
#                 loss = model.forward(tokens, return_type="loss")
#             loss_total += loss.item()
#         losses.append(loss_total / len(pos_examples))
#     return losses


# %%
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: HookedTransformer = HookedTransformer.from_pretrained("google/gemma-2-9b-it", device=device, dtype=torch.float16)

# %%
if __name__ == "__main__":
    pos_examples, neg_examples = load_data("steer_cfgs/gemma-9b-it/golden_gate")
    # pos_examples, neg_examples = load_data("steer_cfgs/gemma-9b-it/anti_liberal")
    print(pos_examples)
    print(neg_examples)
    pos_acts = get_acts(pos_examples, model, device)
    neg_acts = get_acts(neg_examples, model, device)
    steer = pos_acts - neg_acts

# %%
if __name__ == "__main__":
    gen_texts = steer_model(model, steer, layer=20, text="What is your name?", scale=15) 
    for t in gen_texts:
        print(t)
        print("\n**********\n")

# %%
