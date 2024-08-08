import os
import sys
sys.path.append(os.path.abspath('..'))
import signal
import resource
from ft_effects.utils import gen, get_feature_acts, get_scale

from transformer_lens import utils as tutils

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from transformer_lens import HookedTransformer
from steering.sae import JumpReLUSAE
from huggingface_hub import hf_hub_download
import numpy as np
from queue import Empty

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    destroy_process_group()

@torch.no_grad()
def load_model_and_sae(rank):
    device = torch.device(f"cuda:{rank}")
    model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
    # hp = "blocks.12.hook_resid_post"
    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-res",
        filename="layer_12/width_16k/average_l0_82/params.npz",
        force_download=False)
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}
    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)
    sae.to(device)
    # prep data
    data = tutils.load_dataset("NeelNanda/c4-code-20k", split="train")
    tokenized_data = tutils.tokenize_and_concatenate(data, model.tokenizer, max_length=32)
    tokenized_data = tokenized_data.shuffle(42)
    loader = DataLoader(tokenized_data, batch_size=64)
    return model, sae, loader


@torch.no_grad()
def worker(rank, world_size, task_queue, result_queue, features, scale=None):
    setup(rank, world_size)
    model, sae, loader = load_model_and_sae(rank)

    baseline_samples = gen(model=model, n_batches=10)
    baseline_dist = get_feature_acts(model=model, sae=sae, tokens=baseline_samples, batch_size=64)
    
    while True:
        try:
            feature_index = task_queue.get(timeout=1)  # 1 second timeout
        except Empty:
            break
        
        feature = features[feature_index]
        if scale is None:
            opt_scale = get_scale(model=model,
                                  steer=feature.to(sae.W_dec.device),
                                  loader=loader,
                                  scales=list(range(0, 220, 20)),
                                  )
        else:
            opt_scale = scale
        used_feature = feature * opt_scale
        used_feature = used_feature.to(sae.W_dec.device)

        ft_samples = gen(model=model, steer=used_feature)
        ft_dist = get_feature_acts(model=model, sae=sae, tokens=ft_samples, batch_size=64)
        diff = ft_dist - baseline_dist
        result_queue.put((diff.cpu(), used_feature, feature_index))
    cleanup()

@torch.no_grad()
def main(features, save_dir):
    world_size = torch.cuda.device_count()
    num_features = features.shape[0]
    
    mp.set_start_method('spawn')
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Populate task queue
    for i in range(num_features):
        task_queue.put(i)
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, task_queue, result_queue, features.detach()))
        p.start()
        processes.append(p)
    
    # Initialize result containers
    all_effects = [None] * num_features
    all_used_features = [None] * num_features
    
    # Collect results as they become available
    try:
        completed_features = 0
        while completed_features < num_features:
            effect, used_feature, index = result_queue.get()
            all_effects[index] = effect
            all_used_features[index] = used_feature
            completed_features += 1
            print(f"Completed feature {index+1}/{num_features}")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Saving partial results...")
    except Empty:
        print("\nAll tasks completed.")
    finally:
        # Terminate all processes
        for p in processes:
            p.terminate()
        
        # Wait for all processes to finish
        for p in processes:
            p.join(timeout=5)  # Wait up to 5 seconds for each process to join

        # Force kill any processes that didn't join in time
        for p in processes:
            if p.is_alive():
                print(f"Force terminating process {p.pid}")
                os.kill(p.pid, signal.SIGKILL)

    # Filter out None values (uncompleted features)
    all_effects = [e.cpu() for e in all_effects if e is not None]
    all_used_features = [f.cpu() for f in all_used_features if f is not None]
    
    # Convert lists to tensors
    all_effects = torch.stack(all_effects)
    all_used_features = torch.stack(all_used_features)
    
    print("Saving results...")
    # Save results
    torch.save(all_effects, os.path.join(save_dir, "all_effects.pt"))
    torch.save(all_used_features, os.path.join(save_dir, "used_features.pt"))

if __name__ == "__main__":
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-2b-pt-res",
    filename="layer_12/width_16k/average_l0_82/params.npz",
    force_download=False)
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)
    sae._requires_grad = False

    save_dir = "effects/G2_2B_L12/multi_16k_from_0"
    os.makedirs(save_dir)
    main(sae.W_dec, save_dir)


