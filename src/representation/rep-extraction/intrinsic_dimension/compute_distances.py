import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from intrinsic_dimension.pairwise_distances import compute_distances
from intrinsic_dimension.extract_activations import extract_activations
from transformers import PreTrainedModel
import sys


def get_embdims(model, dataloader, target_layers):
    embdims = defaultdict(lambda: None)
    dtypes = defaultdict(lambda: None)

    def get_hook(name, embdims):
        def hook_fn(module, input, output):
            embdims[name] = output.shape[-1]
            dtypes[name] = output.dtype
            #print(f"{name}-outputshape: {output.shape[-1]}")

        return hook_fn

    handles = {}
    print(f"target_layers: {target_layers}")
    for name, module in model.named_modules():
        #print(f"name: {name}")
        if name in target_layers:
            handles[name] = module.register_forward_hook(get_hook(name, embdims))

    batch = next(iter(dataloader))
    _ = model(batch["input_ids"].to("cuda"))

    for name, module in model.named_modules():
        if name in target_layers:
            handles[name].remove()
    print(f"embdims: {len(embdims)} - target_layers: {len(target_layers)}")
    assert len(embdims) == len(target_layers)
    return embdims, dtypes


@torch.inference_mode()
def compute_id(
    model: PreTrainedModel,
    model_name: str,
    dataloader: DataLoader,
    target_layers: dict,
    nsamples,
    maxk=50,
    range_scaling=1050,
    dirpath=".",
    filename="",
    use_last_token=False,
    remove_duplicates=True,
    save_distances=True,
    print_every=100,
):
    dirpath = str(dirpath).lower()
    os.makedirs(dirpath, exist_ok=True)

    if use_last_token:
        filename = f"_{filename}_target"
    else:
        filename = f"_{filename}_mean"

    target_layer_names = list(target_layers.values())
    target_layer_labels = list(target_layers.keys())

    model = model.eval()
    start = time.time()

    print("layer_to_extract: ", target_layer_labels)
    embdims, dtypes = get_embdims(model, dataloader, target_layer_names)

    extr_act = extract_activations(
        model,
        model_name,
        dataloader,
        target_layer_names,
        embdims,
        dtypes,
        nsamples,
        use_last_token=use_last_token,
        print_every=print_every,
    )

    extr_act.extract(dataloader)
    print(f"num_tokens: {extr_act.tot_tokens/10**3}k")
    print((time.time() - start) / 3600, "hours")

    act_dict = extr_act.hidden_states

    for i, (layer, act) in enumerate(act_dict.items()):
        act = act.to(torch.float64).numpy()

        save_backward_indices = False
        if remove_duplicates:
            act, idx, inverse = np.unique(
                act, axis=0, return_index=True, return_inverse=True
            )
            print(len(idx), len(inverse))
            if len(idx) == len(inverse):
                # if no overlapping data has been found return the original ordred array
                assert len(np.unique(inverse)) == len(inverse)
                act = act[inverse]
            else:
                save_backward_indices = True
            print(f"num_duplicates = {len(inverse)-len(idx)}")

        print(f"{layer} act_shape {act.shape}")
        sys.stdout.flush()

        n_samples = act.shape[0]
        range_scaling = min(range_scaling, n_samples - 1)
        maxk = min(maxk, n_samples - 1)

        start = time.time()
        distances, dist_index, mus, _ = compute_distances(
            X=act,
            n_neighbors=maxk + 1,
            n_jobs=1,
            working_memory=2048,
            range_scaling=range_scaling,
            argsort=False,
        )
        print((time.time() - start) / 60, "min")

        if save_distances:
            np.save(f"{dirpath}/l{target_layer_labels[i]}{filename}_dist", distances)
            np.save(f"{dirpath}/l{target_layer_labels[i]}{filename}_index", dist_index)
        if save_backward_indices:
            np.save(f"{dirpath}/l{target_layer_labels[i]}{filename}_inverse", inverse)
        np.save(f"{dirpath}/l{target_layer_labels[i]}{filename}_mus", mus)
