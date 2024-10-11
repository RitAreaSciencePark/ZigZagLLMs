import sys
import torch


def get_target_layers(model, n_layer, option="norm1", every=1):
    map_names = dict(
        norm1=".input_layernorm",
        norm2=".post_attention_layernorm",
        res2="",
    )
    suffix = map_names[option]
    names = [name for name, _ in model.named_modules()]

    target_layers = {i: f"model.layers.{i}{suffix}" for i in range(0, n_layer, every)}
    # target_layers[-1] = "model.embed_tokens"
    target_layers[n_layer] = "model.norm"
    # target_layers[n_layer + 1] = "lm_head"

    for target_layer in target_layers.values():
        assert target_layer in names, (target_layer, names)

    return target_layers

def get_target_layers_pythia(model, n_layer, option="norm1", every=1):
    map_names = dict(
        norm1=".input_layernorm",
        norm2=".post_attention_layernorm",
        res2="",
    )
    suffix = map_names[option]
    names = [name for name, _ in model.named_modules()]

    target_layers = {i: f"gpt_neox.layers.{i}{suffix}" for i in range(0, n_layer, every)}
    target_layers[n_layer] = "gpt_neox.final_layer_norm"


    for target_layer in target_layers.values():
        assert target_layer in names, (target_layer, names)

    return target_layers



def get_target_layers_vit(model, n_layer, option="norm1", every=1):
    map_names = dict(
        norm1=".ln_1",
        norm2=".ln_2",
        res2="",
    )
    suffix = map_names[option]
    names = [name for name, _ in model.named_modules()]

    target_layers = {
        i: f"encoder.layers.encoder_layer_{i}{suffix}" for i in range(0, n_layer, every)
    }
    # target_layers[-1] = "model.embed_tokens"
    target_layers[n_layer] = "encoder.ln"
    # target_layers[n_layer + 1] = "heads.head"

    for target_layer in target_layers.values():
        assert target_layer in names, (target_layer, names)

    return target_layers


def get_target_layers_deberta(model, n_layer, option="norm2", every=1):
    map_names = dict(
        norm1=".ln_1",
        norm2=".output.LayerNorm",
        res2="",
    )
    suffix = map_names[option]
    names = [name for name, _ in model.named_modules()]

    target_layers = {
        i + 1: f"encoder.layer.{i}{suffix}" for i in range(0, n_layer, every)
    }

    target_layers[0] = "embeddings.LayerNorm"
    # target_layers[n_layer] = "encoder.ln"
    # target_layers[n_layer + 1] = "heads.head"

    for target_layer in target_layers.values():
        assert target_layer in names, (target_layer, names)

    return target_layers


def get_target_layers_resnet(model, model_name):

    model_blocks = {"resnet50": [3, 4, 6, 3]}
    names = [name for name, _ in model.named_modules()]

    block_list = model_blocks[model_name]

    target_layers = {0: "maxpool"}
    count = 1
    for i in range(len(block_list)):
        for j in range(block_list[i]):
            target_layers[count] = f"layer{i+1}.{j}"
            count += 1

    target_layers[count] = "avgpool"

    for target_layer in target_layers.values():
        assert target_layer in names, (target_layer, names)

    return target_layers


def print_memory_consumed(rank=None):
    torch.cuda.empty_cache()
    allocated = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    if rank is not None and rank == 0:
        print(f"CUDA mem allocated: {allocated} GB")
        print(f"CUDA mem reserved: {reserved} GB")
    else:
        print(f"CUDA mem allocated: {allocated} GB")
        print(f"CUDA mem reserved: {reserved} GB")
    sys.stdout.flush()
