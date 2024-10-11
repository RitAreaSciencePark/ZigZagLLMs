import warnings
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
import sys
import torch
import logging


def get_model(
    config_name,
    model_name_or_path,
    precision,
    low_cpu_mem_usage,
    use_flash_attn,
    logger,
):
    if config_name:
        config = AutoConfig.from_pretrained(config_name)
    elif model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path)
    else:
        warnings.warn("Using a fake llama for debugging\n", stacklevel=2)
        config = LlamaConfig()
        config.intermediate_size = 1000
        config.num_hidden_layers = 3
        config.num_attention_heads = 2
        config.num_key_value_heads = 2
        config.hidden_size = 500

    if model_name_or_path:
        print("model_loading started. \n\n")
        print(config)
        sys.stdout.flush()
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=precision,
            use_flash_attention_2=True if use_flash_attn else False,
        )
        print("model loading finished. \n\n")
        sys.stdout.flush()
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    return model
