from transformers import AutoTokenizer, LlamaTokenizer


def get_tokenizer(tokenizer_path=None, model_path=None):

    assert tokenizer_path is not None or model_path is not None
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False
    )  # check what happens when true

    num_added_tokens = tokenizer.add_special_tokens(
        {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }
    )
    assert num_added_tokens in [
        0,
        1,
    ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # resize_model_embeddings(model, tokenizer)
    return tokenizer
