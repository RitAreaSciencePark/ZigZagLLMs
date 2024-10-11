import numpy as np
import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline,AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset,concatenate_datasets
from datetime import datetime
import sys
import os

from intrinsic_dimension.extract_activations import extract_activations
from utils.dataloader_utils import get_dataloader
from utils.dataset_utils import get_text_dataset
from utils.helpers import get_target_layers,get_target_layers_pythia
from intrinsic_dimension.compute_distances import get_embdims

from accelerate import load_checkpoint_and_dispatch
import argparse

def run(model_name,output_folder,dataset,access_token,layer_number):
    """
    Function that runs the entire representations extraction. 
    Representations are extracted from the post attention layernorm

    Args:
        model_name (string): model name on Hugging Face
        output_folder (string): output folder for the representations
        dataset (string): dataset from which the representations are extracted
        access_token (string): user token from Hugging Face
        layer_number (int): number of layers of the model
    """
    
    #Load models and tokenizer from HF
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=access_token) 
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",token=access_token)
    
    #Setup and Download SST or Pile-10K
    text_field = ''
    raw_dataset = []
    if dataset == 'sst':
        raw_dataset_tmp = load_dataset("sst","default")
        raw_dataset = concatenate_datasets([raw_dataset_tmp['train'],\
                                            raw_dataset_tmp['validation'],\
                                            raw_dataset_tmp['test']]) 
        text_field='sentence'
    else:
        raw_dataset_tmp = load_dataset("NeelNanda/pile-10k",trust_remote_code=True)
        raw_dataset = raw_dataset_tmp['train']
        text_field='text'

    
    dataset = get_text_dataset(text_field=text_field, tokenizer=tokenizer, raw_dataset=raw_dataset)
    nsamples = dataset.num_rows
    dataloader = get_dataloader(dataset=dataset,pad_token_id=2,batch_size=1)
    
    #Get target layers 
    target_layers = []
    if "Llama" in model_name or "Mistral" in model_name:
        target_layers = get_target_layers(model,layer_number,option="norm2")
        target_layers = [x for x in target_layers.values()]
    elif "pythia" in model_name:
        target_layers = get_target_layers_pythia(model,layer_number,option="norm2")
        target_layers = [x for x in target_layers.values()]
    
    embdim, dtypes = get_embdims(model,dataloader,target_layers)
    name = ''
    if "Llama" in model_name:
        name = "llama"
    elif "Mistral" in model_name:
        name = "mistral"
    elif "pythia" in model_name:
        name = "pythia"

    EA = extract_activations(model,model_name=name,dataloader=dataloader,target_layers=target_layers, embdim=embdim,nsamples=nsamples,dtypes=dtypes,use_last_token=True)
    EA.extract(dataloader)
    
    
    if not os.path.isdir(output_folder): 
        os.makedirs(output_folder) 

    for name in target_layers:
        np.save(f"{output_folder}/rep-{name}.npy",EA.hidden_states[name])

    if "pythia" in model_name:
        os.rename(output_folder+os.sep+"rep-gpt_neox.final_layer_norm.npy",output_folder+os.sep+"rep-gpt_neox.norm.npy")

if __name__ == "__main__":

    model_list = {'meta-llama/Llama-2-7b-hf':32,
                  'meta-llama/Llama-2-13b-hf':40,
                  'meta-llama/Llama-2-70b-hf':80,
                  'meta-llama/Meta-Llama-3-8B':32,
                  'meta-llama/Meta-Llama-3-70B':80,
                  'EleutherAI/pythia-6.9b-deduped':32,
                  'mistralai/Mistral-7B-v0.1':32}

    parser = argparse.ArgumentParser(description='Process inputs for the representation extraction')

    parser.add_argument('--model', type=str, help='input model, e.g. "meta-llama/Meta-Llama-3-8B"')
    parser.add_argument('--output_folder',type=str,help='output fold of representation')
    parser.add_argument('--dataset',type=str,help='dataset can be "pile" or "sst" ')
    parser.add_argument('--access_token',type=str,help='personal access token for hugging face')

    args = parser.parse_args()
    print(args.dataset)
    if args.dataset != "pile" and args.dataset != "sst":
        raise ValueError(f"Invalid value: {args.dataset}, dataset parameter has to be equal to pile or sst")
        exit()
    if args.model not in list(model_list.keys()):
        raise ValueError(f"Invalid value: {args.model}. The model has to be picked from {(model_list.keys())}")
    run(model_name=args.model,
        output_folder=args.output_folder,
        dataset=args.dataset,
        access_token=args.access_token,
        layer_number=model_list[args.model])
    # check dataset string that has to be pile or sst.


   
