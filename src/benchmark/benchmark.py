import argparse
import os
from lm_eval.models.huggingface import HFLM
from transformers import LlamaForCausalLM,AutoTokenizer
import post_process
import lm_eval
from post_process import POSTPROCESS
import json
import numpy as np
import torch

def check_dir(path, model_name):

    folder = path + os.sep + model_name

    if os.path.isdir(folder):
        print("Exists")
    else:
        print("Folder created")
        os.makedirs(folder,exist_ok=True)

    return folder

def similarity_persistance(betti,ph_sim_betti):
    n_layers = len(betti)
    assert n_layers == int(ph_sim_betti.shape[0])
    tmp = np.zeros((n_layers,n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            tmp[i][j] = ph_sim_betti[i,j]/betti[j]
    
    res = np.mean(tmp,axis=1)
    return res


def layers_to_cut_given_r(r,p=0.9,layer_norm_added=True):
    '''
    r = result of the function p_column_knn
    p = value of r above which we cut layers
    layer_norm_added = True if we consider also the output_layer_norm 
    '''
    res = []
    maximum = max(r)
    for i in range(len(r)-1*layer_norm_added):
        if r[i] >= p*maximum:
            res.append(i)
    return res

def full_benchmark(output_path,model_name,task,access_token):

    model = HFLM(pretrained=model_name,\
                 tokenizer=model_name,\
                 parallelize=True,\
                 device_map_option="auto", token=access_token)
    
    results = lm_eval.simple_evaluate(model=model,\
                                    tasks= task,\
                                    device="auto",\
                                    batch_size="auto",\
                                    num_fewshot=5)
    
    out_file = output_path + os.sep + model_name.split('/')[1] +"_"+task[0] + "full_bmk.json"
    with open(out_file,'w',encoding='utf-8') as f:
            json.dump(results['results'],f,ensure_ascii=False,indent=4)

    return



def load_and_cut(layers,model,model_name):
    """Cut the layers of the given model

    Args:
        layers (list): 
        model (pytorch model): model from which the layers will be cut
        model_name (str): name of the model
    Returns:
        model: model with reduced layers
    """
    new_layers = torch.nn.ModuleList()

    remove_layers = layers
    count = 0
    len_layers = 0
    if 'pythia' in model_name:
        len_layers = len(model._model.gpt_neox.layers)
    else:
        len_layers = len(model._model.model.layers)
    for i in range(0, len_layers):
        if i not in remove_layers:
            layer=''
            if 'pythia' in model_name:
                layer = model._model.gpt_neox.layers[i]
                layer.attention.layer_idx=count
            else: 
                layer = model._model.model.layers[i]
                layer.self_attn.layer_idx=count
            layer.layer_idx = count

            
            new_layers.append(layer)
            count += 1
            
    if 'pythia' in model_name:
        model._model.gpt_neox.layers = new_layers
    else:    
        model._model.model.layers = new_layers
    changed_num_hidden_layers = len_layers - len(remove_layers)
    model.config.num_hidden_layers = changed_num_hidden_layers


    return model  

def benchmark_with_cuts(output_path,model_name,task,zigzag_output,zigzag_layers,access_token):
    """Run the benchmark with the persistence similiarity method

    Args:
        output_path (string): Path of the folder where to store the output
        model_name (string): String with the model name on Hugging Face
        task (list): list with strings of the type ['<benchmark>']
        zigzag_output (string): Path to the output of the zigzag
        zigzag_layers (int): Number of layers with intersections layers
        access_token (string): Hugging Face token
    """
    pds = post_process.read_pd_from_csv(zigzag_output, max_dim = 5)
    pp = POSTPROCESS(pds = pds, num_layers = zigzag_layers, start_ind = 1, zigzag = True, debug = False)

    #find betti number and phsim

    betties = pp.find_betti_layers()    
    b1 = betties[1]
    psim_mats = pp.find_ph_sim()
    psim_1 = psim_mats[1]


    for p in [0.8,0.9]:
            
        r = similarity_persistance(b1,psim_1)
        list_layers = layers_to_cut_given_r(r,p=0.9)
        
        print(f"BMK with {p} as threshold")

        r = similarity_persistance(b1,psim_1)
        list_layers = layers_to_cut_given_r(r,p=p)

        my_model = HFLM(pretrained=model_name,\
                        tokenizer=model_name,\
                        parallelize=True,\
                        device_map_option="auto",token=access_token)

        res_model = load_and_cut(layers = list_layers,model=my_model,model_name=model_name)
        results = lm_eval.simple_evaluate(model=res_model,\
                                          tasks= task,\
                                          device="auto",\
                                          num_fewshot=5,\
                                          batch_size="auto")
            
        out_file = output_path + os.sep + model_name.split('/')[1] +"_"+task[0] + "_"+str(p)+"_bmk.json"
        with open(out_file,'w',encoding='utf-8') as f:
            json.dump(results['results'],f,ensure_ascii=False,indent=4)

    return

def benchmark_with_block_cuts(output_path,model_name,task,access_token,begin_block=0,end_block=0):
    """Run the benchmark with a given block of layers to cut

    Args:
        output_path (string): Path of the folder where to store the output
        model_name (string): String with the model name on Hugging Face
        task (list):  list with strings of the type ['<benchmark>']
        access_token (string): User Hugging Face token
        begin_block (int): First layer to cut of the block. Defaults to 0.
        end_block (int): Last layer to cut of the block. Defaults to 0.
    """
            

    print(f"About to cut: [{begin_block},{end_block}]")
    my_model = HFLM(pretrained=model_name,\
                    tokenizer=model_name,\
                    parallelize=True,\
                    device_map_option="auto",token=access_token)

    list_layers = [i for i in range(int(begin_block),int(end_block)+1)]  

    print("Block of layers being cut: ",list_layers)
    res_model = load_and_cut(layers = list_layers,model=my_model,model_name=model_name)
    results = lm_eval.simple_evaluate(model=res_model,\
                                    tasks= task,\
                                    device="auto",\
                                    batch_size="auto",\
                                    num_fewshot=5)


    bmk_path = output_path + os.sep + model_name.split('/')[1]+'_'+task[0] +'_'+ str(begin_block) + '_' + str(end_block)+'.json'
    with open(bmk_path,'w',encoding='utf-8') as f:
        json.dump(results['results'],f,ensure_ascii=False,indent=4)
    del res_model


    return

if __name__ == "__main__":

    #Model_list = {'<model>': number_of_layers+output}
    #Note that in this case the layers are N_layers+1 because in our case the representations
    #are are taken with also the output_layernorm
    model_list = {'meta-llama/Llama-2-7b-hf':33,
                  'meta-llama/Llama-2-13b-hf':41,
                  'meta-llama/Llama-2-70b-hf':81,
                  'meta-llama/Meta-Llama-3-8B':33,
                  'meta-llama/Meta-Llama-3-70B':81,
                  'EleutherAI/pythia-6.9b-deduped':33,
                  'mistralai/Mistral-7B-v0.1':33}
    
    parser = argparse.ArgumentParser(description='Run and reproduce benchmarks')
    parser.add_argument('--full',  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--persistence',  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--personalized',  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--output_folder', help='path where the output will be saved')
    parser.add_argument('--model_name',help="name of the model run")
    parser.add_argument('--zigzag_output',help='folder of the output file of zigzag')
    parser.add_argument('--task',help="benchmark to do")
    parser.add_argument('--start_block',help="Beginning of the block of layers to cut",type=int)
    parser.add_argument('--end_block', help="End of the block of layers to cut",type=int)
    parser.add_argument('--token', help="Your Hugging Face token",type=str)
    args = parser.parse_args()

    #TODO: check on consistency of parameters
    #assert args.model_path != None and args.output_path != None and args.zigzag_layers != None and args.model_name != None and args.betti_path != None and args.tasks != None


    final_out_path=check_dir(path=args.output_folder, model_name=args.model_name)
    final_tasks = [args.task]    

    if args.full:
        print(f"RUNNING FULL BENCHMARK {args.full}-{type(args.full)}")
        full_benchmark(model_name = args.model_name,
                        output_path=final_out_path,
                        task=final_tasks,
                        access_token=args.token)
        exit()
    if args.persistence:
        benchmark_with_cuts(output_path=final_out_path,\
                        model_name=args.model_name,\
                        task=final_tasks,\
                        zigzag_output=args.zigzag_output,\
                        zigzag_layers= model_list[args.model_name]*2,\
                        access_token=args.token)
        exit()
    if args.personalized:
        benchmark_with_block_cuts(output_path = final_out_path,
                                  model_name=args.model_name,
                                  task=final_tasks,
                                  begin_block=args.start_block,
                                  end_block=args.end_block,
                                  access_token=args.token)
