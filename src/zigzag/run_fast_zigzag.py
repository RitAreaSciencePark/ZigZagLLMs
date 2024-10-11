import zigzag_DL as zz
from pyfzz import pyfzz
import csv
from datetime import datetime
import numpy as np
from natsort import natsorted
import tqdm
from tqdm import tqdm
import os
import argparse

def iord(c):
    if c == 'i':
        return +1
    else: 
        return -1
def flattening_for_zigzag_ver(times,f):
    '''
    Input: times - List of lists
    f    : weird format that comes from dionysus
    Data structure developed before giving in output the data stracture for pyfzz
    {'Layer_number':[('i/d',simplex_list),...]

    Output data structure [('i/d',simplex), ... , ('i/d',simplex)]
    '''
        
    layers_summary = []
    for i in range(len(times)):
        simplex = list(np.array(f[i]))
        '''
        single elements of times are list where the EVEN positions are deletions 'd' of the n-th simplex 
        and ODD positions are the insertions 'i' of the n-th simplex
        '''
        t = times[i] 
        for position in range(len(t)):
            if position % 2 == 1:  # deletion
                layers_summary.append((t[position],'d',simplex))
            else: #insertion
                layers_summary.append((t[position],'i',simplex))
    
    layers_summary.sort(key=lambda x:(x[0], len(x[2])*iord(x[1])),reverse=False)

    res = []
    layer_res = []
    for element in layers_summary:
        res.append((element[1],element[2]))
        layer_res.append(element[0])
    

    return layer_res,res


def run(reps_path,knn,dim,output_folder,output_file):
    """Run the FastZigZag on a set of reps by generating a KNN graph
    up to a certain level

    Args:
        reps_path (string): path of the hidden representations
        knn (integer): K parameter for the Knn graph
        dim (integer): dimension for zigzag
        output_folder (string): output folder path 
        output_file (string): output filename
    """
    params = {"knn":knn,
              "dim":dim,
              "dataset":"text",
              "ifiles":reps_path+os.sep+"rep*",
              "outdir":output_folder+os.sep+output_file}
    
    zzio=zz.IO(input_parameters=params)
    reps = zzio.read_reps()  #ORIGINAL CODE
    
    params = zzio.read_params()
    zigclass=zz.ZIGZAG(params,reps=reps)

    
    simplices = simplices_padded = [] 
    """
    if params["labels"] != "":    
        print("with labels: ",params["labels"])
        labels = np.load(params["labels"])

        simplices,simplices_padded = zigclass.generate_simplex_tree(reps = reps,labels=labels)
    else:
        print("without labels")
        simplices,simplices_padded = zigclass.generate_simplex_tree(reps = reps)
    """
    
    simplices,simplices_padded = zigclass.generate_simplex_tree(reps = reps)
    layers = zigclass.compute_layers_with_intersection(simplices_padded)
    f,times = zigclass.compute_filtration_times(simplices,layers)

    
    layers_res,result = flattening_for_zigzag_ver(times,f)


    fzz = pyfzz()
 
    bars = fzz.compute_zigzag(result)

    merged_output = []
    for i in bars:
        # we do not merge points that are born and died in the same layer
        if layers_res[i[0]-1] != layers_res[i[1]]:
            merged_output.append((i[2],layers_res[i[0]-1],layers_res[i[1]]))
    
    with open(output_folder+os.sep+output_file,"w") as out_file:
        csv_out=csv.writer(out_file)
        csv_out.writerow(["dimension","birth","death"])
        for row in merged_output:
            csv_out.writerow(row)


if __name__ == '__main__':

    print("Running ZigZag ...")
    parser = argparse.ArgumentParser(description='Process inputs for the representation extraction')

    parser.add_argument('--reps_path', type=str, help='Path where the representations are present')
    parser.add_argument('--knn',type=int,help='K for the KNN Graph that will be generated')
    parser.add_argument('--dim',type=int,help='Dimension for ZigZag')
    parser.add_argument('--output_folder',type=str,help='Output folder for the results')
    parser.add_argument('--output_file',type=str,help='Output file name, the file extension used should be .csv')

    args = parser.parse_args()
    

    
    
    if not os.path.isdir(args.output_folder): 
        os.makedirs(args.output_folder) 

    run(reps_path=args.reps_path,
        knn=args.knn,
        dim=args.dim,
        output_folder=args.output_folder,
        output_file=args.output_file)