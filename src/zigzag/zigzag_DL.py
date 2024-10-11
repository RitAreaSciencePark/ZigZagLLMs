import numpy as np
import glob
import sklearn
import json
import argparse
from sklearn.neighbors import *
import gudhi as gd
import dionysus as d
import matplotlib.pyplot as plt
from natsort import natsorted
import scipy

import utils.fclaux as utils
import tqdm
from tqdm import tqdm
class IO:
    '''
    This class organises input parameters, output and reading files.
    '''
    def __init__(self, config_file = None,input_parameters = None, verbose = False, scriptmode = False):        

        self.verbose = verbose
        if scriptmode: self.args = vars(self.parse_arguments())
        if  input_parameters != None:
            self.params = input_parameters
        elif config_file is None:
            self.config_file = self.args['config_file']
        else:
            self.config_file = config_file
            
        if input_parameters is None:                
            self.f = open(self.config_file)
            self.params = json.load(self.f)
            if scriptmode: self.params.update(dict((k,self.args[k]) for k in self.args.keys() if self.args[k] is not None))
            with open(self.params['config_file'],'w') as config_file:
                json.dump(self.params,config_file,indent = 4)

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_file', type = str, help = 'JSON configuration file')
        parser.add_argument('--ifiles', type = str, help = 'List of input files')
        parser.add_argument('--outdir', type = str, help = 'Output directory')
        parser.add_argument('--dataset', type = str, help = 'Type of dataset.\
                                                                    esm: protein sequences representations\
                                                                    mnist: drawn numbers classification\
                                                                    image-net: image classification')
        parser.add_argument('--dim', type = int, help = 'Maximum dimension of simplices')
        parser.add_argument('--knn', type = int, help = 'Number of neighbours for kneighbors graph')           

        args = parser.parse_args()

        return args

    def read_params(self):
        if self.verbose:
            print('Input Parameters:')
            for key,value in self.params.items():
                print(key, ' : ', value)
        return self.params

    def read_reps(self,params = None):
        reps = []
        if params is None:
            params = self.params

        if params['dataset'] == 'text':
            #for each file
            filelst= natsorted(glob.glob(params['ifiles'])) 
            if len(filelst) == 0:
                print('No files found! Exiting.')
                raise SystemExit
            else:
                for file in filelst:
                    print(file)
                    reps.append(np.load(file))            
        if params['dataset'] == 'esm':
            #for each file
            filelst= natsorted(glob.glob(params['ifiles'])) 
            if len(filelst) == 0:
                print('No files found! Exiting.')
                raise SystemExit
            else:
                for file in filelst:
                    print(file)
                    reps.append(np.load(file))
        if params['dataset'] == 'mnist':
            #for each file
            filelst= natsorted(glob.glob(params['ifiles'])) 
            if len(filelst) == 0:
                print('No files found! Exiting.')
                raise SystemExit
            else:
                for file in filelst:
                    print(file)
                    reps.append(np.load(file))
            #for each file
            
        if params['dataset'] == 'image-net':
            print('Image-net dataset not implemented yet.')
            raise SystemExit
        print('Done.')
        return reps
    
    def save_PD(self,dgms, params):
        if params is None:
            params = self.params
        outfile = 'zigzag_PD_' + params['dataset'] + '_knn_' + str(params['knn']) + '_dim_'
        for i,dgm in enumerate(dgms):
            print("Saving %d-cycles" %i)
            temp = []
            for p in dgm:
                temp.append([p.birth,p.death])
            np.savez(params['outdir'] + outfile + str(i) + '.npz', temp)      

class ZIGZAG:
    def __init__(self, params, reps = None, verbose = False):  

        self.verbose = verbose
        self.params = params
        self.reps = reps 
        self.sID = {}

    def generate_simplex_tree(self, reps = None, params = None, labels=None):
        if labels is not None:
	        print("Generating simplex tree with labels")
        else:
            print("Generating simplex tree without labels")
	   
        if params is None:
            params = self.params
        if reps is None:
            reps = self.reps
        simplices = []
        simplices_padded = []
        simplices_set = set()
        
        for i in tqdm(range(len(reps))):
            simplices_padded.append([])
            G=sklearn.neighbors.kneighbors_graph(reps[i], n_neighbors= params['knn'])
            
            
            G_arr = G.toarray()
            S = gd.SimplexTree()                    # most time spent building the simplex tree
            #ind = np.array(list_for_numpy)
            ind=np.array(np.where(G_arr ==1)).T
            # adding vertices to the simplex tree
            for k in range(len(reps[i])): 
                S.insert([k])    
            for k in range(len(ind)):
                u,v = ind[k]
                if labels is not None:
                    if labels[u] != labels[v]:
                        S.insert(list(ind[k]))
                else: 
                    S.insert(list(ind[k]))
                
            S.expansion(params['dim'])

            for s in S.get_skeleton(params['dim']):                
                if(tuple(s[0]) not in simplices_set):
                    self.sID[tuple(s[0])] = len(simplices)
                    simplices.append(s[0])
                    simplices_set.add(tuple(s[0]))

                ## we pad simplices with the last appearing number 
                ## in order to have homogeneous arrays for each layer
                simplices_padded[i].append(self.sID[tuple(s[0])])
        
        return simplices,simplices_padded
    
    def compute_layers_with_intersection(self,simplices_padded):
        layers = []
        for i in range(2*len(simplices_padded)-1):
            layers.append([])
            if i % 2 == 1:
                layers[i] = list(set(simplices_padded[i//2]).intersection(simplices_padded[(i+1)//2]))
            else:
                layers[i] = simplices_padded[i//2]
        return layers

    def compute_filtration_times(self,simplices,layers, params = None):
        if params is None:
            params = self.params
        # compute filtration using Dionysus
        f = d.Filtration(simplices)
        # flist_padded = []
        # for elm in f:
        #    flist_padded.append(np.pad(elm,(0,params['dim']+1-len(elm)), mode='edge'))
        flist_padded = np.arange(len(simplices))
        # compute appearence matrix
        appearence_matrix = np.zeros((len(layers),len(flist_padded)),dtype=int)
        for k in range(len(layers)):
            appearence_matrix[k, layers[k]] =  1
        # now compute times
        times = []
        for i in range(len(appearence_matrix[0])):
            temp = list(utils.ranges(np.where(appearence_matrix[:,i] == 1)[0]))
            times.append(temp)
        times=[list(np.array([list(times[i][j]) for j in range(len(times[i]))]).flatten()) for i in range(len(times))]
        return f,times
    
    def compute_zigzag_persistence(self,filtration,times):
        zz, dgms, cells = d.zigzag_homology_persistence(filtration,times,progress=True)
        """
        for i,dgm in enumerate(dgms):
            print("Dimension:", i)
            rep = []
            for p in dgm:
                if not(np.any(np.array(rep) == p)):
                    rep.append([p,1])
                else:
                    for j in range(len(rep)):
                        if p == rep[j][0]:
                            rep[j][1] = rep[j][1] + 1
                    
        # Print each element in rep on a new line
            for element in rep:
                print(element)
        """
        return zz, dgms, cells