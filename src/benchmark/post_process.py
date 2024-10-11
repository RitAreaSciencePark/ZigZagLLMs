#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def read_pd_from_csv(file, max_dim = 10):
    temp = np.array(pd.read_csv(file))
    temp_pd = [[] for _ in range(max_dim)]
    for item in temp:
        if item[0] > max_dim:
            print(f"Error in reading {item}. Increase the dimension.")
            raise SystemExit
        else:
            temp_pd[item[0]].append([item[1], item[2]])
    return temp_pd

def plot_pi(freq_array, ax, fig, title, log_scale = True):
    arr = np.log10(1 + freq_array) if log_scale else freq_array
    im = ax.imshow(arr , cmap='hot', origin = 'lower')
    label = 'Log Frequency' if log_scale else 'Frequency'
    fig.colorbar(im, label=label, ax = ax)
    ax.set_xlabel('birth')
    if 'S' in title:  ax.set_ylabel('target') 
    else: ax.set_ylabel('death')
    ax.set_title(title)

def div(num, den):
    if den == 0 : return 0
    else : return num/den

class POSTPROCESS:
    def __init__(self, pds, num_layers, start_ind, zigzag, meta_data = None, debug = False):
        self.pds = pds
        self.max_hom_dim = len(pds) # maximum homology dimension.
        self.num_layers = num_layers # number of complexes in the sequence.
        self.start_ind = start_ind # index of the first layer - typically 0 or 1.
        self.zigzag = zigzag # True if is a zigzag sequence.
        self.meta_data = meta_data # contains information like knn, ML model type, etc.
        self.debug = debug # print statements enabled if debug is True.
    
    def extract_pairs(self, pairs):
        start_ind = int(self.start_ind)
        arr = [tuple((int(item[0]) - start_ind, int(item[1]) - start_ind))\
               for item in pairs]
        return arr
    
    def find_betti_curve(self):
        num_layers = self.num_layers
        bettis = []
        for pairs in self.pds: # Iterating through hom_dim
            betti = [0] * num_layers 
            arr = self.extract_pairs(pairs) 
            max_death = max(p[1] for p in pairs) if len(pairs) > 0 else 0
            if self.debug : print(max_death)
            for birth, death in arr:
                for d in range(birth, death):
                    betti[d] += 1
            bettis.append(betti)
        bettis = np.array(bettis)
        return bettis
    
    # Works for zigzag only
    def find_betti_layers(self):
        bettis = self.find_betti_curve()
        return bettis[:, ::2]
    
    def find_pis(self):
        pis = []
        for hdim, pairs in enumerate(self.pds):
            arr = self.extract_pairs(pairs)
            pair_counts = dict(Counter(arr))
            num_layers = self.num_layers
            # Adding 1 for infinity. freq_array[d, b] gives the count of barcodes
            # created at b and dying at d. Note that the first index here is death.
            freq_array = np.zeros((num_layers + 1, num_layers + 1)) 
            if len(arr) != 0:
                pairs, frequencies = zip(*pair_counts.items())
                x, y = zip(*pairs)
                
                if max(y) != num_layers and self.debug == True:
                    print(f"Max(death) != num_layers in hom_dim = {hdim}")
                    
                for pair, freq in zip(pairs, frequencies):
                    freq_array[pair[1]][pair[0]] = freq
            else : print(f"No points in hom_dim = {hdim}")
            pis.append(freq_array)
        return np.array(pis)
    
    # Works for zigzag only
    def find_eff_pis(self, pis = None):
        if pis is None:
            pis = self.find_pis()
        print(pis.shape)
        eff_pis = []
        for arr in pis:
            sz = arr.shape[0]
            res = np.zeros((sz//2 + 1, sz//2 + 1))
            """
            for i, row in enumerate(arr[:-1, :-1]):
                for j, element in enumerate(row):
                    eff_d = i//2 + i%2
                    eff_b = j//2 + j%2
                    if eff_d != eff_b : res[eff_d, eff_b] = res[eff_d, eff_b] + arr[i, j] 
            for i, item in enumerate(arr[-1, :-1]):
                if i %2 == 0:
                    res[-1,i//2] = res[-1,i//2] + arr[-1, i]
                
                if i %2 == 1:
                    res[-1,i//2 + 1] = res[-1,i//2 + 1] + arr[-1, i]
            """
            for i, row in enumerate(arr):
                for j, element in enumerate(row):
                    eff_d = i//2 + i%2
                    eff_b = j//2 + j%2
                    if eff_d != eff_b : res[eff_d, eff_b] = res[eff_d, eff_b] + arr[i, j] 
            eff_pis.append(res)
        return np.array(eff_pis)
        
    def find_ph_sim(self, pis = None):
        if pis is None:
            pis = self.find_eff_pis() if self.zigzag else self.find_pis()
        num_layers = pis[0].shape[0] - 1
        max_hom_dim = len(pis)
        psim_mats = np.zeros((max_hom_dim, num_layers, num_layers))
        for hdim in range(max_hom_dim):
            for idx in range(num_layers):
                psim_mats[hdim, idx, idx] =  np.sum(pis[hdim][idx + 1:,:idx + 1])
                for jdx in range(idx + 1, num_layers):
                    # psim_mats[hdim, idx, jdx] =  np.sum(pis[hdim][jdx + 1:,:idx + 1])
                    psim_mats[hdim, idx, jdx] = psim_mats[hdim, idx, jdx - 1] - \
                        np.sum(pis[hdim][jdx,:idx + 1])
                    psim_mats[hdim, jdx, idx] = psim_mats[hdim, idx, jdx]
                
        return psim_mats
    
    def find_correlation_length(self, psim_mats, bettis, pval):
        full_correlations = []
        left_correlations = []
        for hom_dim in range(len(psim_mats)): 
            pmat = psim_mats[hom_dim]
            inv_pi = np.zeros(pmat.shape[0])
            for l in range(0, pmat.shape[0]):
                inv_pi[l]  = np.sum(div(pmat[l], bettis[hom_dim, l]) > pval)
                
            full_correlations.append(inv_pi)
            left_corn = []
            for l in range(0, pmat.shape[0]):
                val = 0
                if bettis[hom_dim, l] != 0: 
                    val = l - np.argmax(div(pmat[l], bettis[hom_dim, l]) > pval)
                left_corn.append(val)
            left_correlations.append(left_corn)

        # shape = max_hom_dim, num_layers
        return {"full": np.array(full_correlations), "left": np.array(left_correlations)} 
    
    def compute_persistence_summaries(self):
        if self.zigzag:
            bettis = self.find_betti_layers()
            pis = self.find_eff_pis()
        else:
            bettis = self.find_betti_curve()
            pis = self.find_pis()
        psim_mats = self.find_ph_sim(pis)
        return {"bettis" : bettis, "pis": pis, "phsim": psim_mats}    
    
    def plot_betti_curve(self, bettis, hom_dim, ax):
        ax.plot(bettis[hom_dim], marker = 'o', label = "k = 5")
        ax.set_title(f"$\\beta_{hom_dim}$")
        ax.grid(True)
        
    def plot_persistence_images(self, pis, hom_dim, ax, fig, log_scale = True):
        plot_pi(pis[hom_dim], ax, fig, f"$Hist_{hom_dim}$", log_scale = log_scale)
    
        # fig.suptitle("Barcode histograms")
        # plt.tight_layout()
        # plt.show()
    
    def plot_psim_mats(self, psim_mats, bettis, hom_dim, ax, \
                       layers = [10, 15, 20, 25, 30]):
        
        colormap = plt.cm.viridis
        for l in layers:
            ax.plot(div(psim_mats[hom_dim, l],bettis[hom_dim, l]), \
                     color = colormap(l/psim_mats.shape[1]), marker = '.', \
                         label = f"$\ell = {l}$")
                
        ax.set_title(f"$PHSim_{hom_dim}(\ell$)")
        ax.grid(True)
        ax.axhline(0.5, c = 'black', linestyle = 'dotted')
        ax.axhline(0.3, c = 'black', linestyle = 'dotted')
        
        # plt.tight_layout()
        # plt.show()
    
    def plot_correlation_lengths(self, psim_mats, bettis, hom_dim, ax_left, \
                                 ax_full, pvals = np.linspace(0.3, 0.6, 4)):
        corns = [self.find_correlation_length(psim_mats, bettis, pval) for pval in pvals]
        colormap = plt.cm.plasma
        for jdx, corn in enumerate(corns):
            color = colormap(pvals[jdx]) 
            ax_left.plot(corn["left"][hom_dim], marker = '.', \
                         label = f"p = {np.round(pvals[jdx], 2)}", color = color)
            ax_full.plot(corn["full"][hom_dim], marker = '.', \
                         label = f"p = {np.round(pvals[jdx], 2)}", color = color)
        
            ax_left.set_title(f"$\kappa_{hom_dim}$ (left)")
            ax_left.grid(True)
            ax_full.set_title(f"$\kappa_{hom_dim}$ (full)")
            ax_full.grid(True)
        
        # print(len(corns), pvals)
        # plt.tight_layout()
        # plt.show()
                    
    def compute_and_plot(self, hom_dim_list = [1, 2], \
                         log_scale = [True, True],\
                         pvals = np.linspace(0.3, 0.6, 4),
                         maxval = None, layers = [10, 15, 20, 25, 30],
                         filename = None, title = None):
        num_rows = len(hom_dim_list); num_cols = 5
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols - 1, 5 * num_rows - 1))
        op = self.compute_persistence_summaries()
        bettis, pis, psim_mats = op["bettis"], op["pis"], op["phsim"]
        if maxval is not None:
            bettis = bettis[:, :maxval]
            pis = pis[:, :maxval, :maxval]
            psim_mats = psim_mats[:, :maxval, :maxval]
        for idx, hom_dim in enumerate(hom_dim_list):
            ax = axes[idx]
            self.plot_betti_curve(bettis, hom_dim, ax = ax[0])
            self.plot_persistence_images(pis, hom_dim, fig = fig, ax = ax[1])
            self.plot_psim_mats(psim_mats, bettis, hom_dim, ax = ax[2], layers = layers)
            self.plot_correlation_lengths(psim_mats, bettis, hom_dim, ax_left = ax[3], ax_full = ax[4], pvals = pvals)
        
        # Create legends outside the plots
        handles_psim, labels_psim = axes[0, 2].get_legend_handles_labels()
        fig.legend(handles_psim, labels_psim, loc='upper center', \
                   bbox_to_anchor=(0.51, 0), ncol = 4, \
                       title="PHSim", fontsize = "x-large")
        handles_corn, labels_corn = axes[0, 3].get_legend_handles_labels()
        fig.legend(handles_corn, labels_corn, loc='upper center', \
                   bbox_to_anchor=(0.8, 0), ncol = 4, title="$\kappa$", \
                       fontsize = "x-large")

        
        if title is not None : fig.suptitle(title)
        plt.tight_layout()
        # plt.show()
        # plt.show()
        if filename is not None : 
            plt.savefig(filename)
            plt.close()
        else : plt.show()

    def compute_and_test(self, pvals = np.linspace(0.3, 0.6, 4),
                         maxval = None):
        op = self.compute_persistence_summaries()
        bettis, pis, psim_mats = op["bettis"], op["pis"], op["phsim"]
        if maxval is not None:
            bettis = bettis[:, :maxval]
            pis = pis[:, :maxval, :maxval]
            psim_mats = psim_mats[:, :maxval, :maxval]
        correlation_lengths = []
        for i in pvals:
            correlation_lengths.append(self.find_correlation_length(psim_mats, bettis, i))
        num_layers = bettis.shape[1]
        layers = np.arange(num_layers)
        return pvals,layers, bettis, pis, psim_mats, correlation_lengths

        
        
        
        

        
        

            
        
        


    

        
