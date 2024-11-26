
from re import L
import numpy as np 
import csv
from rdkit import Chem
import torch

from .featurizing_helpers import *

import itertools

import dgl
import torch
import os

from graphormer.data import register_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import pickle

num_groups = 18
one_hot_encoding = [[0] * num_groups for _ in range(num_groups)]
for i in range(num_groups):
    one_hot_encoding[i][i] = 1

class IRSpectraD(DGLDataset):
    def __init__(self):
        self.mode = ":("
        ## atom encodings
        atom_type_onehot = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]

        formal_charge_onehot =[
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]

        hybridization_onehot =[
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]

        is_aromatic_onehot = [
            [0], 
            [1]
        ]

        total_num_H_onehot = [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]

        explicit_valence_onehot = [
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0], 
            [0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
        ]

        total_bonds_onehot = [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 0, 1],
        ]

        ## TODO: Add encoding for global node in the atom type

        i = 0
        self.one_hotatom_to_int_keys = []
        self.one_hotatom_to_int_values = []
        self.hash_dictatom = {}
        self.comb_atom = False

        if self.comb_atom: ## if you want to do combinatoric atom hashing
            for x1 in atom_type_onehot:
                for x2 in formal_charge_onehot:
                    for x3 in hybridization_onehot:
                        for x4 in is_aromatic_onehot:
                            for x5 in total_num_H_onehot: 
                                for x6 in explicit_valence_oesnehot:
                                    for x7 in total_bonds_onehot:
                                        key = torch.cat([torch.Tensor(y) for y in [x1, x2, x3, x4, x5, x6, x7]])
                                        self.one_hotatom_to_int_keys += [key]
                                        self.one_hotatom_to_int_values += [i]
                                        i+=1
                                                        
            count = 0
            while count < len(self.one_hotatom_to_int_keys):
                h = str(self.one_hotatom_to_int_keys[count])
                self.hash_dictatom[h] = self.one_hotatom_to_int_values[count]
                count +=1
            

        ## combinatoric bond mapping
        bond_type_onehot = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]


        is_in_ring_onehot = [
            [0], 
            [1]
        ]

        bond_stereo_onehot = [
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0]
        ]

        is_global_node = [
            [0],
            [1]
        ] # 2022-12-19

        i = 0
        self.one_hot_to_int_keys = []
        self.one_hot_to_int_values = []
        self.hash_dict = {}
        for x1 in bond_type_onehot:
            for x3 in is_in_ring_onehot:
                for x4 in bond_stereo_onehot:
                    for x5 in is_global_node:
                        key = torch.cat([torch.Tensor(y) for y in [x1, x3, x4, x5]]) ## cute quick way to 
                        self.one_hot_to_int_keys += [key]
                        self.one_hot_to_int_values += [i]
                        i+=1

        count = 0
        while count < len(self.one_hot_to_int_keys):
            h = str(self.one_hot_to_int_keys[count])
            self.hash_dict[h] = self.one_hot_to_int_values[count]
            count +=1

        self.num_classes = 1801
        super().__init__(name='IR Spectra', save_dir='/home/cmkstien/Graphormer/examples/property_prediction/training_dataset/') 

    def process(self):
        
        self.graphs = []
        self.labels = []
        self.smiles = []

        print("I'm in the right file")
        x = import_data(r'/home/cmkstien/Desktop/IR_data/sample_IR_train_data.csv')
        x = x[1:] ## removing header
        
        print("Loading Data and Converting SMILES to DGL graphs")
        count_outliers = 0

        gnode = True ## Turns off global node
        count = 0
        count_hash = 0
        for i in tqdm(x):
            sm = str(i[0]).replace("Q", "#") ## Hashtags break some of our preprocessing scripts so we replace them with Qs to make life easier 
            phase = i[1]

            sp = torch.tensor(np.asarray(i[2:], dtype = np.float64), dtype=torch.float64, device=torch.device('cpu')) 
            sp = torch.clip(sp, min = 10e-8) # clipping
            sp_nanmask = torch.isnan(sp)
            sp[sp_nanmask] = 0 ## Masking out all NaN values
            sp_sum = torch.sum(sp)

            sp = torch.divide(sp, sp_sum)
            sp[sp_nanmask] = np.NaN

            mol = Chem.MolFromSmiles(sm)
            num_atoms = mol.GetNumAtoms()

            add_self_loop = False
            g = mol_to_bigraph(mol, explicit_hydrogens=False, node_featurizer=GraphormerAtomFeaturizer(), edge_featurizer=CanonicalBondFeaturizer(), add_self_loop=False) ## uses DGL featurization function                
            ###########################################################################
            count1 = 0
            count2 = 0

            unif = []
            unifatom = []

            ### GLOBAL NODE Encodings
            
            while count2 < len(g.ndata['h']): ## getting all the parameters needed for the global node generation
                hatom = g.ndata['h'][count2][:]
                unifatom.append(list(np.asarray(hatom)))
                flength = len(list(hatom))
                count2 += 1

            src_list = list(np.full(num_atoms, num_atoms)) ## node pairs describing edges in heteograph - see DGL documentation
            dst_list = list(np.arange(num_atoms))


            features = torch.tensor([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
            total_features = features.repeat(num_atoms, 1)

            if gnode:
                features = torch.tensor([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
                total_features = features.repeat(num_atoms, 1)
                if phase == "nujol mull":
                    g_nm = [100] + list(np.zeros(flength-1)) ## custom encoding for the global node
                    unifatom.append(g_nm)
                    g.add_nodes(1)
                elif phase == "CCl4":
                    g_nm = [101] + list(np.zeros(flength-1))
                    unifatom.append(g_nm)
                    g.add_nodes(1)
                elif phase == "liquid film":
                    g_nm = [102] + list(np.zeros(flength-1))
                    unifatom.append(g_nm)
                    g.add_nodes(1)
                elif phase == "KBr" or phase == "KCl":
                    g_nm = [103] + list(np.zeros(flength-1))
                    unifatom.append(g_nm)
                    g.add_nodes(1)

                elif phase == "gas":
                    g_nm = [104] + list(np.zeros(flength-1))

                    unifatom.append(g_nm)
                    g.add_nodes(1)
                else:
                    print("Not A Valid Phase with " + phase)
                    count_outliers +=1
                    continue
                g.ndata['h'] = torch.tensor(unifatom)
                g.add_edges(src_list, dst_list, {'e': total_features}) ## adding all the edges for the global node

            # ### GLOBAL NODE THINGS
            
            # ###########################################################################
            # Combinatoric Edge Hashing Using Library
            if g.edata == {}:
                print("We did it mom - one atom molecule doesn't break things")
            else:
                while count1 < len(g.edata['e']):
      
                    h = str(g.edata['e'][count1])
                    unif.append(self.hash_dict[h])
                    count1 += 1

                g.edata['e'] = torch.transpose(torch.tensor(unif), 0, -1) + 1

            self.graphs.append(g)
            self.labels.append(sp)
            self.smiles.append(sm)
            count+=1
        
            # if count == 500:
            #     break

    def __getitem__(self, i):
        # print(i)
        return self.graphs[i], self.labels[i], self.smiles[i]

    def __len__(self):
        return len(self.graphs)
    
@register_dataset("IR_test")
def create_customized_dataset():

    dataset = IRSpectraD()
    num_graphs = len(dataset)

    return {
        "dataset": dataset,
        "train_idx": np.arange(0,num_graphs), ## grabbing entire test dataset from file
        "valid_idx": None,
        "test_idx": None,
        "source": "dgl" 
    }
