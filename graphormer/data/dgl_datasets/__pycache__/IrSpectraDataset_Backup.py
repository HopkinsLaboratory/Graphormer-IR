
import numpy as np 
import csv
from dgl.data import QM9 
from rdkit import Chem
import dgllife.utils as d
import torch
import json
import pandas as pd
from functools import partial
from collections import defaultdict

from rdkit.Chem import rdmolfiles, rdmolops
import itertools
import os.path as osp

import dgl
from dgl.data import DGLDataset
import torch
import os

from graphormer.data import register_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import dgl.backend as F
from featurizing_functions import *


from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info


try:
    from rdkit import Chem, RDConfig
    from rdkit.Chem import AllChem, ChemicalFeatures

except ImportError:
    pass


def bond_features(bond):
    return[0]

def construct_bigraph_from_mol(mol, add_self_loop=False):

    g = dgl.graph(([], []), idtype=torch.int32)

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    g.add_nodes(num_atoms + num_bonds)
    # Add edges
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        b = i + num_atoms
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, b])
        dst_list.extend([b, u])

        src_list.extend([v, b])
        dst_list.extend([b, v])

    if add_self_loop:
        nodes = g.nodes().tolist()
        src_list.extend(nodes)
        dst_list.extend(nodes)

    g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))
    return g


def mol_to_graph(mol, graph_constructor, node_featurizer, edge_featurizer,
                 canonical_atom_order, explicit_hydrogens=False, num_virtual_nodes=0):
 
    if mol is None:
        print('Invalid mol found')
        return None

    # Whether to have hydrogen atoms as explicit nodes
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)

    g = graph_constructor(mol)

    if node_featurizer is not None:
        g.ndata.update(node_featurizer(mol))
        
    if edge_featurizer is not None:
        g.edata.update(edge_featurizer(mol))

    if num_virtual_nodes > 0:
        num_real_nodes = g.num_nodes()
        real_nodes = list(range(num_real_nodes))
        g.add_nodes(num_virtual_nodes)

        # Change Topology
        virtual_src = []
        virtual_dst = []
        for count in range(num_virtual_nodes):
            virtual_node = num_real_nodes + count
            virtual_node_copy = [virtual_node] * num_real_nodes
            virtual_src.extend(real_nodes)
            virtual_src.extend(virtual_node_copy)
            virtual_dst.extend(virtual_node_copy)
            virtual_dst.extend(real_nodes)
        g.add_edges(virtual_src, virtual_dst)

        for nk, nv in g.ndata.items():
            nv = torch.cat([nv, torch.zeros(g.num_nodes(), 1)], dim=1)
            nv[-num_virtual_nodes:, -1] = 1
            g.ndata[nk] = nv

        for ek, ev in g.edata.items():
            ev = torch.cat([ev, torch.zeros(g.num_edges(), 1)], dim=1)
            ev[-num_virtual_nodes * num_real_nodes * 2:, -1] = 1
            g.edata[ek] = ev

    return g


def mol_to_bigraph(mol, add_self_loop=False,
                   node_featurizer=None,
                   edge_featurizer=None,
                   canonical_atom_order=True,
                   explicit_hydrogens=False,
                   num_virtual_nodes=0):

    return mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                        node_featurizer, edge_featurizer,
                        canonical_atom_order, explicit_hydrogens, num_virtual_nodes)


class BaseBondFeaturizer(object):

    def __init__(self, featurizer_funcs, feat_sizes=None, self_loop=False):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes
        self._self_loop = self_loop

    def feat_size(self, feat_name=None):

        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, \
                'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        mol = Chem.MolFromSmiles('CCO')
        feats = self(mol)

        return feats[feat_name].shape[1]

    def __call__(self, mol):

        num_bonds = mol.GetNumBonds()
        bond_features = defaultdict(list)

        # Compute features for each bond
        for i in range(num_bonds):

            bond = mol.GetBondWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                feat = feat_func(bond)
                bond_features[feat_name].extend([feat, feat.copy()])
                bond_features[feat_name].extend([feat, feat.copy()])

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in bond_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))

        if self._self_loop and num_bonds > 0:
            num_atoms = mol.GetNumAtoms()
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.cat([feats, torch.zeros(feats.shape[0], 1)], dim=1)
                self_loop_feats = torch.zeros(num_atoms, feats.shape[1])
                self_loop_feats[:, -1] = 1
                feats = torch.cat([feats, self_loop_feats], dim=0)
                processed_features[feat_name] = feats

        if self._self_loop and num_bonds == 0:
            num_atoms = mol.GetNumAtoms()
            toy_mol = Chem.MolFromSmiles('CO')
            processed_features = self(toy_mol)
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.zeros(num_atoms, feats.shape[1])
                feats[:, -1] = 1
                processed_features[feat_name] = feats

        return processed_features



class BaseAtomFeaturizer(object):
    def __init__(self, featurizer_funcs, feat_sizes=None):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes

    def feat_size(self, feat_name=None):
    
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, \
                'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        if feat_name not in self._feat_sizes:
            atom = Chem.MolFromSmiles('C').GetAtomWithIdx(0)
            self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](atom))

        return self._feat_sizes[feat_name]

    def __call__(self, mol):
 
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        atom_features = defaultdict(list)

        # Compute features for each atom
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(atom))
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(bond))

        # Stack the features and convert them to float arrays
        processed_features = dict()

        for feat_name, feat_list in atom_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))

        return processed_features


class ConcatFeaturizer(object):

    def __init__(self, func_list):
        self.func_list = func_list


    def __call__(self, x):
        return list(itertools.chain.from_iterable(
            [func(x) for func in self.func_list]))

class CanonicalBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_field='e', self_loop=False):
        super(CanonicalBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [edge_features]

            )}, self_loop=self_loop)


class GraphormerAtomFeaturizer(object):

    def __init__(self, atom_data_field='h', atom_types=None, chiral_types=None,
                 hybridization_types=None):
        super(GraphormerAtomFeaturizer, self).__init__()

        self._atom_data_field = atom_data_field

        if atom_types is None:
            atom_types = ['H', 'C', 'N', 'O', 'F','Si', 'P', 'S', 'Cl', 'Br', 'I']
        self._atom_types = atom_types

        if chiral_types is None:
            chiral_types = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
        self._chiral_types = chiral_types

        if hybridization_types is None:
            hybridization_types = [Chem.rdchem.HybridizationType.SP,
                                   Chem.rdchem.HybridizationType.SP2,
                                   Chem.rdchem.HybridizationType.SP3]
        self._hybridization_types = hybridization_types

    
        self._featurizer = ConcatFeaturizer([
            # partial(atom_type_one_hot, allowable_set=atom_types),
            # partial(atom_chiral_tag_one_hot, allowable_set=chiral_types),
            # partial(atom_hybridization_one_hot, allowable_set=hybridization_types),
            # atom_formal_charge_one_hot, 
            # atom_partial_charge, 
            atom_mass,
            # atom_total_num_H_one_hot, 
            # atom_explicit_valence_one_hot,
            # atom_is_aromatic_one_hot,
            bond_type_one_hot,
            # bond_stereo_one_hot,
            # bond_is_in_ring_one_hot,
            # bond_is_conjugated_one_hot
        ])

    def feat_size(self):
        mol = Chem.MolFromSmiles('C')
        feats = self(mol)[self._atom_data_field]

        return feats.shape[-1]

    def __call__(self, mol):

        atom_features = []

        AllChem.ComputeGasteigerCharges(mol)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        # Get information for donor and acceptor
        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        mol_feats = mol_featurizer.GetFeaturesForMol(mol)

        # Get a symmetrized smallest set of smallest rings
        # Following the practice from Chainer Chemistry (https://github.com/chainer/
        # chainer-chemistry/blob/da2507b38f903a8ee333e487d422ba6dcec49b05/chainer_chemistry/
        # dataset/preprocessors/weavenet_preprocessor.py)
        sssr = Chem.GetSymmSSSR(mol)
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            feats = self._featurizer(atom)
            atom_features.append(feats)
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            feats = self._featurizer(bond)
            atom_features.append(feats)

        atom_features = np.stack(atom_features)

        return {self._atom_data_field: F.zerocopy_from_numpy(atom_features.astype(np.float32))}

def import_smiles(file):
    with open(file,'r') as rf:
        r=csv.reader(rf)
        next(r)
        smiles=[]
        for row in r:
            smiles.append(row[0])
        return smiles

def import_data(file):
    with open(file,'r') as rf:
        r=csv.reader(rf)
        next(r)
        data=[]
        for row in r:
            data.append(row)
        return data


x = import_data(r'/home/weeb/Desktop/Cailum/data/test_full.csv')

data = QM9(label_keys=["mu"])
print(data)

class IRSpectraD(DGLDataset):
    def __init__(self):
        self.mode = ":("
        self.save_path_2 = '/home/weeb/shit/Graphormer/examples/property_prediction/training_dataset/'
    

        super().__init__(name='IR Spectra') #save_dir='/home/weeb/shit/Graphormer/examples/property_prediction/training_dataset/')
        
    def process(self):
        
        self.graphs = []
        self.labels = []

        x = import_data(r'/home/weeb/Desktop/chemprop-IR_Zenodo/chemprop-IR/testmodel/test_full_chemprop.csv')
        print("also the right file")
        count = 0
        print("Loading Data and Converting SMILES to DGL graphs")
        for i in tqdm(x):
            # print(len(i))
            sm = i[1]
            mol = Chem.MolFromSmiles(sm)

            # n =  mol.GetNumHeavyAtoms()
            sp = torch.tensor(np.asarray(i[2:], dtype = np.float64), dtype=torch.float64) 
            sp_sum = torch.sum(sp)#.item()
            sp = torch.divide(sp, sp_sum)



            try:  
                add_self_loop = False
                g = mol_to_bigraph(mol, node_featurizer=GraphormerAtomFeaturizer(), edge_featurizer=CanonicalBondFeaturizer(), explicit_hydrogens = False)
                for j in g.ndata['h']:
                    print(j)
                self.graphs.append(g)
                self.labels.append(sp)
                print(g.ndata['h'].shape)
                print(g.edata['e'].shape)

            except:
                print(":(")
            count +=1
            # break

            if count == 2:
                print("fuck yeah bud")
                exit()
                break

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    
    # def save(self):
    #     # save graphs and labels

    #     # graph_path = os.path.join(self.save_path_2, self.mode + '_dgl_graph.bin')
    #     # save_graphs(graph_path, self.graphs, {'labels': self.labels})
    #     # # save other information in python dict
    #     # info_path = os.path.join(self.save_path_2, self.mode + '_info.pkl')
    #     # save_info(info_path, {'num_classes': self.num_classes})

    # def load(self):
    #     # load processed data from directory `self.save_path`
    #     graph_path = os.path.join(self.save_path_2, self.mode + '_dgl_graph.bin')
    #     self.graphs, label_dict = load_graphs(graph_path)
    #     self.labels = label_dict['labels']
    #     info_path = os.path.join(self.save_path_2, self.mode + '_info.pkl')
    #     self.num_classes = load_info(info_path)['num_classes']

    # def has_cache(self):
    #     # check whether there are processed data in `self.save_path`
    #     graph_path = os.path.join(self.save_path_2, self.mode + '_dgl_graph.bin')
    #     info_path = os.path.join(self.save_path_2, self.mode + '_info.pkl')
    #     return os.path.exists(graph_path) and os.path.exists(info_path)

@register_dataset("customized_IRSpectraDataset")
def create_customized_dataset():
    save_path = '/home/weeb/shit/Graphormer/examples/property_prediction/training_dataset/'
    dataset = IRSpectraD()

    # graph_path = os.path.join(save_path, 'trainingdatasetgraphs_dgl_graph.bin')
    # info_path = os.path.join(save_path, 'trainingdatasetlabels_info.pkl')
    # if os.path.exists('/home/weeb/shit/Graphormer/examples/property_prediction/training_dataset/trainingdatasetgraphs_dgl_graph.bin'):
    #     dataset = load_graphs(graph_path)
    #     labels = load_info(info_path)
    # else:
        

    # save_graphs(graph_path, dataset.graphs)
    # save_info(info_path, {'labels': dataset.labels})
   
    num_graphs = len(dataset)

    print("Dataset has been Registered")
    # customized dataset split
    train_valid_idx, test_idx = train_test_split(
    np.arange(num_graphs), test_size=1, random_state=0
    )

    train_idx, valid_idx = train_test_split(
        train_valid_idx, test_size=num_graphs // 10, random_state=0
    )
    return {

         "dataset": dataset,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "source": "dgl"
    }




# sm = 'C#N'

# mol = Chem.MolFromSmiles(sm)
# g = mol_to_bigraph(mol, add_self_loop=False, node_featurizer=GraphormerAtomFeaturizer(), edge_featurizer=CanonicalBondFeaturizer(), explicit_hydrogens = False)
# print(g.ndata)
# # print(g.edata['e'])
