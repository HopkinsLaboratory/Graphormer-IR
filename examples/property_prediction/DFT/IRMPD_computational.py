
from re import L
import numpy as np 
import csv
from dgl.data import QM9 
from rdkit import Chem
from dgllife.utils import BaseBondFeaturizer
import dgllife.utils as d
import torch
import json
import pandas as pd
from functools import partial
from rdkit.Chem import rdmolfiles, rdmolops
import ast
from random import shuffle
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
import pickle
# from featurization import MolGraph

from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from joblib import delayed, Parallel

try:
    from rdkit import Chem, RDConfig
    from rdkit.Chem import AllChem, ChemicalFeatures

except ImportError:
    pass

import logging
logger = logging.getLogger(__name__)
logger.setLevel(20)
print(logger.getEffectiveLevel())

num_groups = 18
one_hot_encoding = [[0] * num_groups for _ in range(num_groups)]
for i in range(num_groups):
    one_hot_encoding[i][i] = 1

def atom_group(atom, return_one_hot=True, unknown_group=None):
    """
    Get the group number (column number) of an RDKit atom object in the periodic table.

    Parameters:
        atom (rdkit.Chem.Atom): The RDKit atom object.
        return_one_hot (bool): Whether to return the group as a one-hot encoding.
        unknown_group (int or list or None): The encoding to return for atoms with unknown groups.

    Returns:
        int or list or None: The group number of the atom as an integer (if return_one_hot is False),
                             or a one-hot encoding of the group as a list (if return_one_hot is True),
                             or the value provided in unknown_group if the group is not found.
    """
    # Get the atomic number of the atom
    atomic_number = atom.GetAtomicNum()

    # Map atomic numbers to group numbers (column numbers) in the periodic table
    atomic_number_to_group = {
        1: 1, 2: 18,
        3: 1, 4: 14, 5: 15, 6: 16, 7: 17,
        8: 18, 9: 17, 10: 18, 11: 1, 12: 2,
        13: 13, 14: 14, 15: 15, 16: 16, 17: 17,
        18: 18, 19: 1, 20: 2,
        # Add halogens:
        53: 17,  # Iodine (I) is in group 17
        35: 17,  # Bromine (Br) is in group 17
        9: 17,   # Fluorine (F) is in group 17
        # Continue mapping atomic numbers to groups as needed
    }

    # Get the group based on the atomic number
    group = atomic_number_to_group.get(atomic_number, None)

    if return_one_hot:
        num_groups = 18  # Assuming there are 18 groups in the periodic table

        # Encode group as one-hot
        one_hot_group = [0] * num_groups
        if group is not None:
            one_hot_group[group - 1] = 1

        return one_hot_group if group is not None else unknown_group
    else:
        return [group] if group is not None else unknown_group




def atom_period(atom, return_one_hot=True, unknown_period=None):
    """
    Get the period (row number) of an RDKit atom object in the periodic table.

    Parameters:
        atom (rdkit.Chem.Atom): The RDKit atom object.
        return_one_hot (bool): Whether to return the period as a one-hot encoding.
        unknown_period (int or list or None): The encoding to return for atoms with unknown periods.

    Returns:
        int or list or None: The period (row number) of the atom as an integer (if return_one_hot is False),
                             or a one-hot encoding of the period as a list (if return_one_hot is True),
                             or the value provided in unknown_period if the period is not found.
    """
    # Get the atomic number of the atom
    atomic_number = atom.GetAtomicNum()

    # Map atomic numbers to periods (row numbers) in the periodic table
    atomic_number_to_period = {
        1: 1, 2: 1,
        3: 2, 4: 2, 5: 2, 6: 2, 7: 2,
        8: 2, 9: 2, 10: 2, 11: 3, 12: 3,
        13: 3, 14: 3, 15: 3, 16: 3, 17: 2,  # Chlorine (Cl) is in period 2 (change from the previous version)
        18: 3, 19: 4, 20: 4,
        # Add halogens:
        53: 5,  # Iodine (I) is in period 5
        35: 3,  # Bromine (Br) is in period 3
        9: 2,   # Fluorine (F) is in period 2
        # Continue mapping atomic numbers to periods as needed
    }

    # Get the period based on the atomic number
    period = atomic_number_to_period.get(atomic_number, None)
    if return_one_hot:
        num_periods = 7  # Assuming there are 7 periods in the periodic table
        one_hot = [0] * num_periods
        if period is not None:
            one_hot[period - 1] = 1
        return one_hot if period is not None else unknown_period
    else:
        return [period] if period is not None else unknown_period


def atom_mass(atom, coef=0.01):
    """Get the mass of an atom and scale it.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    coef : float
        The mass will be multiplied by ``coef``.

    Returns
    -------
    list
        List containing one float only.
    """
    return [atom.GetMass() * 100]

def atom_explicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    

    allowable_set = list(range(0, 7))
    # print(atom.GetExplicitValence())
    return d.one_hot_encoding(atom.GetExplicitValence(), allowable_set, encode_unknown)


def atom_type_one_hot(atom, allowable_set=None, encode_unknown=False):

    allowable_set = ['H', 'C', 'N', 'O', 'F','Si', 'P', 'S', 'Cl', 'Br', 'I']
    return d.one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)

def atom_is_aromatic_one_hot(atom, allowable_set=None, encode_unknown=False):
    
    if allowable_set is None:
        allowable_set = [False, True]
    val = atom.GetIsAromatic()
    if val:
        return [0]
    else:
        return [1] 

def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):

    allowable_set = [
                    Chem.rdchem.HybridizationType.S, 
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,                       
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2 
                    ]

    return d.one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)


def atom_formal_charge_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(-2, 4))
    return d.one_hot_encoding(atom.GetFormalCharge(), allowable_set, encode_unknown)


def construct_bigraph_from_mol(mol, add_self_loop=False):
  
    g = dgl.graph(([], []), idtype=torch.int32)

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    # Add edges
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
    # for i in range(num_atoms):
    #     src_list.append(num_atoms+1)
    #     drc_list.append(i)

    if add_self_loop:
        nodes = g.nodes().tolist()
        src_list.extend(nodes)
        dst_list.extend(nodes)

    g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))

    return g




def smiles_to_bigraph(smiles, add_self_loop=False,
                      node_featurizer=None,
                      edge_featurizer=None,
                      canonical_atom_order=True,
                      explicit_hydrogens=False,
                      num_virtual_nodes=0):
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    src = []
    dst = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j or add_self_loop:
                src.append(i)
                dst.append(j)

    g = dgl.graph((torch.IntTensor(src), torch.IntTensor(dst)), idtype=torch.int32)

    return g

def atom_partial_charge(atom):
   
    gasteiger_charge = atom.GetProp('_GasteigerCharge')
    if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
        gasteiger_charge = 0
    return [float(gasteiger_charge)]

featurizer_funcs = ['one_hot_encoding',
           'atom_type_one_hot',
           'atomic_number_one_hot',
           'atomic_number',
           'atom_degree_one_hot',
           'atom_degree',
           'atom_total_degree_one_hot',
           'atom_total_degree',
           'atom_explicit_valence_one_hot',
           'atom_explicit_valence',
           'atom_implicit_valence_one_hot',
           'atom_implicit_valence',
           'atom_hybridization_one_hot',
           'atom_total_num_H_one_hot',
           'atom_total_num_H',
           'atom_formal_charge_one_hot',
           'atom_formal_charge',
           'atom_num_radical_electrons_one_hot',
           'atom_num_radical_electrons',
           'atom_is_aromatic_one_hot',
           'atom_is_aromatic',
           'atom_is_in_ring_one_hot',
           'atom_is_in_ring',
           'atom_chiral_tag_one_hot',
           'atom_chirality_type_one_hot',
           'atom_mass',
           'atom_is_chiral_center',
           'ConcatFeaturizer',
           'BaseAtomFeaturizer',
           'CanonicalAtomFeaturizer',
           'WeaveAtomFeaturizer',
           'PretrainAtomFeaturizer',
           'AttentiveFPAtomFeaturizer',
           'PAGTNAtomFeaturizer',
           'bond_type_one_hot',
           'bond_is_conjugated_one_hot',
           'bond_is_conjugated',
           'bond_is_in_ring_one_hot',
           'bond_is_in_ring',
           'bond_stereo_one_hot',
           'bond_direction_one_hot',
           'BaseBondFeaturizer',
           'CanonicalBondFeaturizer',
           'WeaveEdgeFeaturizer',
           'PretrainBondFeaturizer',
           'AttentiveFPBondFeaturizer',
           'PAGTNEdgeFeaturizer']


class BaseAtomFeaturizer(object):
    def __init__(self, featurizer_funcs, feat_sizes=None):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes

    def feat_size(self, feat_name=None):
        """Get the feature size for ``feat_name``.

        When there is only one feature, users do not need to provide ``feat_name``.

        Parameters
        ----------
        feat_name : str
            Feature for query.

        Returns
        -------
        int
            Feature size for the feature with name ``feat_name``. Default to None.
        """
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
        """Featurize all atoms in a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            For each function in self.featurizer_funcs with the key ``k``, store the computed
            feature under the key ``k``. Each feature is a tensor of dtype float32 and shape
            (N, M), where N is the number of atoms in the molecule.
        """
        num_atoms = mol.GetNumAtoms()
        atom_features = defaultdict(list)

        # Compute features for each atom
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(atom))

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in atom_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))

        return processed_features


def atom_total_bonds(atom, allowable_set=None, encode_unknown=False):
    # print("IM USING THIS FUNCTION")
    mol = atom.GetOwningMol()
    id = atom.GetIdx()
    count = 0
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        if u == id or v == id:
            count += 1

    allowable_set = list(range(0,7))
    # print(count, "YOU HAVE THIS MANY BONDS")
    return d.one_hot_encoding(count, allowable_set, encode_unknown)

class ConcatFeaturizer(object):

    def __init__(self, func_list):
        self.func_list = func_list


    def __call__(self, x):
        return list(itertools.chain.from_iterable(
            [func(x) for func in self.func_list]))

def is_global_node(is_gnode:bool): # 2022-12-19
    return [0]

class CanonicalBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_field='e', self_loop=False):
        super(CanonicalBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [d.bond_type_one_hot,
                #  d.bond_is_conjugated,
                 d.bond_is_in_ring,
                 d.bond_stereo_one_hot,
                 is_global_node]

            )}, self_loop=self_loop)


class GraphormerAtomFeaturizer(object):
    def __init__(self, atom_data_field='h', atom_types=None, chiral_types=None,
                 hybridization_types=None):
        super(GraphormerAtomFeaturizer, self).__init__()

        self._atom_data_field = atom_data_field

        if atom_types is None:
            atom_types = ['C', 'N', 'O', 'F','Si', 'P', 'S', 'Cl', 'Br', 'I']
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
            # d.atomic_number,
            atom_type_one_hot,
            atom_formal_charge_one_hot, 
            atom_hybridization_one_hot, 
            atom_is_aromatic_one_hot,
            d.atom_total_num_H_one_hot, 
            atom_explicit_valence_one_hot,
            atom_total_bonds,

            atom_partial_charge, 
            atom_mass, 
            # atom_group,
            # atom_period,
        
            ])

    def feat_size(self):
        """Get the feature size.

        Returns
        -------
        int
            Feature size.
        """
        mol = Chem.MolFromSmiles('C')
        feats = self(mol)[self._atom_data_field]

        return feats.shape[-1]

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping atom_data_field as specified in the input argument to the atom
            features, which is a float32 tensor of shape (N, M), N is the number of
            atoms and M is the feature size.
        """
        atom_features = []

        AllChem.ComputeGasteigerCharges(mol)
        num_atoms = mol.GetNumAtoms()

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
            # Features that can be computed directly from RDKit atom instances, which is a list
            feats = self._featurizer(atom)
            atom_features.append(feats)
        atom_features = np.stack(atom_features)

        return {self._atom_data_field: F.zerocopy_from_numpy(atom_features.astype(np.float32))}

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

            ev = torch.cat([ev, torch.zeros(g.num_edges(), 1)], dim=1)
            ev[-num_virtual_nodes * num_real_nodes * 2:, -1] = 1
            g.edata[ek] = ev

    return g

def mol_to_bigraph(mol, add_self_loop=False,
                   node_featurizer=None,
                   edge_featurizer=None,
                   canonical_atom_order=False, ## I changed this
                   explicit_hydrogens=False,
                   num_virtual_nodes=0):

    
    return mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                        node_featurizer, edge_featurizer,
                        canonical_atom_order, explicit_hydrogens, num_virtual_nodes=0)


def import_smiles(file):
    with open(file,'r') as rf:
        r=csv.reader(rf)
        # next(r)
        smiles=[]
        for row in r:
            smiles.append(row[0])
        return smiles

def import_data(file):
    with open(file,'r',  encoding='latin-1') as rf:
        r=csv.reader(rf)
        # next(r)
        data=[]
        for row in r:
            data.append(row)
        return data


# x = import_data(r'/home/weeb/Desktop/Cailum/data/test_full.csv')

# data = QM9(label_keys=["mu"])
# print(data)
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
        # for x1 in atom_type_onehot:
        #     for x2 in formal_charge_onehot:
        #         for x3 in hybridization_onehot:
        #             for x4 in is_aromatic_onehot:
        #                 for x5 in total_num_H_onehot: 
        #                     for x6 in explicit_valence_oesnehot:
        #                         for x7 in total_bonds_onehot:
        #                             key = torch.cat([torch.Tensor(y) for y in [x1, x2, x3, x4, x5, x6, x7]])
        #                             self.one_hotatom_to_int_keys += [key]
        #                             self.one_hotatom_to_int_values += [i]
        #                             i+=1
                                                    
        # # print(self.one_hotatom_to_int_keys[-1])
        # count = 0
        # while count < len(self.one_hotatom_to_int_keys):
        #     h = str(self.one_hotatom_to_int_keys[count])
        #     self.hash_dictatom[h] = self.one_hotatom_to_int_values[count]
        #     count +=1


        # combinatoric bond mapping
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
                        key = torch.cat([torch.Tensor(y) for y in [x1, x3, x4, x5]])
                        self.one_hot_to_int_keys += [key]
                        self.one_hot_to_int_values += [i]
                        i+=1

        count = 0
        while count < len(self.one_hot_to_int_keys):
            h = str(self.one_hot_to_int_keys[count])
            self.hash_dict[h] = self.one_hot_to_int_values[count]
            count +=1


        self.num_classes = 1801
        super().__init__(name='IR Spectra',) #save_dir='/home/weeb/shit/Graphormer/examples/property_prediction/training_dataset/')
        


    def process(self):
        
        self.graphs = []
        self.labels = []
        self.smiles = []

        x = import_data(r'../../sample_data/DFT_spectra.csv')
        count = 0                                                                                                                # Does this use df.load_csv under the hood? If so, should be df.load_csv(path, header=None) 

        count = 0

        print("Loading Data and Converting SMILES to DGL graphs")
        count_outliers = 0

        gnode = True 

        gas_phase = False

        for i in tqdm(x):
            sm = str(i[0]).replace("Q", "#")

            phase = i[2]


            sp = torch.tensor(np.asarray(i[3:], dtype = np.float64), dtype=torch.float64, device=torch.device('cpu')) 

            sp = torch.clip(sp, min = 10e-8)
            sp_nanmask = torch.isnan(sp)
            sp[sp_nanmask] = 0
            sp_sum = torch.sum(sp)

            sp = torch.divide(sp, sp_sum)

            sp[sp_nanmask] = np.NaN

            mol = Chem.MolFromSmiles(sm)

            num_atoms = mol.GetNumAtoms()


            add_self_loop = False
            g = mol_to_bigraph(mol, explicit_hydrogens=False, node_featurizer=GraphormerAtomFeaturizer(), edge_featurizer=CanonicalBondFeaturizer(), add_self_loop=False)                
            count1 = 0
            unif = []
            unifatom = []
            count2 = 0

            while count2 < len(g.ndata['h']): 
                hatom = g.ndata['h'][count2][:]


                unifatom.append(list(np.asarray(hatom)))

                flength = len(list(hatom))
                count2 += 1

            src_list = list(np.full(num_atoms, num_atoms))
            dst_list = list(np.arange(num_atoms))
            features = torch.tensor([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
            total_features = features.repeat(num_atoms, 1)

            if gnode:
                features = torch.tensor([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
                total_features = features.repeat(num_atoms, 1)
                if gas_phase:        
                    g_nm = [104] + list(np.zeros(flength-1))

                    unifatom.append(g_nm)
                    g.add_nodes(1)
                elif phase == "nujol mull":
                    g_nm = [100] + list(np.zeros(flength-1))
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
                elif phase == "dep":
                    g_nm = [105] + list(np.zeros(flength-1))

                    unifatom.append(g_nm)
                    g.add_nodes(1)
                elif phase == "pro":
                    g_nm = [106] + list(np.zeros(flength-1))
                    unifatom.append(g_nm)
                    g.add_nodes(1)
                elif phase == "sod":
                    g_nm = [107] + list(np.zeros(flength-1))

                    unifatom.append(g_nm)
                    g.add_nodes(1)
                    
                else:
                    print("Not A Valid Phase with " + phase)
                    count_outliers +=1
                    continue

                g.ndata['h'] = torch.tensor(unifatom)
                g.add_edges(src_list, dst_list, {'e': total_features})

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


    def __getitem__(self, i):
        return self.graphs[i], self.labels[i], self.smiles[i]

    def __len__(self):
        return len(self.graphs)

@register_dataset("DFT")
def create_customized_dataset():

    directory = '/home/weeb/Desktop/data'
    dataset = IRSpectraD()
    num_graphs = len(dataset)

    train_idx =  np.asarray(np.arange(0, np.floor(num_graphs * 0.9)), dtype=int)
    valid_idx = np.asarray(np.arange(np.floor(num_graphs * 0.9),num_graphs), dtype=int)

    return {
        "dataset": dataset,
        "train_idx":train_idx,#
        "valid_idx": valid_idx,
        "test_idx": None,
        "source": "dgl" 
    }