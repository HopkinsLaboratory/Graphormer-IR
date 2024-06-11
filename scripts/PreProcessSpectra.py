import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os
import csv
import math
from time import time
from time import sleep
from scipy import ndimage
from scipy import sparse
from scipy.sparse.linalg import spsolve

from rdkit import Chem

######################################################################################################

## | WARNING this script uses interp1d with an extrapolation for values outside the interpolation range
## | extrapolation can be erroneous, especially with noisy for non-monotonic data.

fin = 'IRMPDspectraRaw.csv'
maindir = r'C:\Users\Sideshow Bob\Desktop\IRMPD_Spectra\ExpFilesForHMDBv20221128\output'
fout = '2023_08_28_interpolated_IRMPD_data_lam_3e5_lam1_1_10e-4_p_0p04_ROUND3.csv'

show_graphs = False # depicts pyplot for each .csv  ## NOTE: The interpolated absorbance are renormalized, hence look larger when nujol mull is present

upper = 4000 # upper wavenumber bound 
lower = 400 # lower wavenumber bound
step = 2  # step size 

######################################################################################################

def baseline_ials(xy, lam, lam1, p, name, plot=True):
    # S. He, W. Zhang, L. Liu, Y. Huang, J. He, W. Xie, P. Wu, and C. Du, Anal. Methods 6, 4402 (2014).
    # code idea from     # https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    '''xy is a Nx2 array, 10^2<lam<10^6, lam1<10^-4 (smoothness param) and p the asymmetry param (p<0.1)'''
    # xy[:] = xy[:, ~np.isnan(xy[:,1])]
    # xy = xy[~np.isnan(xy).any(axis=1)]
    y = xy[:,1]
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D1 = sparse.diags([1,-1],[0,-1], shape=(L,L-1))
    w = np.ones(L)
    w0 = np.ones(L)*2.
    wthresh = 1e-4
    count = 0
    while np.linalg.norm((w-w0)/w0) > wthresh: # RMS threshold 
        count += 1
        w0 = np.copy(w) # safe "old" w
        W = sparse.spdiags(w, 0, L, L)
        Z = W.dot(W.transpose()) + lam1*D1.dot(D1.transpose()) + lam*D.dot(D.transpose())
        Z1 = W.dot(W.transpose()) + lam1*D1.dot(D1.transpose())
        z = spsolve(Z, Z1.dot(y))
        w = p * (y > z) + (1-p) * (y < z)
        if count > 100: break
    #print(count)
    
    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(xy[:,0],y,'k--',label='raw')
        # plt.plot(xy[:,0],z,'r--',label='baseline')
        plt.xlabel('wavenumber / $cm^{-1}$')
        plt.ylabel('absorbance')
        # plt.title(name + ' baseline correction')
        # plt.ylim(np.min(z)*0.9,np.max(z)*1.2)
        # plt.show()
    
    return np.vstack([xy[:,0],z]).T

######################################################################################################


start = time()

path = f'{maindir}/{fin}'

try:
    outdir = os.mkdir(f'{maindir}\output')
except FileExistsError:
    pass # if directory already exists, do nothing

outdir = f'{maindir}\output' 

output = os.path.join(outdir,fout)

def make_dataframe(path):
    df = pd.read_csv(f'{path}', delimiter=',',names=['smiles_str','state','wvnum','trans'])
    return df

def make_number(str):
    if str == 'nan' or str == '':
        str = np.nan
    else:
        str = float(str)
    return str

def get_key(val, dict):
    for key, value in dict.items():
        if val == value:
            return key

states = []
state_dic = { ## Used for consolidating labels for spectral medium
            "nujol mull":["mull", "nujol", "mineral", "oil", "suspension"], 
            "CCl4": ["CCl4",],
            "CHCl3": ["CHCL3"],
            "Liquid Film": ["Liquid", "Film",  "NEAT"], 
            "KBr": ['KBR', 'SOLID (BETWEEN SALTS)', 'SOLID (PELLET)', 'Solid (powder)', 'SOLID', 'solid'],
            "KCl": ["KCL"], 
            "gas": ["Gas", "Vapor"],
            'dep':['deprotonated'],
            'pro':["protonated"],
            'sod':['sodiated'],
            }


##                   [H, C, N, O, F, Si, P,  S,  Cl, Br, I]
allowed_atomic_num = [1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]

def is_valid_mol(smiles_str):
    smiles_str = smiles_str.replace('Q', '#')
    molecule = Chem.MolFromSmiles(smiles_str)
    if molecule is None:
        return False        
    else:
        if Chem.GetFormalCharge(molecule) != 0:
            return False
        else:
            for atom in molecule.GetAtoms():
                atomic_num = atom.GetAtomicNum()
                if atomic_num not in allowed_atomic_num:
                    return False
    return True

### Script for pre-procssing the IR data - baseline correction, interpolation, and normalization. Also filters for valid molecules, etc. 
def process_dataframe(df):
    with open(output, 'w', newline='') as f: # optional 'a' for appending to end of pre-existing csv and 'w' for rewriting from beginning 
        writer = csv.writer(f, delimiter=',')
        c= 0

        for row in df.to_numpy(): 
            if c == 0:
                c+=1
                continue
            smiles_str = row[0]
            # print(smiles_str)

            species = row[2]
            state = row[1]

            
            for val in state_dic.values():
                for s in val:
                    if s.lower() in state.lower():
                        state = get_key(val,state_dic)
                        break
            
            xs = list(map(lambda x: make_number(x), row[3].strip('][').split(',')))
            ys = list(map(lambda x: make_number(x), row[4].strip('][').split(',')))

            coords_arr = np.array([xs,ys])
            coords = pd.DataFrame(coords_arr).to_numpy().T
            coords = coords[~np.isnan(coords).any(axis=1)].T

            ## only include x values in range [upper,lower], with 5 * step padding to aid interpolation
            mask = (upper+step*5 >= coords[0,:]) & (coords[0,:] >= lower-step*5)
            coords = coords[:,mask]
                
            ## make an array of x- and y-coordinates
            xs = coords[0,:]
            ys = coords[1,:]
            ys /= np.nanmax(ys)

            expdata = np.vstack([xs,ys]).T
            

            baseline_ys = baseline_ials(expdata, lam=3e5, lam1=10e-4, p = 0.04, name= "IR", plot = True)[:,1] # TODO Tinker with settings

            ## Apply baseline correction
            ys = ys - baseline_ys

            ## Set min to zero and renormalize
            ys += -np.nanmin(ys)
            ys /= np.nanmax(ys)

            b1 = xs[0]
            b2 = xs[-1]

            if b1 > b2:
                temp = b1
                b1 = b2
                b2 = temp

            b1 = math.ceil(b1 / 2) * 2 
            b2 = math.floor(b2 / 2) * 2

            if b1 > 400:
                pad1 = (b1 - 400) // 2 
            else:
                b1 = 400
                pad1 = 0
            if b2 < 4000:
                pad2 = (4000 - b2) // 2
            else:
                b2 = 4000
                pad2 = 0

            f = interpolate.interp1d(xs,ys)#,fill_value="extrapolate")

            ## make new x-coordinates satisfying the desired bounds and step size
            xsnew = list(np.arange(b1,b2+step,step))

            ## use the interpolation function to determine new y-coordinates   
            ysnew = list(f(xsnew))


            ## combine the new arrays 
            data = np.column_stack((xsnew,ysnew))
            
            ['KBr' 'nujol mull' 'liquid film' 'CCl4' 'gas' 'CHCl3' 'KCl']
            if state == 'CCl4':
                mask = np.ma.masked_inside(data[:,0], 696, 850)
                mask = np.ma.masked_inside(mask, 1500, 1600).mask
            elif state == 'nujol mull':
                mask = np.ma.masked_inside(data[:,0], 2750, 3000).mask
            else:
                mask = np.full_like(data[:,0], False, dtype=bool)          

                
            interp_ys = list(np.pad(np.ma.masked_array(data[:,1], mask=mask).filled(np.nan), (pad1,pad2), mode='constant', constant_values=(np.nan, np.nan)))
            if len(interp_ys) != 1801:
                print(smiles_str)
                exit() 

            interp_ys += -np.nanmin(interp_ys)
            interp_ys /= np.nanmax(interp_ys)

            temp = np.arange(400,4002,2)
            plt.plot(temp,interp_ys,'k',label='final') #! HERE

            if show_graphs:
                plt.title(f'IR baseline correction: {smiles_str} {state}')
                plt.grid()
                plt.legend()
                plt.show()

            plt.close()

            mol_data = [species,smiles_str,state]
            # print(len(data))#

            print(len(interp_ys))
            mol_data.extend(interp_ys)
            # print(len(data))
            # print(len(finaldata))
            # print(smiles_str)
            print(state, "FINAL STATE")
            writer.writerow(mol_data)
            print(time() - start)


path = f'{maindir}\{fin}'
path = r'C:\Users\Sideshow Bob\Desktop\IR_Spectra\IR_df2.csv'

df = pd.read_csv(path, delimiter=',',names=['ionType','id','state','xs','ys'])
df['id']=df['id'].astype(str)
df['state']=df['state'].astype(str)

outpath = f'{maindir}/IRMPDf_Jan24.parquet'

df.to_parquet(outpath)

datapath = f'{outpath}'
interpdf = pd.read_parquet(datapath, engine='pyarrow')
print(interpdf.head())
process_dataframe(interpdf)