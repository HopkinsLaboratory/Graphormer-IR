import pandas as pd
import numpy as np
import os

def make_split_preserve_unique_labels(df, frac1):
    mask = np.random.rand(len(df)) < frac1
    
    df1 = df[mask]
    df2 = df[~mask]

    ind = df2['smiles'].isin(df1['smiles'])
    from1to2 = df2[ind]
    frames = [df1,from1to2]

    new_df1 = pd.concat(frames)
    new_df2 = df2[~ind]  

    return new_df1, new_df2

def get_indices(df, frac_div2, inc, goal):
    ## generates indices for training and testing dataframes, while obeying SMILES restriction

    df1, df2 = make_split_preserve_unique_labels(df, frac_div2)
    total = df.shape[0]
    frac = df1.shape[0] / total
    while frac_div2 < goal:
        fromdf2Todf1, df2 = make_split_preserve_unique_labels(df2, inc)

        frames = [df1,fromdf2Todf1]
        df1 = pd.concat(frames)

        curr_split = df1.shape[0]
        frac_div2 = curr_split / total

    return df1, df1.index.to_numpy(), df2, df2.index.to_numpy()

def encode_phase(phase: str):
    ## generate one hot encoding for phase
    phase_dict = {  'n':'1',
                    'liquid film':'0,1,0,0,0',
                    'KBr':'0,0,1,0,0',
                    'KCl':'0,0,1,0,0',
                    'nujol mull':'0,0,0,1,0',
                    'CCl4':'0,0,0,0,1'}
    for ph, encoding in phase_dict.items():
        if phase == ph:
            phase = phase.replace(phase, encoding)
    return phase


def translate_to_chemprop(df):
    ## translates dataframe to chemprop input format
    phase_header = ['gas','liquid','KBr','nujolmull','CCl4']

    smiles_phase_df = df[['smiles','phase']].copy()
    phase_df = df[['phase']].copy()
    spectra_df = df.drop(['phase'], axis=1)

    phase_df['phase'] = df['phase'].apply(lambda x: encode_phase(x)) 
    phase_df['phase_list'] = phase_df['phase'].str.split(',')
    phase_df = pd.DataFrame(phase_df['phase_list'].tolist(), columns=phase_header)

    return smiles_phase_df, phase_df, spectra_df

def gen_splits(file: str, directory: str, num_splits: int):
    ## Generates train/test/spits for data in the format of the csv file. Makes sure that there are no repeat smiles in different splits

    if os.path.exists(f'{directory}/splits_{file[file.find(r"/") + 1:-4]}'):
        pass
    else:
        os.makedirs(f'{directory}/splits_{file[file.find(r"/") + 1:-4]}')

    outpath = f'{directory}/splits_{file[file.find(r"/") + 1:-4]}'

    df = pd.read_csv(file)

    for i in range(num_splits):
        if os.path.exists(f'{outpath}/split_{i + 1}/graphormer'):
            pass
        else:
            os.makedirs(f'{outpath}/split_{i + 1}/graphormer')

        if os.path.exists(f'{outpath}/split_{i + 1}/chemprop'):
            pass
        else:
            os.makedirs(f'{outpath}/split_{i + 1}/chemprop')
        
        if os.path.exists(f'{outpath}/split_{i + 1}/chemprop/debugging'):
            pass
        else:
            os.makedirs(f'{outpath}/split_{i + 1}/chemprop/debugging')

        train_df, train_idx, test_df, test_idx = get_indices(df, 0.01, 0.005, 0.9)
        print('################################################################')
        print(f'Split Number {i + 1}')
        print("Fraction of testing / total:", test_df.shape[0] / df.shape[0])
        print("Fraction of training / total:", train_df.shape[0] / df.shape[0])
        print("Sum of test and train fractions:", test_df.shape[0] / df.shape[0] + train_df.shape[0] / df.shape[0])

        new_valid_df, valid_idx, new_train_df, train_idx = get_indices(train_df, 0.01, 0.001, 0.1)  # dataframe, initial split, increment, goal split

        rand_valid_idx = np.random.choice(valid_idx, size=len(valid_idx), replace=False)
        rand_train_idx = np.random.choice(train_idx, size=len(train_idx), replace=False)

        print("Fraction of valid / training:", rand_valid_idx.shape[0] / train_df.shape[0])
        print("Fraction of valid / total", new_valid_df.shape[0] / df.shape[0])
        print("Fraction of training - valid / total", new_train_df.shape[0] / df.shape[0])

        graphormer_outpath = f'{outpath}/split_{i + 1}/graphormer'
        chemprop_outpath = f'{outpath}/split_{i + 1}/chemprop' 

        pd.DataFrame(rand_valid_idx).transpose().to_csv(f'{graphormer_outpath}/valid_indices.csv', header=None, index=False)
        pd.DataFrame(rand_train_idx).transpose().to_csv(f'{graphormer_outpath}/train_indices.csv', header=None, index=False)
        
        test_df.to_csv(f'{graphormer_outpath}/testing_dataset.csv', header=None, index=False, na_rep='nan')
        train_df.to_csv(f'{graphormer_outpath}/training_dataset.csv', header=None, index=False, na_rep='nan')

        test_smiles_phase_df, test_phase_df, test_spectra_df = translate_to_chemprop(test_df)
        train_smiles_phase_df, train_phase_df, train_spectra_df = translate_to_chemprop(new_train_df)
        valid_smiles_phase_df, valid_phase_df, valid_spectra_df = translate_to_chemprop(new_valid_df)
        
        test_spectra_df.to_csv(f'{chemprop_outpath}/testing_dataset.csv', header=True, index=False, na_rep='nan')
        test_phase_df.to_csv(f'{chemprop_outpath}/testing_phase.csv', header=True, index=False)
        test_smiles_phase_df.to_csv(f'{chemprop_outpath}/debugging/test_smiles_phase.csv', header=None, index=False)
        
        train_spectra_df.to_csv(f'{chemprop_outpath}/training_dataset.csv', header=True, index=False, na_rep='nan')
        train_phase_df.to_csv(f'{chemprop_outpath}/training_phase.csv', header=True, index=False)
        train_smiles_phase_df.to_csv(f'{chemprop_outpath}/debugging/train_smiles_phase.csv', header=None, index=False)
        
        valid_spectra_df.to_csv(f'{chemprop_outpath}/valid_dataset.csv', header=True, index=False, na_rep='nan')
        valid_phase_df.to_csv(f'{chemprop_outpath}/valid_phase.csv', header=True, index=False)
        valid_smiles_phase_df.to_csv(f'{chemprop_outpath}/debugging/valid_smiles_phase.csv', header=None, index=False)

gen_splits('/home/cmkstien/Desktop/Cailum/chemprop_data_ourspectra/Data_Patrice/make_easy_splits/2023_02_04_interpolated_data_lam_3e5_lam1_10e-4_p_0p04.csv','make_splits_auto', 1)
