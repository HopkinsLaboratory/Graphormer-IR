import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
import sys
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import math
import sys
from os import path
import pickle
from tqdm import tqdm
import csv
from rdkit import Chem
from rdkit.Chem import Draw

sys.path.append( path.dirname(   path.dirname( path.abspath(__file__) ) ) )
from pretrain import load_pretrained_model

import logging

def import_data(file):
    with open(file,'r') as rf:
        r=csv.reader(rf)
        next(r)
        data=[]
        for row in r:
            data.append(row)
        return data

def gen_histogram(d_set):
    n, bins, patches = plt.hist(x=d_set, bins=np.arange(0, 1.1, 0.05), color='darkmagenta',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('SIM')
    plt.ylabel('Frequency')
    m = np.mean(d_set)
    std = np.std(d_set)
    title = r"$\mu$ = " + str(np.round(m, 4)) + r"   $\sigma$ = " + str(np.round(std, 4) )
    plt.title(title)
    plt.show()

def make_conv_matrix(frequencies=list(range(400,4002,2)),std_dev=15):
    length=len(frequencies)
    gaussian=[(1/(2*math.pi*std_dev**2)**0.5)*math.exp(-1*((frequencies[i])-frequencies[0])**2/(2*std_dev**2)) for i in range(length)]
    conv_matrix=np.empty([length,length])
    for i in range(length):
        for j in range(length):
            conv_matrix[i,j]=gaussian[abs(i-j)]
    return conv_matrix

def spectral_information_similarity(spectrum1,spectrum2,conv_matrix,smiles,frequencies=list(range(400,4002,2)),threshold=1e-10,std_dev=15):
    save=True

    length = len(spectrum1)
    nan_mask=np.isnan(spectrum1)+np.isnan(spectrum2)

    spectrum1 = np.clip(spectrum1, a_min = 10e-8, a_max = 1)
    spectrum2 = np.clip(spectrum2, a_min = 10e-8, a_max = 1)

    spectrum1[nan_mask]= 0
    spectrum2[nan_mask]= 0 ## nan mask, clipping

    normalize = False 
    if normalize:## This term normalizes the fingerprint region to sum to 1 according to teh bounds n1,n - discussed in paper in more detail why we do this

        spectrum1 /= np.sum(spectrum1)
        spectrum2 /= np.sum(spectrum2)
        n1 = 100
        n = 550

        spectrum2[n1:n] = spectrum2[n1:n] / np.sum(spectrum2[n1:n]) ## 400 cm-1 to 1500 cm-1
        spectrum1[n1:n] = spectrum1[n1:n] / np.sum(spectrum1[n1:n]) ## 400 cm-1 to 1500 cm-1

        spectrum1 /= np.max(spectrum1)
        spectrum2 /= np.max(spectrum2)

    spectrum1=np.expand_dims(spectrum1,axis=0)
    spectrum2=np.expand_dims(spectrum2,axis=0)

    conv1=np.matmul(spectrum1,conv_matrix) ## Gaussian convolution using matrix
    conv2=np.matmul(spectrum2,conv_matrix)
    conv1[0,nan_mask]=np.nan
    conv2[0,nan_mask]=np.nan
    sum1=np.nansum(conv1)
    sum2=np.nansum(conv2)

    norm1=conv1/sum1
    norm2=conv2/sum2

    distance=(norm1*np.log(norm1/norm2)+norm2*np.log(norm2/norm1))
    distance[0,nan_mask] = 0


    sim=1/(1+np.sum(distance)) ## Calculating SIS

    if save:
        simL = np.concatenate((norm1, norm2, [smiles, sim]), axis =None) ## If you want detailed data vs just the similarity score
        return simL
    else:
        return sim

def eval(args, use_pretrained, checkpoint_path=None, logger=None):
    cfg = convert_namespace_to_omegaconf(args) 
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)
    # initialize task
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)

    if use_pretrained:
        model_state = load_pretrained_model(cfg.task.pretrained_model_name)
    else:
        model_state = torch.load(checkpoint_path)["model"]

    model.load_state_dict(
        model_state, strict=True, model_cfg=cfg.model
    )
    del model_state

    model.to(torch.cuda.current_device())
    # load dataset
    split = args.split
    task.load_dataset(split)

    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=cfg.dataset.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=0,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )
    itr = batch_iterator.next_epoch_itr(
        shuffle=False, set_dataset_epoch=False
    )
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
    )

    # infer
    y_pred = []
    y_true = []
    smilesL = []
    with torch.no_grad():
        model.eval()

        for i, sample in enumerate(progress): ## Grabbing batched input, SMILES
            sample = utils.move_to_cuda(sample)

            y = model(**sample["net_input"])
            smilesL.extend(sample["net_input"]['batched_data']['smiles'])
            y = y[:, :].reshape(-1)
            y_pred.extend(y.detach().cpu())
            y_true.extend(sample["target"].detach().cpu().reshape(-1)[:y.shape[0]])
            torch.cuda.empty_cache()


    # save predictions
    # evaluate pretrained models
    if use_pretrained:
        if cfg.task.pretrained_model_name == "pcqm4mv1_graphormer_base":
            evaluator = ogb.lsc.PCQM4MEvaluator()
            input_dict = {'y_pred': y_pred, 'y_true': y_true}
            result_dict = evaluator.eval(input_dict)
            logger.info(f'PCQM4Mv1Evaluator: {result_dict}')
        elif cfg.task.pretrained_model_name == "pcqm4mv2_graphormer_base":
            evaluator = ogb.lsc.PCQM4Mv2Evaluator()
            input_dict = {'y_pred': y_pred, 'y_true': y_true}
            result_dict = evaluator.eval(input_dict)
            logger.info(f'PCQM4Mv2Evaluator: {result_dict}')
    else: 
        if args.metric == "auc":
            auc = roc_auc_score(y_true, y_pred)
            logger.info(f"auc: {auc}")
        elif args.metric == "mae":
            mae = np.mean(np.abs(y_true - y_pred))
            logger.info(f"mae: {mae}")
        else: 
            y_true = np.asarray(y_true, dtype = np.float64)
            y_pred = np.asarray(y_pred, dtype = np.float64)

            dset_size = 1801 ## size of wavenumber vector (400, 4000)
            sim_L = []
            eval_only = False ## If there is no target specra
            save = True

            total = len(y_pred)//dset_size
            conv_matrix = make_conv_matrix(std_dev=15) ## in wavenumber, used for smoothing gaussian convolution

            x = []
            stack = []
            sim_solo = []

            for i in range(total): 
                smiles = smilesL[i]
                subL = []
                y_val_true = np.asarray(y_true[i*dset_size: (i+1)*dset_size], dtype=np.float64)
                y_val_pred = y_pred[i*dset_size: (i+1)*dset_size] ## Grabbing batched data
                y_val_pred /= np.nanmax(y_val_pred) ## normalizing to sum

                if eval_only:
                    conv1=np.matmul(y_val_pred,conv_matrix)
                    sum1=np.nansum(conv1)
                    norm1=conv1/sum1 ## prediction 
                    norm2 = list(np.zeros_like(norm1)) # padded true value
                    norm2.extend(norm1)
                    norm2.extend([smiles, 'eval'])
                    stack.append(norm2) ## if there is no target spectra (testing predictions)
                    continue

                else:
                    sp = y_val_true / np.nanmax(y_val_true)
                    y_val_true /= np.nanmax(y_val_true)
                    
                    sim = spectral_information_similarity(y_val_true, y_val_pred, conv_matrix, smiles) ## this contains smiles, and the normalized vectors

                    sim_L.append(float(sim[-1]))
                    stack.append(sim)

            if save:
                wv = np.arange(400, 4000, 2)
                wv_true = [str(i) + '_true' for i in wv]
                wv_pred = [str(i) + '_pred' for i in wv]
                header = wv_true + wv_pred + ['smiles', 'sim']
                with open('./eval_results.csv', 'w', newline='\n') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',')
                    csvwriter.writerow(header)
                    for row in stack:
                        csvwriter.writerow(row)

            m = np.round(np.mean(sim_L), 5)
            std = np.round(np.std(sim_L), 5)
            print(m, std)
            gen_histogram(sim_L) ## for summary statistics
            logger.info(f"sim average: {m}")

def main():
    parser = options.get_training_parser()
    parser.add_argument(
        "--split",
        type=str,
    )

    parser.add_argument(
        "--metric",
        type=str,
    )
    args = options.parse_args_and_arch(parser, modify_parser=None)
    logger = logging.getLogger(__name__)
    if args.pretrained_model_name != "none":
        eval(args, True, logger=logger)
    elif hasattr(args, "save_dir"):
        for checkpoint_fname in os.listdir(args.save_dir):
            checkpoint_path = Path(args.save_dir) / checkpoint_fname
            print("hi")
            logger.info(f"evaluating checkpoint file {checkpoint_path}")
            eval(args, False, checkpoint_path, logger)

if __name__ == '__main__':
    main()
