from re import T
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
import math
from tqdm import trange
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Mapping, Sequence, Tuple
from numpy import mod
import torch
from torch import Tensor
import torch.nn.functional as F
from fairseq.dataclass.configs import FairseqDataclass
import matplotlib.pyplot as plt


from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("sid", dataclass=FairseqDataclass)
class SID(FairseqCriterion):
    """
    Implementation for the binary log loss used in graphormer model training.
    """

    def __init__(self, task):
        super().__init__(task)
        self.threshold = 1e-8 
        self.eps = 1e-8
        self.torch_device = 'cuda'

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]
        values = model(**sample["net_input"])


        values = torch.clip(values, min = 10e-8, max = 500)
        label = sample['target'] 
        label = torch.clip(label, min = 10e-8, max = 500) ## clipping values to avoid negative logarithms
        values = values.squeeze(1)
        loss = self.sid(values, label)

        logging_output = {
            "loss": loss,
            "sample_size": sample_size,
            "ntokens": 1,
            "nsentences": sample_size,
            "ncorrect": 0,
        }

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / sample_size, sample_size, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed():
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


    def sid(self, model_spectra: torch.tensor, target_spectra: torch.tensor):
        # normalize the model spectra before comparison
        model_spectra[model_spectra <= 10**-6] = 10**-6 ## normalizing to avoid underflow issues
        target_spectra[target_spectra <= 10**-6] = 10**-6
            
        nan_mask=torch.isnan(target_spectra)+torch.isnan(model_spectra)
        nan_mask=nan_mask.to(device=self.torch_device) ## determining nans in both the target and model spectra

        zero_sub=torch.zeros_like(target_spectra,device=self.torch_device, dtype=torch.float16)
        sum_model_spectra = torch.sum(model_spectra,axis=1)

        sum_model_spectra = torch.unsqueeze(sum_model_spectra,axis=1)
        model_spectra = torch.div(model_spectra,sum_model_spectra) ## normalizing to sum to one (targets are already normalized this way)

        if not isinstance(target_spectra,torch.Tensor):
            target_spectra = torch.tensor(target_spectra)
        target_spectra = target_spectra.to(self.torch_device)


        target_spectra[nan_mask] = 0 ## removing nans
        model_spectra[nan_mask] = 0

        target_spectra[target_spectra <= 10**-8] = 10**-8
        model_spectra[model_spectra <= 10**-8] = 10**-8 ## clipping again to avoid underflow issues where very small values go to zero

        loss = (torch.mul(torch.log(torch.div(model_spectra,target_spectra)),model_spectra) + torch.mul(torch.log(torch.div(target_spectra,model_spectra)),target_spectra)) ## SID equation

        loss[nan_mask]=0 ## eliminating all loss values where there are NaNs
        loss = torch.nansum(loss,axis=1) ## Tryed both mean and sum, mean works better
        loss = torch.nansum(loss) * 1000 / loss.shape[0] ## accounts for batched data, arbitrary scaling factor to improve learning (Found to help with early gradients)

        return loss