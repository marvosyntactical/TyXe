"""
A collection of metrics useful for evaluating tyxe bnns.
"""

import torch
import torch.nn.functional as F
import torch.distributions.utils as dist_utils
import torch.distributions as torchdist
from torch.distributions import transforms
from torch import Tensor

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from likelihoods import *


def rmse(x: Tensor, y: Tensor) -> Tensor:
    raise NotImplemented

def aleatoric_uncertainty(x: Tensor, y: Tensor) -> Tensor:
    raise NotImplemented

def epistemic_uncertainty(x: Tensor, y: Tensor) -> Tensor:
    raise NotImplemented

def normed_elbo(x: Tensor, y: Tensor) -> Tensor:
    raise NotImplemented






