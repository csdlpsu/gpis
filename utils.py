import math, functools
import torch
from torch import tensor
from torch.distributions import Normal, MultivariateNormal
from typing import Callable, Dict, Tuple, List

# BoTorch / GPyTorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.double)
device = torch.device("cpu")

STD_NORMAL = Normal(loc=tensor(0.0, dtype=torch.double), scale=tensor(1.0, dtype=torch.double))

def set_seed(seed=1234):
    torch.manual_seed(seed)
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)

def normal_pdf(x, mu, sigma):
    z = (x - mu)/sigma
    return torch.exp(-0.5*z*z) / (sigma*math.sqrt(2*math.pi))

def mvn_pdf(x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor):
    mvn = MultivariateNormal(mean, covariance_matrix=cov)
    return torch.exp(mvn.log_prob(x))
