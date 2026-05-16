import math
import random

import numpy as np
import torch
from torch import tensor
from torch.distributions import Normal, MultivariateNormal

DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(DTYPE)

# Backward-compatible aliases used by older scripts in this repository.
dtype = DTYPE
device = DEVICE

STD_NORMAL = Normal(
    loc=tensor(0.0, dtype=DTYPE, device=DEVICE),
    scale=tensor(1.0, dtype=DTYPE, device=DEVICE),
)


def as_tensor(x, *, dtype: torch.dtype = DTYPE, device: torch.device = DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)

def set_seed(seed=1234):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def normal_pdf(x, mu, sigma):
    z = (x - mu)/sigma
    return torch.exp(-0.5*z*z) / (sigma*math.sqrt(2*math.pi))

def mvn_pdf(x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor):
    mvn = MultivariateNormal(mean, covariance_matrix=cov)
    return torch.exp(mvn.log_prob(x))
