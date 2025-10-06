import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize, standardize
import math

class Surrogates():

    def __init__(self, train_x, train_y, bounds):
        self.train_x = train_x
        self.train_y = train_y
        self.bounds = torch.tensor(bounds.T, dtype=train_y.dtype) # BoTorch expects bounds to be 2xd
        return

    def fit_gp(self):
        gp = SingleTaskGP(normalize(self.train_x, self.bounds), self.train_y)
        # gp = SingleTaskGP(self.train_x, self.train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()
        return gp