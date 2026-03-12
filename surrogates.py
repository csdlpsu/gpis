import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize, standardize
from botorch.models.transforms import Normalize, Standardize
import math

# class Surrogates():

#     def __init__(self, train_x, train_y, bounds):
#         self.train_x = train_x
#         self.train_y = train_y
#         self.bounds = bounds # BoTorch expects bounds to be 2xd
#         return

#     def fit_gp(self):
#         gp = SingleTaskGP(normalize(self.train_x, self.bounds), self.train_y)
#         mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
#         fit_gpytorch_mll(mll)
#         # gp.eval()
#         return gp


class Surrogates:
    def __init__(self, train_x, train_y, bounds):
        self.train_x = train_x
        self.train_y = train_y
        self.bounds = bounds

    def fit_gp(self):
        model = SingleTaskGP(
            self.train_x,
            self.train_y,
            input_transform=Normalize(d=self.train_x.shape[-1], bounds=self.bounds),
            outcome_transform=Standardize(m=self.train_y.shape[-1]),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        model.likelihood.eval()
        return model