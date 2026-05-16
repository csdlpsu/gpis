from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from utils import DEVICE, DTYPE


class Surrogates:
    def __init__(self, train_x, train_y, bounds, *, dtype=DTYPE, device=DEVICE):
        self.dtype = dtype
        self.device = train_x.device if hasattr(train_x, "device") else device
        self.train_x = train_x.to(dtype=dtype, device=self.device)
        self.train_y = train_y.to(dtype=dtype, device=self.device)
        bounds = bounds.to(dtype=dtype, device=self.device)
        self.bounds = bounds.T if bounds.shape == (self.train_x.shape[-1], 2) else bounds

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
