
import numpy as np
from surrogates import Surrogates
from test_functions import Herbie
from sampling_torch import CustomDistribution, fit_and_sample_kde, get_kde_weights
from typing import Optional, Callable
import torch
from torch import Tensor
from mis_estimator import MISEestimator_, ISEestimator_
from mpi4py import MPI

device = torch.device("cpu")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Problem Setup
func = Herbie()
func_= Herbie().eval # Herbie().eval
D = func.dim
bounds = func.bounds
# Rare failure threshold for <1e-6 failure probability
t         = 2.1
m         = int(10_000 * D) # number of pilot samples for KDE
q         = 5               # batch size per iteration
n_init    = 5
num_iters = 200          # number of sequential updates
# KDE parameters
alpha     = 0.97        # exponent in q_n(x) ∝ p(x) * π^α
h         = 0.2         # bandwidth for Gaussian kernel in KDE
REPS      = 1
estimator = "mis"
# Construct input distribution p(x)

# Uniform distribution
def pdf_unif(x):
    volume = (bounds[:, 1] - bounds[:, 0]).prod()  # Domain volume

    return 1.0 / volume


def make_uniform_box_sampler(
        bounds: Tensor,
        *,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
) -> Callable[[int, Optional[torch.Generator]], Tensor]:
    """
    Factory: returns a sampler(n, generator) that draws n samples uniformly
    from the hyper-rectangle defined by `bounds` (D x 2).
    """
    b = torch.as_tensor(bounds, dtype=dtype, device=device)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"`bounds` must be (D,2), got {tuple(b.shape)}")
    lo = b[:, 0].reshape(1, -1)  # (1,D)
    hi = b[:, 1].reshape(1, -1)  # (1,D)
    D = b.shape[0]

    def sampler(n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        z = torch.rand((n, D), dtype=dtype, device=device, generator=generator)
        return lo + z * (hi - lo)

    return sampler

sampler = make_uniform_box_sampler(bounds)

px = CustomDistribution(dim=D, pdf=pdf_unif, sampler=sampler)


for REP in range(REPS):

    if REP % size == rank:

        np.random.seed(111 + REP)
        torch.manual_seed(111 + REP)

        # Generate pilot samples for KDE (uniform p(x))
        pilot_X = px.sample(m)
        # Initial training design
        train_X = px.sample(n_init)
        train_Y = func_(train_X).reshape(-1, 1)

        # Sequential Loop: fit GP, compute π_n, KDE-sample, update GP
        list_of_weights = []
        fp = []
        samples_X = [train_X]
        samples_Y = [train_Y]
        proposals = [px]

        for it in range(1, num_iters + 1):
            # Fit GP surrogate
            gp = Surrogates(train_X, train_Y, bounds).fit_gp()
            gp.eval()

            weights = get_kde_weights(gp, px, pilot_X, train_X, bounds, t, alpha=0.97)
            list_of_weights.append(weights)

            new_X, qx = fit_and_sample_kde(pilot_X, weights, q=5)

            # Clip to domain
            for d in range(D):
                new_X[:, d] = torch.clip(new_X[:, d], bounds[d, 0], bounds[d, 1])

            new_Y = func_(new_X).reshape(-1, 1)

            # Update training data
            train_X = torch.cat([train_X, new_X], dim=0)
            train_Y = torch.cat([train_Y, new_Y], dim=0)

            # failure prob
            samples_X.append(new_X)
            samples_Y.append(new_Y)
            proposals.append(qx)
            def failure_fn(X: torch.Tensor) -> torch.Tensor:
                return (X.view(-1) > t)
            if estimator.lower() == "mis":
                fp_, _, _, _ = MISEestimator_(proposals, samples_X, samples_Y, failure_fn)
            elif estimator.lower() == "is":
                fp_, _, _, _ = ISEestimator_(proposals, samples_X, samples_Y, failure_fn)
            fp.append(fp_)

            print(f"REP {REP} Iteration {it}: total training points = {train_X.shape[0]} fp {fp_}", flush=True)
