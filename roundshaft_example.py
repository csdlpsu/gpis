
import numpy as np
from surrogates import Surrogates
from test_functions import Herbie, FourBranch, CantileverBeam, RoundShaftBT
from sampling_torch import CustomDistribution, weighted_kde_sample, fit_and_sample_kde, get_kde_weights, fit_and_sample_kde_
import math
from typing import Optional, Callable
import torch
from torch import Tensor
from kde_test import GaussianKDE
from mis_estimator import MISEstimator, MISEestimator_, ISEestimator_
import os
from mpi4py import MPI

device = torch.device("cpu")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Problem Setup
func = RoundShaftBT()
func_= RoundShaftBT().eval # Herbie().eval
D = func.dim
bounds = func.bounds
# Rare failure threshold for <1e-6 failure probability
t         = 1.
m         = int(10_000 * D) # number of pilot samples for KDE
q         = 5               # batch size per iteration
n_init    = 5
num_iters = 200          # number of sequential updates
# KDE parameters
alpha     = 0.97         # exponent in q_n(x) ∝ p(x) * π^α
h         = 0.2         # bandwidth for Gaussian kernel in KDE
REPS      = 10
# Construct input distribution p(x)

sampler = func.sampler_p_shaft
pdf = func.pdf_p_shaft

px = CustomDistribution(dim=D, pdf=pdf, sampler=sampler)

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

            weights = get_kde_weights(gp, px, pilot_X, bounds, t, alpha=alpha)
            list_of_weights.append(weights)

            # new_X, qx = fit_and_sample_kde(pilot_X, weights, q=5)
            new_X, qx = fit_and_sample_kde_(pilot_X, weights, q=5, train_X=train_X)
            # new_X = weighted_kde_sample(pilot_X, weights, h, q)

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
            fp_, _, _, _ = MISEestimator_(proposals, samples_X, samples_Y, failure_fn)
            # fp_, _, _, _ = ISEestimator_(proposals, samples_X, samples_Y, failure_fn)
            fp.append(fp_)

            print(f"REP {REP} Iteration {it}: total training points = {train_X.shape[0]} fp {fp_:1.10f}", flush=True)

            try:
                filename = f"results/roundshaft_new/REP_{REP}.npy"
                np.save(filename, np.array(fp))
            except FileNotFoundError:
                directory_name = "results/roundshaft_new"
                filename = directory_name + "/" + f"REP_{REP}.npy"
                os.mkdir(directory_name)
