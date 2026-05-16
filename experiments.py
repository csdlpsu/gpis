r"""
Example usage: 

python experiments.py --testfunction roundshaft 
--wd expt 
--t 1.0 
--m 100000 
--q 10 
--n_init 20 
--num_iters 5 
--alpha 0.97 
--h 0.2 
--REPS 1 
--estimator mfmis

"""
import numpy as np
from surrogates import Surrogates
from test_functions import TestFunctions
from sampling_torch import CustomDistribution, fit_and_sample_kde, get_kde_weights
import torch
from mis_estimator import MISEestimator_, ISEestimator_, MISEestimatorMF
import os
from mpi4py import MPI
import argparse
from utils import DEVICE, DTYPE, set_seed

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))

device = DEVICE
dtype = DTYPE
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


parser = argparse.ArgumentParser(description="Run KDE-AIS experiment")

parser.add_argument("--testfunction", type=str, default="herbie",
                    choices=["herbie", "fourbranch", "cantilever", "roundshaft"],
                    help="test function")

parser.add_argument("--wd", type=str, default="expt", help="working directory")

parser.add_argument("--t", type=float, default=1.0)

parser.add_argument("--m", type=int, default=100_000,
                    help="number of pilot samples for KDE")

parser.add_argument("--q", type=int, default=10,
                    help="batch size per iteration")

parser.add_argument("--n_init", type=int, default=20,
                    help="number of seed points for GP")

parser.add_argument("--num_iters", type=int, default=50,
                    help="number of sequential updates")

# KDE parameters
parser.add_argument("--alpha", type=float, default=0.97,
                    help="exponent in q_n(x) ∝ p(x) * π^α")

parser.add_argument("--h", type=float, default=0.2,
                    help="bandwidth for Gaussian KDE")

parser.add_argument("--REPS", type=int, default=1)

parser.add_argument("--estimator", type=str, default="mis",
                    choices=["mis", "is", "mfmis"],
                    help="importance estimator type")

parser.add_argument("--budget", type=int, default=100,
                    help="total number of evaluations")

parser.add_argument("--two_stage", type=lambda x: x.lower() == 'true')

args = parser.parse_args()

# Access values
tf = TestFunctions(args.testfunction, dtype=dtype, device=device)
func = tf.func
func_= tf.func_
D = func.dim
bounds = func.bounds.T

t = args.t
m = int(np.maximum(args.m, int(D * 50_000)))
q = args.q
n_init = args.n_init
num_iters = args.num_iters
alpha = args.alpha
h = args.h
REPS = args.REPS
estimator = args.estimator
two_stage = args.two_stage
budget = args.budget

sampler = func.sampler
pdf = func.pdf
px = CustomDistribution(dim=D, pdf=pdf, sampler=sampler)


def clip_to_bounds(X, bounds):
    mask = ((X >= bounds[0]) & (X <= bounds[1])).all(dim=1)
    return X[mask]
    
def failure_fn(X: torch.Tensor) -> torch.Tensor:
    return (X.view(-1) > t)

outdir = os.path.join("results", args.wd)
os.makedirs(outdir, exist_ok=True)

for REP in range(REPS):

    if REP % size == rank:

        set_seed(111 + REP)
        
        # Generate pilot samples for KDE (uniform p(x))
        pilot_X = clip_to_bounds(px.sample(m * 100).to(dtype=dtype, device=device), bounds.to(dtype=dtype, device=device))[:m, ...]
        
        # Initial training design
        train_X = clip_to_bounds(px.sample(n_init * 100).to(dtype=dtype, device=device), bounds.to(dtype=dtype, device=device))[:n_init, ...]
        train_Y = func_(train_X).reshape(-1, 1)
        
        # Sequential Loop: fit GP, compute π_n, KDE-sample, update GP
        list_of_weights = []
        fp = []
        samples_X = [train_X]
        samples_Y = [train_Y]
        proposals = [px]        
        
        for it in range(1, num_iters + 1):
            # Fit GP surrogate
            gp = Surrogates(train_X, train_Y, bounds, dtype=dtype, device=device).fit_gp()
        
            weights = get_kde_weights(gp, px, pilot_X, train_X, bounds, t, alpha=0.9)
            list_of_weights.append(weights)
        
            new_X, qx = fit_and_sample_kde(pilot_X, weights, q=500) # 500 choose a large number, but keep the first q
            new_X = clip_to_bounds(new_X.to(dtype=dtype, device=device), bounds.to(dtype=dtype, device=device))[:q, ...]        
            new_Y = func_(new_X).reshape(-1, 1)
                        
            # Update training data
            train_X = torch.cat([train_X, new_X], dim=0)
            train_Y = torch.cat([train_Y, new_Y], dim=0)
                    
            samples_X.append(new_X)
            samples_Y.append(new_Y)
            proposals.append(qx)

            # failure prob
            if not two_stage or two_stage is None:
                if estimator.lower() == "mis":
                    fp_, _, _, _ = MISEestimator_(proposals, samples_X, samples_Y, failure_fn)
                elif estimator.lower() == "is":
                    fp_, _, _, _ = ISEestimator_(proposals, samples_X, samples_Y, failure_fn)
                elif estimator.lower() == "mfmis":
                    fp_ = MISEestimatorMF(gp, proposals, samples_X, samples_Y, failure_fn, bounds, 
                                          MC_size=2_00_000, clip_to_bounds=clip_to_bounds)
            else:
                n_estimator_samples = budget - n_init                
                samples = torch.as_tensor(qx.sample(n_estimator_samples * 100), dtype=dtype, device=device)                
                samples_clipped = clip_to_bounds(samples, bounds.to(dtype=dtype, device=device))[:n_estimator_samples, ...]
                samples_X.append(samples_clipped)
                samples_Y.append(func_(samples_clipped).reshape(-1, 1) )
                print(f"computing two-stage failure probability ...", flush=True)
                fp_, _, _, _ = ISEestimator_(proposals, samples_X, samples_Y, failure_fn)
                print(f"... done.", flush=True)
                it = num_iters + 1    
        
            print(f"REP {REP} Iteration {it}: newx shape {new_X.shape}, total training points = {train_X.shape[0]} fp {fp_}", flush=True)

            # save to file
            fp_val = fp_.item() if torch.is_tensor(fp_) else float(fp_)
            fp.append(fp_val)

            np.save(os.path.join(outdir, f"FP_{REP}.npy"),
                    np.array(fp, dtype=np.float64))
            np.save(os.path.join(outdir, f"X_{REP}.npy"),
                    train_X.detach().cpu().numpy())
            np.save(os.path.join(outdir, f"Y_{REP}.npy"),
                    train_Y.detach().cpu().numpy())                
