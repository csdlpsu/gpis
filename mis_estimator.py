
"""
Modular Multiple Importance Sampling (MIS) estimator for failure probability.

Given M proposal densities {q_j}_{j=1,..,M} and N = \sum_j n_j samples,
with α_j = n_j / N (or user-specified mixture weights), the MIS (DM) estimator is:

  \hat{P}_F^DM = (1/N) * sum_{i=1}^N  1_{F}(x_i) * p(x_i) / ψ(x_i),

where ψ(x) = Σ_{j=1}^M α_j q_j(x) is the mixture of proposals,
and p is the *original* input distribution under which we want the failure probability.

This estimator is unbiased and typically has lower variance than single-proposal IS.

We also provide the self-normalized estimator:

  \tilde{P}_F^{SNIS} = [ Σ 1_F(x_i) w_i ] / [ Σ w_i ],  w_i = p(x_i)/ψ(x_i),

which is consistent (asymptotically unbiased). Both are supported.

"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Dict, Union, Tuple
import math

import torch
from torch import Tensor

# Optional: wrap your numpy KDE (from kde_test import GaussianKDE) for convenience.
try:
    import numpy as np
    from kde_test import GaussianKDE
    _HAS_NUMPY_KDE = True
except Exception:
    _HAS_NUMPY_KDE = False


# ------------------------------
# Utility helpers
# ------------------------------

def _as_tensor(x: Union[Tensor, float], *, dtype: torch.dtype, device: torch.device) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)

def _logsumexp(a: Tensor, dim: int = -1) -> Tensor:
    """Stable log-sum-exp."""
    a_max, _ = torch.max(a, dim=dim, keepdim=True)
    out = a_max + torch.log(torch.sum(torch.exp(a - a_max), dim=dim, keepdim=True))
    return out.squeeze(dim)

def _check_shapes(X: Tensor, name: str = "X") -> None:
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n,d); got shape {tuple(X.shape)}")


# ------------------------------
# Base/Uniform proposals for convenience
# ------------------------------

@dataclass
class Box:
    """Axis-aligned box: bounds[d, 2] with [low, high] per dim."""
    bounds: Tensor  # (d, 2)

    def __post_init__(self):
        if not isinstance(self.bounds, Tensor):
            self.bounds = torch.as_tensor(self.bounds, dtype=torch.float64)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must be (d, 2) tensor.")
        if torch.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError("Each upper bound must be > lower bound.")


class UniformBox:
    """
    Uniform distribution on an axis-aligned box.
    Provides a Torch-first interface: logpdf(X) and sample(n).
    """
    def __init__(self, bounds: Tensor, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None):
        if not isinstance(bounds, Tensor):
            bounds = torch.as_tensor(bounds, dtype=dtype, device=device)
        self.bounds = bounds.to(dtype=dtype, device=device)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must be (d,2)")
        self.d = self.bounds.shape[0]
        self.dtype = dtype
        self.device = self.bounds.device if device is None else device
        self.volume = torch.prod(self.bounds[:, 1] - self.bounds[:, 0]).to(dtype=dtype)
        self._logpdf_inside = -torch.log(self.volume)

    def logpdf(self, X: Tensor) -> Tensor:
        X = X.to(dtype=self.dtype, device=self.device)
        _check_shapes(X, "X")
        inside = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
        for k in range(self.d):
            low, high = self.bounds[k, 0], self.bounds[k, 1]
            inside &= (X[:, k] >= low) & (X[:, k] <= high)
        out = torch.full((X.shape[0],), -torch.inf, dtype=self.dtype, device=self.device)
        out[inside] = self._logpdf_inside
        return out

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        g = generator
        u = torch.rand((n, self.d), dtype=self.dtype, generator=g, device=self.device)
        return self.bounds[:, 0].unsqueeze(0) + u * (self.bounds[:, 1] - self.bounds[:, 0]).unsqueeze(0)


class KDEProposal:
    """
    Torch-friendly wrapper around numpy-based GaussianKDE (if available).

    This is optional. If you already have a Torch-native proposal with a .logpdf(X) method
    (e.g., `sampling_torch.CustomDistribution`), you don't need this wrapper.

    Note: sampling uses the KDE's Gaussian noise sampling; .logpdf is exact to the KDE definition.
    """
    def __init__(self, kde: "GaussianKDE", dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None):
        if not _HAS_NUMPY_KDE:
            raise RuntimeError("GaussianKDE not available. Import failed.")
        self.kde = kde
        self.dtype = dtype
        self.device = torch.device("cpu") if device is None else device

    @classmethod
    def from_samples(
        cls,
        X: Tensor,
        weights: Optional[Tensor] = None,
        H: Optional[Union[str, float, np.ndarray]] = None,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ) -> "KDEProposal":
        if not _HAS_NUMPY_KDE:
            raise RuntimeError("GaussianKDE not available. Import failed.")
        Xn = X.detach().to("cpu", dtype=torch.double).numpy()
        wn = None if weights is None else weights.detach().to("cpu", dtype=torch.double).numpy()
        kde = GaussianKDE(Xn, weights=wn, H=H)
        return cls(kde, dtype=dtype, device=device)

    def logpdf(self, X: Tensor, chunk_size: int = 4096) -> Tensor:
        Xn = X.detach().to("cpu", dtype=torch.double).numpy()
        # Use chunking to avoid memory spikes
        out = []
        for i in range(0, Xn.shape[0], chunk_size):
            sl = slice(i, min(i + chunk_size, Xn.shape[0]))
            out.append(self.kde.logpdf(Xn[sl]))
        out = np.concatenate(out, axis=0)
        return torch.from_numpy(out).to(dtype=self.dtype, device=self.device)

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        # GaussianKDE.sample doesn't take a torch.Generator; we ignore `generator` here.
        Xn = self.kde.sample(int(n))  # (n,d)
        return torch.from_numpy(Xn).to(dtype=self.dtype, device=self.device)


# ------------------------------
# MISEstimator
# ------------------------------

class MISEstimator:
    """
    Multiple Importance Sampling estimator for failure probability under a target p(x).
    Works with any list of proposal objects exposing `logpdf(X)` (and optionally `sample(n)`).

    Parameters
    ----------
    p : object with logpdf(X)->(n,) Tensor
        The *target/original* input distribution for the failure probability.
    proposals : list
        List of proposal objects, each exposing logpdf(X)->(n,) Tensor.
    failure_fn : callable
        A function taking X:(n,d)-> bool or {0,1} tensor. Example:
        failure_fn = lambda X: (func.eval(X).squeeze(-1) > 0)  # true if failure
    dtype, device : torch types
        Global dtype/device for computations.

    Notes
    -----
    - Deterministic Mixture (DM) estimator (balance heuristic) is unbiased.
    - Self-normalized IS (SNIS) is also supported (consistent).
    - We return standard error and 95% normal-approximate CI for quick diagnostics.
    """

    def __init__(
        self,
        p,
        proposals: Sequence,
        failure_fn: Callable[[Tensor], Union[Tensor, torch.BoolTensor]],
        *,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        self.p = p
        self.proposals = list(proposals)
        self.failure_fn = failure_fn
        self.dtype = dtype
        self.device = torch.device("cpu") if device is None else device

    # ---- mixture evaluation helpers ----
    def _mixture_logpdf(self, X: Tensor, log_alphas: Tensor) -> Tensor:
        """
        ψ(x) = Σ_j α_j q_j(x)
        Return log ψ(x) via log-sum-exp with log α_j + log q_j(x).
        """
        # (N, M)
        # Ensure each proposal returns (N,) vector; flatten if (N,1)
        log_q_cols = []
        for q in self.proposals:
            lq = q.logpdf(X)
            lq = lq.view(-1) if lq.ndim > 1 else lq
            log_q_cols.append(lq)
        log_q = torch.stack(log_q_cols, dim=1).to(dtype=self.dtype, device=self.device)
        # Broadcast add (N, M) + (M,) -> (N, M)
        lse = _logsumexp(log_q + log_alphas.unsqueeze(0), dim=1)
        return lse

    # ---- public APIs ----
    @torch.no_grad()
    def estimate(
        self,
        X: Tensor,
        prop_ids: Tensor,
        *,
        alphas: Optional[Tensor] = None,
        estimator: str = "dm",   # "dm" or "snis"
        return_per_sample: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Estimate P_F from a single concatenated batch X with proposal IDs for each row.

        Parameters
        ----------
        X : (N, d) tensor
        prop_ids : (N,) long tensor of values in [0, M-1]
        alphas : (M,) tensor of mixture weights; default = empirical n_j / N
        estimator : "dm" or "snis"
        return_per_sample : if True, include per-sample weights and indicators

        Returns
        -------
        dict with keys: 'pf_hat', 'stderr', 'ci95', 'N', 'alphas', optionally 'w', 'I'
        """
        X = X.to(dtype=self.dtype, device=self.device)
        _check_shapes(X)
        prop_ids = prop_ids.to(device=self.device)
        if prop_ids.ndim != 1 or prop_ids.shape[0] != X.shape[0]:
            raise ValueError("prop_ids must be shape (N,) and match X.shape[0].")

        N = X.shape[0]
        M = len(self.proposals)
        # mixture weights
        if alphas is None:
            counts = torch.bincount(prop_ids, minlength=M).to(dtype=self.dtype, device=self.device)
            alphas = counts / counts.sum()
        else:
            alphas = alphas.to(dtype=self.dtype, device=self.device)
            if alphas.shape != (M,):
                raise ValueError(f"alphas must be shape ({M},)")

        log_alphas = torch.log(alphas)
        log_p = self.p.logpdf(X).to(dtype=self.dtype, device=self.device)
        log_psi = self._mixture_logpdf(X, log_alphas)  # log Σ α_j q_j(x)

        # Failure indicator
        I = self.failure_fn(X)
        I = I.to(dtype=self.dtype, device=self.device).view(-1)

        # Importance weights w = p/ψ (in log-space for stability)
        log_w = log_p - log_psi
        w = torch.exp(log_w)

        # --- estimators ---
        if estimator.lower() == "dm":
            # Unbiased deterministic mixture estimator
            Z = I * w  # per-sample contribution
            pf_hat = torch.mean(Z)

            # SE via sample variance of Z_i / N
            # var_hat(Z) = 1/(N-1) * Σ (Z_i - mean)^2; stderr = sqrt(var_hat/N)
            if N > 1:
                var_hat = torch.var(Z, unbiased=True)
                stderr = torch.sqrt(var_hat / N)
            else:
                stderr = torch.nan

            ci95 = torch.stack([pf_hat - 1.96 * stderr, pf_hat + 1.96 * stderr]) if torch.isfinite(stderr) else torch.tensor([torch.nan, torch.nan], dtype=self.dtype, device=self.device)

        elif estimator.lower() == "snis":
            # Self-normalized IS estimator
            num = torch.sum(I * w)
            den = torch.sum(w)
            pf_hat = num / den

            # Delta-method style SE: approximate via normalized weights
            # Effective sample size
            w_norm = w / den
            ess = 1.0 / torch.sum(w_norm ** 2)
            # Bernoulli-ish variance proxy around pf_hat
            var_proxy = pf_hat * (1.0 - pf_hat)
            stderr = torch.sqrt(var_proxy / torch.clamp(ess, min=1.0))
            ci95 = torch.stack([pf_hat - 1.96 * stderr, pf_hat + 1.96 * stderr])

        else:
            raise ValueError("estimator must be 'dm' or 'snis'.")

        out = {
            "pf_hat": pf_hat,
            "stderr": stderr,
            "ci95": ci95,
            "N": torch.tensor(N, dtype=self.dtype, device=self.device),
            "alphas": alphas,
        }
        if return_per_sample:
            out["w"] = w
            out["I"] = I
            out["log_w"] = log_w
            out["log_p"] = log_p
            out["log_psi"] = log_psi
        return out

    @torch.no_grad()
    def estimate_from_batches(
        self,
        batches: Sequence[Tensor],
        *,
        alphas: Optional[Tensor] = None,
        estimator: str = "dm",
        return_per_sample: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Convenience wrapper: provide a list of sample batches, one per proposal j.
        Shapes: batches[j] is (n_j, d). The method concatenates them and sets
        the empirical α_j = n_j/N if not provided by the user.

        Returns: same dict as `estimate`.
        """
        # Prepare concatenated X and prop_ids
        N_list = [b.shape[0] for b in batches]
        X = torch.cat(batches, dim=0)
        prop_ids = torch.cat([torch.full((N_list[j],), j, dtype=torch.long, device=X.device) for j in range(len(batches))], dim=0)
        return self.estimate(X, prop_ids, alphas=alphas, estimator=estimator, return_per_sample=return_per_sample)


def MISEestimator_(proposals, samples_X, samples_Y, failure_fn):
    r"""
    proposals -- list of n+1 densities where the first one is the nominal p(x). Each should expose a logpdf method
    samples -- Batches of points sampled from each density. A list of arrays
    failure_fn -- a callable that evaluates failure on a vector Y
    """
    n_proposals = len(proposals)
    nbatch = [(samples_Y[i].squeeze()).shape[0] for i in range(len(samples_Y))]
    prop_weights = torch.tensor(nbatch) / sum(nbatch)
    indicators = failure_fn(torch.concat(samples_Y))

    num = []
    den = []

    px = proposals[0]  # nominal
    for i in range(len(proposals)):
        num.append(torch.exp(px.logpdf(samples_X[i])))
        den.append(prop_weights[i] * np.exp(proposals[i].logpdf(samples_X[i])))
    fpmis = torch.mean((indicators * 1) * torch.concat(num) / torch.concat(den).sum())

    return fpmis, indicators, num, den

def ISEestimator_(proposals, samples_X, samples_Y, failure_fn):
    r"""
    proposals -- list of n+1 densities where the first one is the nominal p(x). Each should expose a logpdf method
    samples -- Batches of points sampled from each density. A list of arrays
    failure_fn -- a callable that evaluates failure on a vector Y
    """
    n_proposals = len(proposals)
    nbatch = [ i for i in range(n_proposals)]
    nbatch.append(1)
    prop_weights = torch.tensor(nbatch) / sum(nbatch)
    indicators = failure_fn(torch.concat(samples_Y))

    num = []
    den = []

    px = proposals[0]  # nominal
    for i in range(len(proposals)):
        num.append(torch.exp(px.logpdf(samples_X[i])))
        den.append(prop_weights[i] * np.exp(proposals[i].logpdf(samples_X[i])))
    fpmis = torch.mean((indicators * 1) * torch.concat(num) / torch.concat(den).sum())

    return fpmis, indicators, num, den

def MCEestimator_(proposals, samples_X, samples_Y, failure_fn):
    r"""
    proposals -- list of n+1 densities where the first one is the nominal p(x). Each should expose a logpdf method
    samples -- Batches of points sampled from each density. A list of arrays
    failure_fn -- a callable that evaluates failure on a vector Y
    """
    n_proposals = len(proposals)
    nbatch = [ i for i in range(n_proposals)]
    nbatch.append(1)
    prop_weights = torch.tensor(nbatch) / sum(nbatch)
    indicators = failure_fn(torch.concat(samples_Y))

    num = []
    den = []

    px = proposals[0]  # nominal
    for i in range(len(proposals)):
        num.append(torch.exp(px.logpdf(samples_X[i])))
        den.append(prop_weights[i] * np.exp(proposals[i].logpdf(samples_X[i])))
    fpmis = torch.mean((indicators * 1) * torch.concat(num) / torch.concat(den).sum())

    return fpmis, indicators, num, den
# ------------------------------
# Minimal example (can be run as a script)
# ------------------------------

if __name__ == "__main__":
    # Toy 2D example using a uniform "p" on [-2,2]^2 and two proposals,
    # where q0 is uniform (same as p), and q1 is a KDE fit on a biased set.
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    # Define bounds and target distribution
    bounds = torch.tensor([[-2.0, 2.0],
                           [-2.0, 2.0]], dtype=torch.float64, device=device)
    p = UniformBox(bounds)

    # Synthetic "failure" function: failure if x1 + x2 > 1
    def failure_fn(X: Tensor) -> Tensor:
        return (X[:, 0] + X[:, 1] > 1.0)

    # Proposals: q0 uniform, q1 = KDE around some biased cloud near the failure boundary
    q0 = UniformBox(bounds)

    # Make some biased samples near failure
    rng = torch.Generator(device=device).manual_seed(0)
    base = q0.sample(500, generator=rng)
    mask = (base[:, 0] + base[:, 1] > 0.5)
    pilot_X = base[mask]
    if pilot_X.shape[0] < 50:
        pilot_X = base  # fall back to all points if mask is too strict

    proposals = [q0]
    batches = [q0.sample(2000, generator=rng)]

    if _HAS_NUMPY_KDE:
        kde_prop = KDEProposal.from_samples(pilot_X)
        proposals.append(kde_prop)
        # Draw from KDE
        # NOTE: sample() ignores torch.Generator due to numpy backend.
        batches.append(kde_prop.sample(1000))
    else:
        # If numpy KDE isn't available, just reuse uniform to complete the API
        proposals.append(q0)
        batches.append(q0.sample(1000, generator=rng))

    est = MISEstimator(p, proposals, failure_fn, device=device)

    # DM estimator (unbiased)
    out_dm = est.estimate_from_batches(batches, estimator="dm")
    print("[DM]  pf_hat =", float(out_dm["pf_hat"]), "  CI95 =", tuple(map(float, out_dm["ci95"])))

    # SNIS estimator (consistent)
    out_snis = est.estimate_from_batches(batches, estimator="snis")
    print("[SNIS] pf_hat =", float(out_snis["pf_hat"]), "  CI95 =", tuple(map(float, out_snis["ci95"])))
