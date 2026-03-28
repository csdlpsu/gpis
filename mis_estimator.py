
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
from botorch.utils.transforms import normalize

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

def MISEestimatorMF(gp, proposals, samples_X, samples_Y, failure_fn, bounds, MC_size=100_000, clip_to_bounds=None):
    r"""
    proposals -- list of n+1 densities where the first one is the nominal p(x). Each should expose a logpdf method
    samples -- Batches of points sampled from each density. A list of arrays
    failure_fn -- a callable that evaluates failure on a vector Y
    """
    n_proposals = len(proposals)
    nbatch = [(samples_Y[i].squeeze()).shape[0] for i in range(len(samples_Y))]
    prop_weights = torch.tensor(nbatch) / sum(nbatch)
    indicators_hf = failure_fn(torch.concat(samples_Y))
    # indicators_lf = failure_fn( gp.posterior(normalize(torch.concat(samples_X), bounds)).mean.detach() )
    indicators_lf = failure_fn( gp.posterior(torch.concat(samples_X)).mean.detach() )
    num = []
    den = []

    px = proposals[0]  # nominal
    for i in range(len(proposals)):
        num.append(torch.exp(px.logpdf(samples_X[i])))
        den.append(prop_weights[i] * np.exp(proposals[i].logpdf(samples_X[i])))
    fpmis = torch.mean( ((indicators_hf * 1) - (indicators_lf * 1)) * torch.concat(num) / torch.concat(den).sum())

    if clip_to_bounds is not None:
        X = clip_to_bounds(px.sample(MC_size).double(), bounds.double())  
    else:
        X = px.sample(MC_size)
    # Y = gp.posterior(normalize(X, bounds)).mean.detach()
    Y = gp.posterior(X).mean.detach()
    
    fpmc = torch.mean(failure_fn(Y) * 1.)

    return fpmc + fpmis

def MFEestimator_(proposals, samples_X, samples_Y, failure_fn, gp_model=None):
    r"""
    Multifidelity deterministic-mixture MIS estimator.

    proposals  -- list of m densities [p, q1, ..., q_{m-1}] with a logpdf(X) method.
                 (To use multifidelity term0, each proposal must also be samplable;
                 see _draw() below for supported method names.)
    samples_X  -- list of tensors; HF inputs drawn from each proposal i (ONLY HF inputs).
                 Must satisfy samples_X[i].shape[0] == samples_Y[i].shape[0].
    samples_Y  -- list of tensors; HF outputs corresponding to samples_X[i].
    failure_fn -- callable mapping outputs Y -> indicator (bool/0-1) per row.
    gp_model   -- optional surrogate/GP model used to predict outputs at X.

    Returns:
      pf_est, indicators_true, num, den

      indicators_true : failure_fn applied to concatenated HF outputs
      num             : list of p(x) evaluated at HF X batches (vectors)
      den             : list of q_mix(x) evaluated at HF X batches (vectors)
    """

    # ----------------------------
    # Small helpers (robustness)
    # ----------------------------
    def _to_torch(a, like):
        if isinstance(a, torch.Tensor):
            return a.to(device=like.device, dtype=like.dtype)
        return torch.as_tensor(a, device=like.device, dtype=like.dtype)

    def _logpdf(prop, X):
        return _to_torch(prop.logpdf(X), X).reshape(-1)

    def _mix_pdf(X, prop_weights):
        # q_mix(x) = sum_j alpha_j q_j(x)
        mix = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
        for j in range(len(proposals)):
            mix = mix + prop_weights[j] * torch.exp(_logpdf(proposals[j], X))
        return torch.clamp(mix, min=1e-32)

    def _gp_predict(gp, X):
        with torch.no_grad():
            if hasattr(gp, "predict"):
                y = gp.predict(X)
            else:
                y = gp(X)
                # gpytorch-like: output may have .mean
                if hasattr(y, "mean"):
                    y = y.mean
                # some models return (mean, var) or similar
                if isinstance(y, (tuple, list)):
                    y = y[0]
            return _to_torch(y, X)

    def _draw(prop, n, ref_X):
        # Try a few common sampler APIs.
        # - torch.distributions: sample((n,)) or rsample((n,))
        # - scipy.stats: rvs(size=n)
        # - scipy.stats.gaussian_kde: resample(n) -> (d, n)
        x = None
        if hasattr(prop, "rsample"):
            try:
                x = prop.rsample((n,))
            except Exception:
                pass
        if x is None and hasattr(prop, "sample"):
            try:
                x = prop.sample((n,))
            except Exception:
                try:
                    x = prop.sample(n)
                except Exception:
                    pass
        if x is None and hasattr(prop, "rvs"):
            try:
                x = prop.rvs(size=n)
            except Exception:
                pass
        if x is None and hasattr(prop, "resample"):
            try:
                x = prop.resample(n)
                # gaussian_kde returns (d, n)
                if hasattr(x, "T"):
                    x = x.T
            except Exception:
                pass
        if x is None:
            raise TypeError(
                "Cannot draw surrogate samples: proposal has no supported sampler "
                "(rsample/sample/rvs/resample)."
            )

        x = _to_torch(x, ref_X)
        # Ensure (n, d) shape if HF X is 2D
        if ref_X.ndim == 2 and x.ndim == 1:
            x = x.unsqueeze(-1)
        return x

    # ----------------------------
    # HF bookkeeping / weights
    # ----------------------------
    m = len(proposals)
    assert len(samples_X) == m, "samples_X must match proposals length"
    assert len(samples_Y) == m, "samples_Y must match proposals length"

    # Make sure HF X and Y align 1:1 (as per your new requirement)
    for i in range(m):
        if samples_X[i].shape[0] != samples_Y[i].squeeze().shape[0]:
            raise ValueError(
                f"HF alignment mismatch at i={i}: "
                f"samples_X has {samples_X[i].shape[0]} rows but samples_Y has {samples_Y[i].squeeze().shape[0]}."
            )

    nbatch_hf = [samples_X[i].shape[0] for i in range(m)]
    N_hf = int(sum(nbatch_hf))
    if N_hf <= 0:
        raise ValueError("Need at least one HF sample.")

    prop_weights_hf = torch.tensor(nbatch_hf, dtype=torch.get_default_dtype())
    prop_weights_hf = prop_weights_hf / prop_weights_hf.sum()

    px = proposals[0]  # nominal density p(x)

    # True indicators on HF outputs
    Y_hf_all = torch.concat(samples_Y, dim=0)
    indicators_true = failure_fn(Y_hf_all).to(torch.get_default_dtype()).reshape(-1)

    # MIS weights on HF inputs
    num, den = [], []
    for i in range(m):
        Xi = samples_X[i]
        pi = torch.exp(_logpdf(px, Xi))
        qmix_i = _mix_pdf(Xi, prop_weights_hf)
        num.append(pi)
        den.append(qmix_i)

    w_hf = torch.concat(num, dim=0) / torch.concat(den, dim=0)
    pf_hf_mis = torch.mean(indicators_true * w_hf)

    # If no GP given, just return standard HF MIS
    if gp_model is None:
        return pf_hf_mis, indicators_true, num, den

    # ----------------------------
    # Term 0: surrogate MIS (large, internal sampling)
    # ----------------------------
    # Controls how many *surrogate-only* points you draw relative to HF.
    # Increase this to reduce MC noise in term0 (subject to compute/memory).
    SURROGATE_MULT = 100

    # Stream in chunks to avoid huge concatenations if N is large.
    CHUNK_SIZE = 50_000

    # Since we scale all nbatch by a constant integer, the mixture weights match prop_weights_hf exactly.
    term0_sum = torch.zeros((), device=samples_X[0].device, dtype=samples_X[0].dtype)
    N_surr_total = 0

    for i in range(m):
        n_i = int(SURROGATE_MULT * nbatch_hf[i])
        if n_i <= 0:
            continue
        N_surr_total += n_i

        for start in range(0, n_i, CHUNK_SIZE):
            n_chunk = min(CHUNK_SIZE, n_i - start)
            Xs = _draw(proposals[i], n_chunk, ref_X=samples_X[i])

            Yhat = _gp_predict(gp_model, Xs)
            ind_hat = failure_fn(Yhat).to(Xs.dtype).reshape(-1)

            ps = torch.exp(_logpdf(px, Xs))
            qmix_s = _mix_pdf(Xs, prop_weights_hf)
            ws = ps / qmix_s

            term0_sum = term0_sum + torch.sum(ind_hat * ws)

    if N_surr_total <= 0:
        raise ValueError("Internal surrogate sampling produced zero samples; check HF batches and SURROGATE_MULT.")
    term0 = term0_sum / N_surr_total

    # ----------------------------
    # Term 1: HF correction MIS
    # ----------------------------
    X_hf_all = torch.concat(samples_X, dim=0)
    Yhat_hf = _gp_predict(gp_model, X_hf_all)
    indicators_hat_hf = failure_fn(Yhat_hf).to(X_hf_all.dtype).reshape(-1)

    delta = indicators_true - indicators_hat_hf
    term1 = torch.mean(delta * w_hf)

    fpmf = term0 + term1
    return fpmf, indicators_true, num, den


def MFEestimator2_(proposals, samples_X, samples_Y, failure_fn, gp_model=None):
    r"""
    Fast multifidelity stratified-IS estimator.

    Key change vs deterministic-mixture MIS:
      - For surrogate term0 (and correction term1), uses per-proposal IS weights p(x)/q_i(x)
        instead of mixture weights p(x)/sum_j alpha_j q_j(x).
      - This avoids O(m) logpdf calls per sample and is typically 10-100x faster.

    proposals  -- list of m densities [p, q1, ..., q_{m-1}], each exposing:
                  - logpdf(X)
                  - a sampling method for internal surrogate sampling (see _draw()).
    samples_X  -- list of tensors; HF inputs only from each proposal i.
    samples_Y  -- list of tensors; HF outputs aligned 1:1 with samples_X[i].
    failure_fn -- callable mapping outputs Y -> indicator per row (bool/0-1)
    gp_model   -- optional surrogate (your Surrogates/BoTorch GP object)

    Returns:
      pf_est, indicators_true, num, den
      - num: list of p(x) evaluated at HF X batches
      - den: list of q_i(x) evaluated at HF X batches  (NOTE: not mixture denominator)
    """

    # ----------------------------
    # Internal constants
    # ----------------------------
    SURROGATE_MULT = 50      # reduce/increase depending on speed/variance tradeoff
    CHUNK_SIZE = 50_000
    EPS = 1e-32

    m = len(proposals)
    assert len(samples_X) == m and len(samples_Y) == m, "samples_X/samples_Y must match proposals length"

    # ----------------------------
    # Helpers
    # ----------------------------
    def _to_torch(a, like):
        if isinstance(a, torch.Tensor):
            return a.to(device=like.device, dtype=like.dtype)
        return torch.as_tensor(a, device=like.device, dtype=like.dtype)

    def _logpdf(prop, X):
        return _to_torch(prop.logpdf(X), X).reshape(-1)

    def _draw(prop, n, ref_X):
        # Supported sampler APIs: rsample / sample / rvs / resample
        x = None
        if hasattr(prop, "rsample"):
            try:
                x = prop.rsample((n,))
            except Exception:
                pass
        if x is None and hasattr(prop, "sample"):
            try:
                x = prop.sample((n,))
            except Exception:
                try:
                    x = prop.sample(n)
                except Exception:
                    pass
        if x is None and hasattr(prop, "rvs"):
            try:
                x = prop.rvs(size=n)
            except Exception:
                pass
        if x is None and hasattr(prop, "resample"):
            try:
                x = prop.resample(n)
                if hasattr(x, "T"):
                    x = x.T  # e.g. scipy gaussian_kde returns (d, n)
            except Exception:
                pass
        if x is None:
            raise TypeError(
                "Cannot draw surrogate samples: each proposal must support rsample/sample/rvs/resample."
            )

        x = _to_torch(x, ref_X)
        if ref_X.ndim == 2 and x.ndim == 1:
            x = x.unsqueeze(-1)
        return x

    def _gp_predict(model, X):
        # Works with BoTorch/GPyTorch GP: gp(X) -> posterior with .mean
        with torch.no_grad():
            model.eval()
            out = model(X)
            if hasattr(out, "mean"):
                return out.mean
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

    # ----------------------------
    # HF alignment check
    # ----------------------------
    for i in range(m):
        nx = samples_X[i].shape[0]
        ny = samples_Y[i].squeeze().shape[0]
        if nx != ny:
            raise ValueError(f"HF alignment mismatch at i={i}: samples_X has {nx}, samples_Y has {ny}.")

    nbatch_hf = [samples_X[i].shape[0] for i in range(m)]
    N_hf = int(sum(nbatch_hf))
    if N_hf <= 0:
        raise ValueError("Need at least one HF sample.")

    px = proposals[0]

    # ----------------------------
    # HF true indicator
    # ----------------------------
    Y_hf_all = torch.concat(samples_Y, dim=0)
    indicators_true = failure_fn(Y_hf_all).to(torch.get_default_dtype()).reshape(-1)

    # If no GP, return stratified IS on HF only (fast, unbiased)
    # (This replaces your previous deterministic-mixture MIS with a faster unbiased variant.)
    # Weight for HF point from proposal i is p/q_i.
    num, den = [], []
    hf_sum = torch.zeros((), device=samples_X[0].device, dtype=samples_X[0].dtype)
    offset = 0
    for i in range(m):
        Xi = samples_X[i]
        ni = Xi.shape[0]
        if ni == 0:
            num.append(torch.empty((0,), device=Xi.device, dtype=Xi.dtype))
            den.append(torch.empty((0,), device=Xi.device, dtype=Xi.dtype))
            continue

        logp = _logpdf(px, Xi)
        logqi = _logpdf(proposals[i], Xi)
        pi = torch.exp(logp)
        qi = torch.exp(logqi).clamp_min(EPS)

        num.append(pi)
        den.append(qi)

        ind_i = indicators_true[offset:offset + ni].to(Xi.dtype)
        wi = pi / qi
        hf_sum = hf_sum + torch.sum(ind_i * wi)
        offset += ni

    pf_hf = hf_sum / N_hf

    if gp_model is None:
        return pf_hf, indicators_true, num, den

    # ----------------------------
    # Term 0: surrogate estimate of E_p[ I_hat ] via internal stratified IS
    # ----------------------------
    term0_sum = torch.zeros((), device=samples_X[0].device, dtype=samples_X[0].dtype)
    N_surr_total = 0

    for i in range(m):
        n_i = int(SURROGATE_MULT * nbatch_hf[i])
        if n_i <= 0:
            continue
        N_surr_total += n_i

        for start in range(0, n_i, CHUNK_SIZE):
            n_chunk = min(CHUNK_SIZE, n_i - start)
            Xs = _draw(proposals[i], n_chunk, ref_X=samples_X[i])

            Yhat = _gp_predict(gp_model, Xs)
            ind_hat = failure_fn(Yhat).to(Xs.dtype).reshape(-1)

            logp = _logpdf(px, Xs)
            logqi = _logpdf(proposals[i], Xs)
            ws = torch.exp(logp - logqi).clamp_max(1e32)  # optional safety

            term0_sum = term0_sum + torch.sum(ind_hat * ws)

    if N_surr_total <= 0:
        raise ValueError("Internal surrogate sampling produced zero samples.")
    term0 = term0_sum / N_surr_total

    # ----------------------------
    # Term 1: HF correction E_p[ I - I_hat ] using the SAME HF points (stratified IS)
    # ----------------------------
    X_hf_all = torch.concat(samples_X, dim=0)
    Yhat_hf = _gp_predict(gp_model, X_hf_all)
    indicators_hat_hf = failure_fn(Yhat_hf).to(X_hf_all.dtype).reshape(-1)

    delta = indicators_true.to(X_hf_all.dtype) - indicators_hat_hf

    # We already accumulated p/q_i weights by strata for HF in pf_hf computation; reuse them efficiently:
    term1_sum = torch.zeros((), device=X_hf_all.device, dtype=X_hf_all.dtype)
    offset = 0
    for i in range(m):
        Xi = samples_X[i]
        ni = Xi.shape[0]
        if ni == 0:
            continue

        # reuse num/den already computed for HF strata
        wi = (num[i] / den[i]).to(Xi.dtype)
        di = delta[offset:offset + ni].to(Xi.dtype)
        term1_sum = term1_sum + torch.sum(di * wi)
        offset += ni

    term1 = term1_sum / N_hf

    fpmf = term0 + term1
    return fpmf, indicators_true, num, den
    
def ISEestimator_(proposals, samples_X, samples_Y, failure_fn):
    r"""
    proposals -- list of n+1 densities where the first one is the nominal p(x). Each should expose a logpdf method
    samples -- Batches of points sampled from each density. A list of arrays
    failure_fn -- a callable that evaluates failure on a vector Y
    """
    n_proposals = len(proposals)
    nbatch = [ 0 for i in range(n_proposals)]
    nbatch.append(1)
    prop_weights = torch.tensor(nbatch) / sum(nbatch)
    indicators = failure_fn(torch.concat(samples_Y))

    num = []
    den = []

    px = proposals[0]  # nominal
    # for i in range(len(proposals)):
    #     num.append(torch.exp(px.logpdf(samples_X[i])))
    #     den.append(prop_weights[i] * np.exp(proposals[i].logpdf(samples_X[i])))
    num.append(torch.exp(px.logpdf(samples_X[-1])))
    den.append( torch.tensor( np.exp(proposals[-1].logpdf(samples_X[-1]))) )
    # fpmis = torch.mean((indicators * 1) * torch.concat(num) / torch.concat(den).sum())
    # fpmis = torch.mean(torch.concat(num) / torch.concat(den).sum())  # debug
    fpmis = torch.mean(torch.concat(num) / torch.concat(den).sum())  # debug
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
    # fpmis = torch.mean((indicators * 1) * torch.concat(num) / torch.concat(den).sum())
    fpmis = torch.mean( torch.concat(num) / torch.concat(den).sum()) # debug

    return fpmis, indicators, num, den
