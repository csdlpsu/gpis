import math
from typing import Callable, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from kde_test import GaussianKDE
from botorch.utils.transforms import normalize

TensorLike = Union[Tensor, float]

# =========================
# Utilities
# =========================
def _as_2d(x: TensorLike, dim: int, *, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> Tensor:
    X = x if isinstance(x, Tensor) else torch.tensor(x, dtype=dtype, device=device)
    X = X.to(dtype=dtype, device=device)
    if X.ndim == 0:
        X = X.reshape(1, 1)
    elif X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != dim:
        if X.shape[0] == dim and X.shape[1] == 1:
            X = X.T
        else:
            raise ValueError(f"Expected last dimension {dim}, got {tuple(X.shape)}")
    return X


def _vectorize_logpdf(
    logpdf: Callable[[Tensor], TensorLike],
    dim: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Callable[[Tensor], Tensor]:
    """
    Wrap a single-point or already-vectorized logpdf(x) into a function
    that takes X:(n,d) -> (n,)
    """
    def v(X: Tensor) -> Tensor:
        X = _as_2d(X, dim, dtype=dtype, device=device)
        # Try vectorized call first
        try:
            out = logpdf(X)
            out = out if isinstance(out, Tensor) else torch.tensor(out, dtype=dtype, device=device)
            out = out.to(dtype=dtype, device=device).reshape(-1)
            if out.numel() != X.shape[0]:
                raise ValueError
            return out
        except Exception:
            # Fallback: apply row-wise
            vals = []
            for i in range(X.shape[0]):
                yi = logpdf(X[i])
                yi = yi if isinstance(yi, Tensor) else torch.tensor(yi, dtype=dtype, device=device)
                vals.append(yi.reshape(()))  # scalar
            return torch.stack(vals, dim=0).to(dtype=dtype, device=device)
    return v


# =========================
# Core classes
# =========================
class CustomDistribution:
    """
    Flexible distribution interface in pure PyTorch.

    You can provide:
      - pdf(x)  OR  logpdf(x)
      - Optionally a normalizing constant logZ if logpdf is unnormalized
      - Optional custom sampler(n, generator) -> (n,d) Tensor
      - Optional support_indicator(x):(n,) -> {0,1}/bool Tensor

    All functions may accept vectorized X:(n,d). If they only support (d,),
    the wrapper will fall back to row-wise evaluation.
    """

    def __init__(
        self,
        dim: int,
        *,
        pdf: Optional[Callable[[Tensor], TensorLike]] = None,
        logpdf: Optional[Callable[[Tensor], TensorLike]] = None,
        normalized: bool = True,
        logZ: Optional[float] = None,
        sampler: Optional[Callable[[int, Optional[torch.Generator]], Tensor]] = None,
        support_indicator: Optional[Callable[[Tensor], Tensor]] = None,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        if pdf is None and logpdf is None:
            raise ValueError("Provide at least one of pdf or logpdf.")

        self.dim = int(dim)
        self.normalized = bool(normalized)
        self.logZ = None if logZ is None else float(logZ)
        self._user_pdf = pdf
        self._user_logpdf = logpdf
        self._sampler = sampler
        self.dtype = dtype
        self.device = device

        if self._user_pdf is not None:
            # derive logpdf from pdf
            _pdf = self._user_pdf

            def _pdf_wrapped(X: Tensor) -> Tensor:
                out = _pdf(X)
                out = out if isinstance(out, Tensor) else torch.tensor(out, dtype=dtype, device=device)
                return out.to(dtype=dtype, device=device)

            _vpdf = _vectorize_logpdf(_pdf_wrapped, self.dim, dtype=dtype, device=device)

            def _derived_logpdf(X: Tensor) -> Tensor:
                # guard against negatives
                val = torch.clamp(_vpdf(X), min=0.0)
                # log(0) -> -inf handled by torch.log
                return torch.log(val)

            self._logpdf = _derived_logpdf
            self.normalized = True  # pdf implies normalized
        else:
            # use provided logpdf
            _lp = self._user_logpdf  # type: ignore
            def _lp_wrapped(X: Tensor) -> Tensor:
                out = _lp(X)  # type: ignore
                out = out if isinstance(out, Tensor) else torch.tensor(out, dtype=dtype, device=device)
                return out.to(dtype=dtype, device=device)

            self._logpdf = _vectorize_logpdf(_lp_wrapped, self.dim, dtype=dtype, device=device)

        if support_indicator is None:
            self._support_ind = lambda X: torch.ones(_as_2d(X, self.dim, dtype=dtype, device=device).shape[0],
                                                     dtype=torch.bool, device=device)
        else:
            def _si_wrapped(X: Tensor) -> Tensor:
                out = support_indicator(X)
                out = out if isinstance(out, Tensor) else torch.tensor(out, dtype=torch.bool, device=device)
                return out.to(dtype=torch.bool, device=device).reshape(-1)
            self._support_ind = _si_wrapped

    # ---------- density evaluation ----------
    def logpdf(self, x: TensorLike) -> Tensor:
        X = _as_2d(x, self.dim, dtype=self.dtype, device=self.device)
        mask = self._support_ind(X)
        logp = torch.full((X.shape[0],), -torch.inf, dtype=self.dtype, device=self.device)
        if mask.any():
            base = self._logpdf(X[mask]).reshape(-1)
            if (not self.normalized) and (self.logZ is not None):
                base = base - float(self.logZ)
            logp[mask] = base
        return logp if (not isinstance(x, Tensor) or x.ndim > 1) else logp[0]

    def pdf(self, x: TensorLike) -> Tensor:
        return torch.exp(self.logpdf(x))

    # ---------- generic M-H sampler (Gaussian random walk) ----------
    @torch.no_grad()
    def sample(
        self,
        n: int,
        *,
        generator: Optional[torch.Generator] = None,
        init: Optional[Tensor] = None,
        burn_in: int = 500,
        thin: int = 1,
        step_scale: Union[float, Tensor] = 0.5,
        adapt_steps: int = 200,
        target_accept: float = 0.30,
    ) -> Tensor:
        """
        Draw samples using Metropolis–Hastings with a Gaussian random-walk proposal.

        If a custom sampler was provided, that is used instead.

        Returns: Tensor of shape (n, dim)
        """
        if self._sampler is not None:
            return self._sampler(n, generator)

        device = self.device
        dtype = self.dtype
        gen = generator

        step = step_scale if isinstance(step_scale, Tensor) else torch.tensor(step_scale, dtype=dtype, device=device)
        step = step.reshape(-1)
        if step.numel() == 1:
            step = step.repeat(self.dim)
        if step.numel() != self.dim:
            raise ValueError("step_scale must be scalar or length=dim.")

        if init is None:
            x = torch.zeros(self.dim, dtype=dtype, device=device)
        else:
            x = init.to(dtype=dtype, device=device).reshape(-1)
        if x.numel() != self.dim:
            raise ValueError("init must have shape (dim,)")

        total_kept = int(n)
        kept = []
        total_iters = burn_in + total_kept * thin

        logp_x = self.logpdf(x).item()
        if not math.isfinite(logp_x):
            # try to find a valid start near zeros
            for _ in range(2000):
                cand = x + torch.randn(self.dim, dtype=dtype, device=device, generator=gen)
                logp_c = self.logpdf(cand).item()
                if math.isfinite(logp_c):
                    x, logp_x = cand, logp_c
                    break
            if not math.isfinite(logp_x):
                raise RuntimeError("Could not find a valid starting point inside support.")

        # Robbins–Monro adaptation of log step size
        log_step = torch.log(torch.clamp(step, min=torch.finfo(dtype).tiny))

        for t in range(total_iters):
            prop = x + torch.exp(log_step) * torch.randn(self.dim, dtype=dtype, device=device, generator=gen)
            logp_prop_t = self.logpdf(prop)
            logp_prop = logp_prop_t.item() if logp_prop_t.ndim == 0 else float(logp_prop_t.reshape(()))

            accepted = 0
            if math.isfinite(logp_prop):
                log_alpha = logp_prop - logp_x
                if torch.log(torch.rand((), device=device, generator=gen)).item() < log_alpha:
                    x, logp_x = prop, logp_prop
                    accepted = 1

            # adapt during burn-in
            if t < min(adapt_steps, burn_in):
                a = 1.0 / math.sqrt(t + 1.0)
                log_step = log_step + a * (accepted - target_accept)

            if t >= burn_in and ((t - burn_in) % thin == 0):
                kept.append(x.clone())

        return torch.stack(kept, dim=0)


# =========================
# Convenience distributions
# =========================
class Gaussian(CustomDistribution):
    """Multivariate normal N(m, Σ) with full covariance."""
    def __init__(self, mean: TensorLike, cov: TensorLike, *, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None):
        m = _as_2d(mean, dim=1, dtype=dtype, device=device).reshape(-1)  # (d,)
        C = torch.as_tensor(cov, dtype=dtype, device=device)
        d = m.numel()
        if C.shape != (d, d):
            raise ValueError("cov must be (dim, dim)")

        # Cholesky + logdet + inverse (for Mahalanobis)
        L = torch.linalg.cholesky(C)
        log_det = 2.0 * torch.log(torch.diagonal(L)).sum()
        invC = torch.linalg.inv(C)
        norm_const = -0.5 * (d * math.log(2.0 * math.pi) + log_det)

        def logpdf(X: Tensor) -> Tensor:
            X = _as_2d(X, d, dtype=dtype, device=device)
            D = X - m
            q = torch.einsum("ni,ij,nj->n", D, invC, D)  # Mahalanobis^2
            return norm_const - 0.5 * q

        def sampler(n: int, g: Optional[torch.Generator]) -> Tensor:
            z = torch.randn((n, d), dtype=dtype, device=device, generator=g)
            return m + z @ L.T

        super().__init__(dim=d, logpdf=logpdf, normalized=True, sampler=sampler, dtype=dtype, device=device)


class MixtureOfGaussians(CustomDistribution):
    """
    Mixture \sum_k w_k N(m_k, Σ_k).
    weights: (K,), means: (K, d), covs: (K, d, d)
    """
    def __init__(
        self,
        weights: TensorLike,
        means: TensorLike,
        covs: TensorLike,
        *,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        w = torch.as_tensor(weights, dtype=dtype, device=device).reshape(-1)  # (K,)
        if (w < 0).any():
            raise ValueError("weights must be nonnegative")
        w = w / w.sum()

        M = torch.as_tensor(means, dtype=dtype, device=device)
        S = torch.as_tensor(covs, dtype=dtype, device=device)
        if M.ndim != 2 or S.ndim != 3 or M.shape[0] != S.shape[0]:
            raise ValueError("means must be (K,d), covs must be (K,d,d)")
        K, d = M.shape
        if S.shape[1:] != (d, d):
            raise ValueError("bad cov shape")

        L = torch.zeros_like(S)
        invS = torch.zeros_like(S)
        log_dets = torch.zeros(K, dtype=dtype, device=device)
        for k in range(K):
            L[k] = torch.linalg.cholesky(S[k])
            invS[k] = torch.linalg.inv(S[k])
            log_dets[k] = 2.0 * torch.log(torch.diagonal(L[k])).sum()
        norm_consts = -0.5 * (d * math.log(2.0 * math.pi) + log_dets)  # (K,)

        def comp_logpdf(X: Tensor) -> Tensor:
            X = _as_2d(X, d, dtype=dtype, device=device)     # (n,d)
            # Broadcast: D:(n,K,d)
            D = X[:, None, :] - M[None, :, :]
            q = torch.einsum("nkd,kdj,nkj->nk", D, invS, D)  # (n,K)
            return norm_consts[None, :] - 0.5 * q            # (n,K)

        def mix_logpdf(X: Tensor) -> Tensor:
            log_comps = comp_logpdf(X) + torch.log(w)[None, :]
            m = log_comps.max(dim=1, keepdim=True).values
            return (m + torch.log(torch.exp(log_comps - m).sum(dim=1, keepdim=True))).reshape(-1)

        def sampler(n: int, g: Optional[torch.Generator]) -> Tensor:
            # sample component indices
            ks = torch.multinomial(w, num_samples=n, replacement=True, generator=g)
            out = torch.empty((n, d), dtype=dtype, device=device)
            for k in range(K):
                idx = (ks == k).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    z = torch.randn((idx.numel(), d), dtype=dtype, device=device, generator=g)
                    out[idx] = M[k] + z @ L[k].T
            return out

        super().__init__(dim=d, logpdf=mix_logpdf, normalized=True, sampler=sampler, dtype=dtype, device=device)


# Some helper functions

def weighted_kde_sample(pilot_X, weights, h, q, jitter=False):
    """
    Sample `q` points from the weighted Gaussian-KDE defined by (pilot_X, weights).
    """
    # 1) pick q centers according to the discrete weights
    idx = torch.multinomial(weights, num_samples=q, replacement=False)
    centers = pilot_X[idx]
    # 2) jitter each center by N(0, h^2 I)
    if jitter:
        samples = centers + h * np.random.randn(q, pilot_X.shape[1])
    else:
        samples = centers
    return samples


def fit_and_sample_kde(pilot_X, weights, q=1):
    pilot_X_n = pilot_X.cpu().numpy()
    weights_n = weights.cpu().numpy()
    kde = GaussianKDE(pilot_X, weights=weights, bandwidth="silverman")

    return torch.tensor(kde.sample(q), dtype=torch.double), kde


def get_kde_weights(gp, px, pilot_X, bounds, threshold, alpha=1.0):
    train_X = gp.train_inputs[0]
    # Compute posterior failure prob π_n on pilot set
    with torch.no_grad():
        post = gp.posterior(normalize(pilot_X, bounds.T))
        mu = post.mean.squeeze()
        sigma = post.variance.sqrt().squeeze()
    pi_vals_ = (1.0 - torch.distributions.Normal(mu, sigma).cdf(torch.tensor(threshold))).clamp(1e-12, 1.0)
    eta = min(1., 5. / np.sqrt(train_X.shape[0]))
    pi_vals = (1. - eta) * (pi_vals_ ** alpha) + eta * px.pdf(pilot_X)

    # Compute weights and sample new points via KDE
    weights = pi_vals
    weights /= weights.sum()
    return weights

def fit_and_sample_kde_(pilot_X, weights, q=1, *, train_X=None):
    kde = GaussianKDE(pilot_X, weights=weights, bandwidth="scott")
    samples = torch.tensor(kde.sample(100 * q), dtype=torch.double)
    if train_X is None:
        return samples[:q], kde
    return maximin(samples, train_X, q), kde

def maximin(samples, train_X, q):
    samples = samples.detach().cpu().double()
    train_X = train_X.detach().cpu().double()
    if train_X.ndim == 1:
        train_X = train_X.unsqueeze(0)

    chosen = []
    mask = torch.ones(samples.size(0), dtype=torch.bool)
    curr = train_X
    for _ in range(q):
        d = torch.cdist(samples[mask], curr).min(dim=1).values
        i_rel = torch.argmax(d)
        idxs = torch.arange(samples.size(0))[mask]
        i_abs = idxs[i_rel]
        chosen.append(samples[i_abs])
        curr = torch.cat([curr, samples[i_abs:i_abs+1]], dim=0)
        mask[i_abs] = False
    return torch.stack(chosen, dim=0)


# =========================
# Example usage (CPU)
# =========================
if __name__ == "__main__":
    device = torch.device("cpu")
    dtype = torch.float64
    gen = torch.Generator(device=device).manual_seed(0)

    # 1) User-defined normalized pdf in 1D (bimodal, normalized by hand)
    def pdf_1d(x: Tensor) -> Tensor:
        x = x.reshape(-1) if x.ndim == 1 else x.reshape(-1)
        # Mixture 0.6*N(-2,1) + 0.4*N(3, 0.5^2)
        w = torch.tensor([0.6, 0.4], dtype=dtype, device=device)
        mus = torch.tensor([-2.0, 3.0], dtype=dtype, device=device)
        sigs = torch.tensor([1.0, 0.5], dtype=dtype, device=device)
        comps = []
        for m, s in zip(mus, sigs):
            comps.append((1.0 / (math.sqrt(2.0 * math.pi) * s)) * torch.exp(-0.5 * ((x - m) / s) ** 2))
        return w[0] * comps[0] + w[1] * comps[1]

    p1 = CustomDistribution(dim=1, pdf=pdf_1d, dtype=dtype, device=device)
    xs = torch.linspace(-6, 6, 5, dtype=dtype, device=device)
    print("pdf(x) at grid:", p1.pdf(xs))
    s1 = p1.sample(5, generator=gen, step_scale=0.7)
    print("samples:", s1[:5].ravel())

    # 2) Multivariate Gaussian
    mean = torch.tensor([0.0, 1.0], dtype=dtype, device=device)
    cov  = torch.tensor([[1.0, 0.5],
                         [0.5, 2.0]], dtype=dtype, device=device)
    p2 = Gaussian(mean, cov, dtype=dtype, device=device)
    print("logpdf([0,1]):", p2.logpdf(torch.tensor([0.0, 1.0], dtype=dtype, device=device)))
    s2 = p2.sample(1000, generator=gen)
    print("samples shape:", tuple(s2.shape))

    # 3) Mixture of Gaussians in 2D
    w = torch.tensor([0.3, 0.7], dtype=dtype, device=device)
    means = torch.tensor([[0.0, 0.0],
                          [3.0, -2.0]], dtype=dtype, device=device)
    covs = torch.stack([
        torch.eye(2, dtype=dtype, device=device),
        torch.tensor([[0.5, 0.2],
                      [0.2, 1.0]], dtype=dtype, device=device)
    ], dim=0)
    p3 = MixtureOfGaussians(w, means, covs, dtype=dtype, device=device)
    print("pdf([0,0]):", p3.pdf(torch.tensor([0.0, 0.0], dtype=dtype, device=device)))
    s3 = p3.sample(500, generator=gen)
    print("mixture samples shape:", tuple(s3.shape))
