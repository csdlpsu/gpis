import numpy as np
from typing import Callable, Optional, Tuple, Union

ArrayLike = Union[np.ndarray, float]

def _as_2d(x: ArrayLike, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != dim:
        if x.shape[0] == dim and x.shape[1] == 1:
            x = x.T
        else:
            raise ValueError(f"Expected last dimension {dim}, got {x.shape}")
    return x

def _vectorize_logpdf(logpdf: Callable[[np.ndarray], float], dim: int) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrap a single-point logpdf(x: (d,)) -> float into a vectorized version
    that accepts X: (n, d) and returns (n,).
    If logpdf already supports vectorized inputs, it should still work.
    """
    def v(X: np.ndarray) -> np.ndarray:
        X = _as_2d(X, dim)
        try:
            out = logpdf(X)
            out = np.asarray(out, dtype=float).reshape(-1)
            if out.shape[0] != X.shape[0]:
                raise ValueError
            return out
        except Exception:
            # Fallback: apply along rows
            return np.apply_along_axis(lambda xi: float(logpdf(xi)), 1, X)
    return v

class CustomDistribution:
    """
    A flexible distribution interface.

    You can:
      - provide a normalized pdf(x) OR logpdf(x)
      - OR provide only an unnormalized logpdf(x) and still sample via M-H;
        density will then be returned only up to a constant unless you also
        provide a normalizing constant (logZ).

    Parameters
    ----------
    dim : int
        Dimension of x.
    pdf : callable, optional
        Function pdf(x) -> float or array; must integrate to 1 if provided.
    logpdf : callable, optional
        Function logpdf(x) -> float or array. If provided without `pdf`,
        set `normalized=True` only if it is truly normalized.
    normalized : bool, default=True
        If False, density() will return values up to a proportionality constant
        unless `logZ` is given.
    logZ : float, optional
        Log normalizing constant if you have an unnormalized logpdf and know Z.
    sampler : callable, optional
        Custom sampler: sampler(n, rng) -> (n, dim) array.
    support_indicator : callable, optional
        support_indicator(x) -> bool or {0,1}; returning False/-inf outside support
        helps MH reject invalid proposals gracefully.
    """

    def __init__(
        self,
        dim: int,
        pdf: Optional[Callable[[np.ndarray], ArrayLike]] = None,
        logpdf: Optional[Callable[[np.ndarray], ArrayLike]] = None,
        normalized: bool = True,
        logZ: Optional[float] = None,
        sampler: Optional[Callable[[int, np.random.Generator], np.ndarray]] = None,
        support_indicator: Optional[Callable[[np.ndarray], Union[bool, np.ndarray]]] = None,
    ):
        if pdf is None and logpdf is None:
            raise ValueError("Provide at least one of pdf or logpdf.")
        self.dim = int(dim)
        self._user_pdf = pdf
        self._user_logpdf = logpdf
        self._sampler = sampler
        self.normalized = bool(normalized)
        self.logZ = logZ
        self._support_ind = support_indicator

        if self._user_pdf is not None:
            # derive logpdf from pdf
            _pdf = self._user_pdf
            _vpdf = _vectorize_logpdf(lambda X: _pdf(X), self.dim)
            self._logpdf = lambda X: np.log(np.maximum(_vpdf(X), 0.0))
            self.normalized = True  # if pdf is provided, we assume normalized
        else:
            # use provided logpdf
            self._logpdf = _vectorize_logpdf(self._user_logpdf, self.dim)

        if self._support_ind is None:
            # default: everywhere supported
            self._support_ind = lambda X: np.ones(_as_2d(X, self.dim).shape[0], dtype=bool)

    # ---------- density evaluation ----------
    def logpdf(self, x: ArrayLike) -> np.ndarray:
        X = _as_2d(x, self.dim)
        mask = self._support_ind(X)
        logp = np.full(X.shape[0], -np.inf, dtype=float)
        if np.any(mask):
            base = self._logpdf(X[mask])
            base = np.asarray(base, dtype=float).reshape(-1)
            if not self.normalized and self.logZ is not None:
                base = base - float(self.logZ)
            logp[mask] = base
        return logp if np.ndim(x) > 1 else logp[0]

    def pdf(self, x: ArrayLike) -> np.ndarray:
        return np.exp(self.logpdf(x))

    # ---------- generic M-H sampler (Gaussian random walk) ----------
    def sample(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        init: Optional[np.ndarray] = None,
        burn_in: int = 500,
        thin: int = 1,
        step_scale: Union[float, np.ndarray] = 0.5,
        adapt_steps: int = 200,
        target_accept: float = 0.30,
    ) -> np.ndarray:
        """
        Draw samples using a simple Metropolis–Hastings random-walk kernel.

        If a custom sampler was provided at init, that is used instead.

        Parameters
        ----------
        n : int
            Number of kept samples (post burn-in & thinning).
        rng : np.random.Generator, optional
        init : (dim,), optional
            Starting point. If None, starts at zeros.
        burn_in : int, default=500
        thin : int, default=1
        step_scale : float or (dim,), default=0.5
            Proposal x' = x + step_scale * N(0, I).
        adapt_steps : int, default=200
            Number of adaptation steps within burn-in to tune step_scale.
        target_accept : float, default=0.30
            Target acceptance rate for adaptation.

        Returns
        -------
        samples : (n, dim) ndarray
        """
        if self._sampler is not None:
            if rng is None:
                rng = np.random.default_rng()
            return self._sampler(n, rng)

        if rng is None:
            rng = np.random.default_rng()
        step = np.array(step_scale, dtype=float).reshape(-1)
        if step.size == 1:
            step = np.full(self.dim, step.item())
        if step.size != self.dim:
            raise ValueError("step_scale must be scalar or length=dim.")

        x = np.zeros(self.dim) if init is None else np.array(init, dtype=float).reshape(-1)
        if x.size != self.dim:
            raise ValueError("init must have shape (dim,)")

        total_kept = n
        kept = []
        total_iters = burn_in + total_kept * thin
        logp_x = float(self.logpdf(x))
        if not np.isfinite(logp_x):
            # try to find a valid start within a small ball
            for _ in range(2000):
                cand = x + rng.normal(size=self.dim) * 1.0
                logp_c = float(self.logpdf(cand))
                if np.isfinite(logp_c):
                    x, logp_x = cand, logp_c
                    break
            if not np.isfinite(logp_x):
                raise RuntimeError("Could not find a valid starting point inside support.")

        # simple Robbins-Monro adaptation on log step
        log_step = np.log(np.maximum(step, 1e-12))

        accepts = 0
        for t in range(total_iters):
            prop = x + rng.normal(size=self.dim) * np.exp(log_step)
            logp_prop = float(self.logpdf(prop))
            if np.isfinite(logp_prop):
                log_alpha = logp_prop - logp_x  # symmetric proposal
                if np.log(rng.uniform()) < log_alpha:
                    x, logp_x = prop, logp_prop
                    accepted = 1
                else:
                    accepted = 0
            else:
                accepted = 0

            # adapt during the initial adapt_steps of the burn-in
            if t < min(adapt_steps, burn_in):
                a = 1.0 / np.sqrt(t + 1.0)  # diminishing
                # global adaptation (same across dims)
                log_step += a * (accepted - target_accept)

            # store if beyond burn-in and at thinning interval
            if t >= burn_in and ((t - burn_in) % thin == 0):
                kept.append(x.copy())
            accepts += accepted

        return np.asarray(kept, dtype=float)

# ---------- Convenience concrete distributions ----------

class Gaussian(CustomDistribution):
    """
    Multivariate normal N(m, Σ).
    Supports full covariance; internally uses Cholesky for stability.
    """
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        mean = np.asarray(mean, dtype=float).reshape(-1)
        cov = np.asarray(cov, dtype=float)
        dim = mean.size
        if cov.shape != (dim, dim):
            raise ValueError("cov must be (dim, dim)")
        L = np.linalg.cholesky(cov)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        inv = np.linalg.inv(cov)
        norm_const = -0.5 * (dim * np.log(2.0 * np.pi) + log_det)

        def logpdf(x: np.ndarray) -> np.ndarray:
            X = _as_2d(x, dim)
            d = X - mean
            q = np.einsum("...i,ij,...j->...", d, inv, d)  # Mahalanobis^2
            return norm_const - 0.5 * q

        def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
            z = rng.normal(size=(n, dim))
            return mean + z @ L.T

        super().__init__(dim=dim, logpdf=logpdf, normalized=True, sampler=sampler)

class MixtureOfGaussians(CustomDistribution):
    """
    Mixture \sum_k w_k N(m_k, Σ_k).
    weights: (K,), means: (K, d), covs: (K, d, d)
    """
    def __init__(self, weights: np.ndarray, means: np.ndarray, covs: np.ndarray):
        w = np.asarray(weights, dtype=float).reshape(-1)
        K = w.size
        if np.any(w < 0): raise ValueError("weights must be nonnegative")
        w = w / w.sum()

        means = np.asarray(means, dtype=float)
        covs = np.asarray(covs, dtype=float)
        if means.ndim != 2 or covs.ndim != 3 or means.shape[0] != K or covs.shape[0] != K:
            raise ValueError("means must be (K,d), covs must be (K,d,d)")
        K, d = means.shape
        if covs.shape[1:] != (d, d): raise ValueError("bad cov shape")

        L = np.zeros_like(covs)
        log_dets = np.zeros(K)
        invs = np.zeros_like(covs)
        for k in range(K):
            L[k] = np.linalg.cholesky(covs[k])
            log_dets[k] = 2.0 * np.sum(np.log(np.diag(L[k])))
            invs[k] = np.linalg.inv(covs[k])
        norm_consts = -0.5 * (d * np.log(2.0 * np.pi) + log_dets)  # (K,)

        def comp_logpdf(X: np.ndarray) -> np.ndarray:
            # returns (n, K) of component log-densities
            X = _as_2d(X, d)
            n = X.shape[0]
            # Mahalanobis via broadcasting
            D = X[:, None, :] - means[None, :, :]        # (n,K,d)
            q = np.einsum("nkd,kdj,nkj->nk", D, invs, D) # (n,K)
            return norm_consts[None, :] - 0.5 * q

        def mix_logpdf(x: np.ndarray) -> np.ndarray:
            X = _as_2d(x, d)
            log_comps = comp_logpdf(X) + np.log(w)[None, :]
            # log-sum-exp across components
            m = np.max(log_comps, axis=1, keepdims=True)
            return (m + np.log(np.sum(np.exp(log_comps - m), axis=1, keepdims=True))).ravel()

        def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
            ks = rng.choice(K, size=n, p=w)
            out = np.empty((n, d))
            for k in range(K):
                idx = np.where(ks == k)[0]
                if idx.size > 0:
                    z = rng.normal(size=(idx.size, d))
                    out[idx] = means[k] + z @ L[k].T
            return out

        super().__init__(dim=d, logpdf=mix_logpdf, normalized=True, sampler=sampler)

# ---------- Example usage ----------
r"""
rng = np.random.default_rng(0)

    # 1) User-defined normalized pdf in 1D (bimodal, normalized by hand)
    def pdf_1d(x):
        x = np.atleast_1d(x)
        # Mixture 0.6*N(-2,1) + 0.4*N(3, 0.5^2)
        w = np.array([0.6, 0.4])
        mus = np.array([-2.0, 3.0])
        sigs = np.array([1.0, 0.5])
        comps = []
        for m, s in zip(mus, sigs):
            comps.append(1.0/np.sqrt(2*np.pi*s**2) * np.exp(-0.5*((x - m)/s)**2))
        return w[0]*comps[0] + w[1]*comps[1]

    p1 = CustomDistribution(dim=1, pdf=pdf_1d)
    xs = np.linspace(-6, 6, 5)
    print("pdf(x) at grid:", p1.pdf(xs))
    s1 = p1.sample(5, rng=rng, step_scale=0.7)
    print("samples:", s1[:5].ravel())

    # 2) Multivariate Gaussian
    mean = np.array([0.0, 1.0])
    cov  = np.array([[1.0, 0.5],
                     [0.5, 2.0]])
    p2 = Gaussian(mean, cov)
    print("logpdf([0,1]):", p2.logpdf([0.0, 1.0]))
    s2 = p2.sample(1000, rng=rng)
    print("samples shape:", s2.shape)

    # 3) Mixture of Gaussians in 2D
    w = np.array([0.3, 0.7])
    means = np.array([[0.0, 0.0],
                      [3.0, -2.0]])
    covs = np.stack([
        np.eye(2),
        np.array([[0.5, 0.2],
                  [0.2, 1.0]])
    ], axis=0)
    p3 = MixtureOfGaussians(w, means, covs)
    print("pdf([0,0]):", p3.pdf([0.0, 0.0]))
    s3 = p3.sample(500, rng=rng)
    print("mixture samples shape:", s3.shape)
"""
