import numpy as np
from typing import Optional, Union
from scipy.stats import gaussian_kde

class GaussianKDE:
    """
    Weighted multivariate Gaussian KDE.

    pdf(x)  = sum_i w_i * N(x | X_i, H)
    H (bandwidth) can be: scalar*Sigma, diagonal, or full SPD matrix.

    Parameters
    ----------
    X : (n, d) array
        Samples.
    weights : (n,) array, optional
        Nonnegative weights; will be normalized to sum to 1.
    bandwidth : {"scott","silverman"} | float | (d,) | (d,d)
        - "scott":  H = n_eff^{-2/(d+4)} * Sigma
        - "silverman": H = c * n_eff^{-2/(d+4)} * Sigma, c=(4/(d+2))^{2/(d+4)}
        - float s:  H = s^2 * Sigma
        - (d,):     H = diag(bw**2)
        - (d,d):    H = bw (must be SPD)
    """
    Array = np.ndarray
    def __init__(
        self,
        X: Array,
        weights: Optional[Array] = None,
        bandwidth: Union[str, float, Array] = "scott",
    ):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be (n, d)")
        self.X = X
        self.n, self.d = X.shape

        # Weights
        if weights is None:
            w = np.ones(self.n, dtype=float) / self.n
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.shape[0] != self.n:
                raise ValueError("weights must have shape (n,)")
            if np.any(w < 0):
                raise ValueError("weights must be nonnegative")
            s = w.sum()
            if s <= 0:
                raise ValueError("sum of weights must be positive")
            w = w / s
        self.w = w
        self.logw = np.log(self.w + 1e-300)  # guard tiny weights
        # Effective sample size (useful for rules-of-thumb)
        self.n_eff = (self.w.sum() ** 2) / (np.sum(self.w ** 2) + 1e-300)

        # Weighted mean and covariance
        mu = (self.w[:, None] * self.X).sum(axis=0)
        Xm = self.X - mu
        # Weighted covariance with weights summing to 1:
        # cov = sum w_i (x_i - mu)(x_i - mu)^T / (1 - sum w_i^2)  for an unbiased-ish version;
        # but KDE bandwidth rules typically use the "population" version sum w_i(..)(..)^T.
        cov = (self.w[:, None, None] * (Xm[:, :, None] * Xm[:, None, :])).sum(axis=0)
        self.mu = mu
        self.Sigma = cov

        # Bandwidth matrix H
        self.H = self._make_bandwidth(bandwidth)

        # Cholesky + constants
        self.L = np.linalg.cholesky(self.H)  # H = L L^T
        self._log_sqrt_detH = np.sum(np.log(np.diag(self.L)))  # log det(H)^{1/2} = sum log diag(L)
        self._log_norm = 0.5 * self.d * np.log(2.0 * np.pi) + self._log_sqrt_detH

        # Precompute L^{-1} for whitening (optional; speeds eval for many points)
        self.L_inv = np.linalg.inv(self.L)

    # ---- bandwidth helpers -------------------------------------------------
    def _make_bandwidth(self, bandwidth: Union[str, float, Array]) -> Array:
        if isinstance(bandwidth, str):
            if bandwidth.lower() not in {"scott", "silverman"}:
                raise ValueError("bandwidth must be 'scott' or 'silverman'")
            alpha = self.n_eff ** (-2.0 / (self.d + 4.0))
            c = 1.0 if bandwidth.lower() == "scott" else (4.0 / (self.d + 2.0)) ** (2.0 / (self.d + 4.0))
            return float(c * alpha) * self.Sigma

        bw = np.asarray(bandwidth, dtype=float)
        if bw.ndim == 0:
            # scalar factor times Sigma (interpreted as std scale)
            s2 = float(bw) ** 2
            return s2 * self.Sigma
        elif bw.ndim == 1:
            if bw.shape[0] != self.d:
                raise ValueError("bandwidth vector must have length d")
            return np.diag(bw ** 2)
        elif bw.ndim == 2:
            if bw.shape != (self.d, self.d):
                raise ValueError("bandwidth matrix must be (d, d)")
            # Verify SPD via Cholesky
            _ = np.linalg.cholesky(bw)
            return bw
        else:
            raise ValueError("Invalid bandwidth shape")

    # ---- core math ---------------------------------------------------------
    @staticmethod
    def _logsumexp(a: Array, axis: int = None) -> Array:
        m = np.max(a, axis=axis, keepdims=True)
        s = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
        return np.squeeze(s, axis=axis)

    # ---- API ----------------------------------------------------------------
    def logpdf(self, Xq: Union[Array, list], chunk_size: int = 2048) -> Array:
        """
        Evaluate log density at points Xq: (m, d) or (d,). Returns (m,).
        Uses a log-sum-exp over mixture of N(x | X_i, H).
        """
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq[None, :]
        if Xq.shape[1] != self.d:
            raise ValueError(f"Xq must have shape (m, {self.d})")

        out = np.empty(Xq.shape[0], dtype=float)
        for start in range(0, Xq.shape[0], chunk_size):
            Ys = Xq[start : start + chunk_size]
            # Differences shape (m_chunk, n, d)
            D = Ys[:, None, :] - self.X[None, :, :]
            # Whiten: Z = solve(L, D^T)^T  -> (m_chunk, n, d)
            Z = np.linalg.solve(self.L, D.reshape(-1, self.d).T).T.reshape(D.shape)
            r2 = np.sum(Z ** 2, axis=2)  # (m_chunk, n)
            # log kernel up to the mixture normalization
            logs = self.logw[None, :] + (-0.5 * r2)
            # log pdf = logsumexp(logs, axis=1) - normalization
            lp = self._logsumexp(logs, axis=1) - self._log_norm
            out[start : start + Ys.shape[0]] = lp
        return out if out.size > 1 else out[0]

    def pdf(self, Xq: Union[Array, list], chunk_size: int = 2048) -> Array:
        return np.exp(self.logpdf(Xq, chunk_size=chunk_size))

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> Array:
        """
        Sample from the KDE mixture: pick a data point with prob w_i,
        then add a small Gaussian noise N(0, H).
        """
        rng = np.random.default_rng() if rng is None else rng
        idx = rng.choice(self.n, size=n, p=self.w)
        base = self.X[idx]
        z = rng.normal(size=(n, self.d)) @ self.L.T
        return base + z


class GaussianKDE_():

    def __init__(self, X,  bw_method="silverman", weights=None):

        X_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X, dtype=float)
        self.X = X_np.T
        self.bw_method = bw_method
        if weights is None:
            self.weights = None
        else:
            w_np = weights.detach().cpu().numpy() if hasattr(weights, "detach") else np.asarray(weights, dtype=float)
            self.weights = w_np / w_np.sum()
        self.kde = gaussian_kde(self.X, bw_method=self.bw_method, weights=self.weights)
        return

    def pdf(self, X):
        X_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X, dtype=float)
        return self.kde.pdf(X_np.T)

    def logpdf(self, X):
        X_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X, dtype=float)
        return self.kde.logpdf(X_np.T)

    def sample(self, n):
        return self.kde.resample(n).T
