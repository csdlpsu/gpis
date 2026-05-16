import math
from typing import Callable, Tuple

import torch

from utils import device


class KDEMixture:
    """
    Weighted Gaussian-kernel KDE on a fixed pilot set U ~ p, mixture with p:
      q_n(x) = (1 - eta_n) * KDE_n(x) + eta_n * p(x)
    """
    def __init__(
        self,
        pilot_X: torch.Tensor,           # (m,d)
        p_sampler: Callable[[int], torch.Tensor],  # k -> (k,d) samples ~ p
        p_logpdf: Callable[[torch.Tensor], torch.Tensor],  # X -> log p(X) [N]
        bandwidth: float,
        eta_schedule: Callable[[int], float],      # n -> eta_n
    ):
        self.U = pilot_X.to(device)                 # (m,d)
        self.m, self.d = self.U.shape
        self.h = float(bandwidth)
        self.p_sampler = p_sampler
        self.p_logpdf = p_logpdf
        self.eta_schedule = eta_schedule
        self.history = []   # store tuples (weights, eta_n) per iteration

        self.kcoef = (2*math.pi*self.h*self.h)**(-self.d/2)

    def kde_density(self, X: torch.Tensor, weights: torch.Tensor, chunk_m: int = 4000) -> torch.Tensor:
        """
        Evaluate KDE_n(X) = sum_j w_j N(X | U_j, h^2 I).
        X: (N,d), weights: (m,), returns (N,)
        """
        X = X.to(device)
        N = X.shape[0]
        out = torch.zeros(N, dtype=torch.double, device=device)
        for start in range(0, self.m, chunk_m):
            stop = min(start + chunk_m, self.m)
            U_chunk = self.U[start:stop]                              # (m',d)
            w_chunk = weights[start:stop]                             # (m',)
            # pairwise squared distances
            # (N,m') via ||x||^2 - 2 x.u + ||u||^2 trick
            x2 = (X**2).sum(dim=1, keepdim=True)                      # (N,1)
            u2 = (U_chunk**2).sum(dim=1).unsqueeze(0)                 # (1,m')
            cross = X @ U_chunk.t()                                   # (N,m')
            dist2 = x2 - 2.0*cross + u2
            K = torch.exp(-0.5 * dist2 / (self.h*self.h)) * self.kcoef  # (N,m')
            out += K @ w_chunk
        return out

    def sample_kde(self, weights: torch.Tensor, K: int) -> torch.Tensor:
        """ Sample K points via center selection (Discrete(weights)) and Gaussian jitter """
        # Draw centers by multinomial
        idx = torch.multinomial(weights, num_samples=K, replacement=True)  # (K,)
        centers = self.U[idx]                                              # (K,d)
        noise = torch.randn_like(centers) * self.h
        return centers + noise

    def sample_mixture(self, iter_idx: int, K: int, weights: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """ Draw K from (1-eta)*KDE + eta*p. Return samples and eta_n used. """
        eta_n = float(self.eta_schedule(iter_idx))
        K_kde = torch.distributions.Binomial(total_count=K, probs=1.0 - eta_n).sample().to(torch.int64).item()
        K_p   = K - K_kde
        Xk = self.sample_kde(weights, K_kde) if K_kde > 0 else torch.empty(0, self.d, dtype=torch.double, device=device)
        Xp = self.p_sampler(K_p) if K_p > 0 else torch.empty(0, self.d, dtype=torch.double, device=device)
        X  = torch.cat([Xk, Xp], dim=0) if K > 0 else Xk
        return X, eta_n

    def q_density(self, X: torch.Tensor, weights: torch.Tensor, eta_n: float) -> torch.Tensor:
        logp = self.p_logpdf(X)
        kde  = self.kde_density(X, weights)
        return (1.0 - eta_n)*kde + eta_n*torch.exp(logp)
