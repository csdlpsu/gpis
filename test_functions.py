import math
from typing import Optional, Tuple

import torch

from utils import DEVICE, DTYPE


class TestFunctions:
    def __init__(
        self,
        functionname: str,
        negate: bool = False,
        *,
        dtype: torch.dtype = DTYPE,
        device: torch.device = DEVICE,
    ):
        self.negate = negate
        name = functionname.lower()

        if name == "roundshaft":
            self.func = RoundShaftBT(negate=negate, dtype=dtype, device=device)
        elif name == "cantilever":
            self.func = CantileverBeam(negate=negate, dtype=dtype, device=device)
        elif name == "herbie":
            self.func = Herbie(negate=negate, dtype=dtype, device=device)
        elif name == "fourbranch":
            self.func = FourBranch(negate=negate, dtype=dtype, device=device)
        elif name == "weldedbeam":
            self.func = WeldedBeam(negate=negate, dtype=dtype, device=device)
        else:
            raise NotImplementedError("Entered function name is not implemented")

        self.func_ = self.func.eval


class Herbie:
    def __init__(self, negate: bool = False, dtype: torch.dtype = DTYPE, device: torch.device = DEVICE):
        self.negate = negate
        self.dtype = dtype
        self.device = torch.device(device)
        self.dim = 2
        self.bounds = torch.tensor([[-2.0, 2.0], [-2.0, 2.0]], dtype=dtype, device=self.device)

    def eval(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=self.dtype, device=self.device)
        term = (
            torch.exp(-(X - 1.0) ** 2)
            + torch.exp(-0.8 * (X + 1.0) ** 2)
            - 0.05 * torch.sin(8.0 * (X + 0.1))
        )
        y = term.sum(dim=1)
        return -y if self.negate else y

    def eval_g(self, X: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
        return threshold - self.eval(X)

    def f_eval(self, X: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
        return self.eval(X) - threshold

    def sampler(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return torch.randn((n, self.dim), dtype=self.dtype, device=self.device, generator=generator)

    def logpdf_p_herbie(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=self.dtype, device=self.device)
        return -0.5 * torch.sum(X * X, dim=1) - math.log(2.0 * math.pi)

    def pdf(self, X: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.logpdf_p_herbie(X))


class FourBranch:
    def __init__(self, negate: bool = False, dtype: torch.dtype = DTYPE, device: torch.device = DEVICE):
        self.negate = negate
        self.dtype = dtype
        self.device = torch.device(device)
        self.dim = 2
        self.bounds = torch.tensor([[-8.0, 8.0], [-8.0, 8.0]], dtype=dtype, device=self.device)

    def eval(self, X: torch.Tensor, pconst: float = 6.0) -> torch.Tensor:
        X = X.to(dtype=self.dtype, device=self.device)
        x1, x2 = X[:, 0], X[:, 1]
        a = 3 + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / math.sqrt(2)
        b = 3 + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / math.sqrt(2)
        c = (x1 - x2) + (pconst / math.sqrt(2))
        d = (x2 - x1) + (pconst / math.sqrt(2))
        y = torch.minimum(torch.minimum(a, b), torch.minimum(c, d))
        return -y if self.negate else y

    def f_eval(self, X: torch.Tensor) -> torch.Tensor:
        return -self.eval(X)

    def sampler(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return torch.randn((n, self.dim), dtype=self.dtype, device=self.device, generator=generator)

    def logpdf_p_fb(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=self.dtype, device=self.device)
        return -0.5 * torch.sum(X * X, dim=1) - math.log(2 * math.pi)

    def pdf(self, X: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.logpdf_p_fb(X))


class CantileverBeam:
    def __init__(self, negate: bool = False, dtype: torch.dtype = DTYPE, device: torch.device = DEVICE):
        self.negate = negate
        self.dtype = dtype
        self.device = torch.device(device)
        self.dim = 4
        self.bounds = torch.tensor(
            [
                [9.0807e03, 3.0000e00, 1.7096e11, 1.0000e-01],
                [1.0934e04, 3.1000e00, 2.5658e11, 2.0000e-01],
            ],
            dtype=dtype,
            device=self.device,
        ).T
        self.b_fixed = 0.30
        self.Dmax = 0.05

    def ln_params_from_mean_cv(self, mean: float, cv: float) -> Tuple[float, float]:
        s2 = math.log(1.0 + cv * cv)
        m = math.log(mean) - 0.5 * s2
        return m, math.sqrt(s2)

    def sampler(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        P = torch.normal(
            mean=torch.tensor(1e4, dtype=self.dtype, device=self.device),
            std=torch.tensor(2e2, dtype=self.dtype, device=self.device),
            size=(n,),
            generator=generator,
        )
        L = torch.empty(n, dtype=self.dtype, device=self.device).uniform_(3.0, 3.1, generator=generator)
        mE, sE = self.ln_params_from_mean_cv(2.1e11, 0.05)
        E = torch.distributions.LogNormal(
            loc=torch.tensor(mE, dtype=self.dtype, device=self.device),
            scale=torch.tensor(sE, dtype=self.dtype, device=self.device),
        ).sample((n,))
        t = torch.empty(n, dtype=self.dtype, device=self.device).uniform_(0.10, 0.20, generator=generator)
        return torch.stack([P, L, E, t], dim=1)

    def logpdf_p_cantilever(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=self.dtype, device=self.device)
        P, L, E, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        lp = -0.5 * ((P - 1e4) / 2e2) ** 2 - math.log(2 * math.pi * (2e2) ** 2) / 2
        lL = torch.where((L >= 3.0) & (L <= 3.1), torch.full_like(L, -math.log(0.1)), torch.full_like(L, -1e12))
        mE, sE = self.ln_params_from_mean_cv(2.1e11, 0.05)
        lE = -torch.log(E * sE) - ((torch.log(E) - mE) ** 2) / (2 * sE * sE)
        lt = torch.where((t >= 0.10) & (t <= 0.20), torch.full_like(t, -math.log(0.10)), torch.full_like(t, -1e12))
        return lp + lL + lE + lt

    def pdf(self, X: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.logpdf_p_cantilever(X))

    def eval(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=self.dtype, device=self.device)
        P, L, E, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        inertia = self.b_fixed * (t ** 3) / 12.0
        y = P * (L ** 3) / (3.0 * E * inertia)
        return -y if self.negate else y


class WeldedBeam:
    def __init__(self, negate: bool = False, dtype: torch.dtype = DTYPE, device: torch.device = DEVICE):
        self.negate = negate
        self.dtype = dtype
        self.device = torch.device(device)
        self.dim = 4
        self.WB_lb = torch.tensor([0.125, 0.1, 0.1, 0.125], dtype=dtype, device=self.device)
        self.WB_ub = torch.tensor([5.0, 10.0, 10.0, 5.0], dtype=dtype, device=self.device)
        self.bounds = torch.stack([self.WB_lb, self.WB_ub], dim=1)

        self.P_wb = 6000.0
        self.L_wb = 14.0
        self.E_wb = 30e6
        self.tau_max = 13600.0
        self.sigma_max = 30000.0
        self.delta_max = 0.25

    def sampler(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        U = torch.rand((n, self.dim), dtype=self.dtype, device=self.device, generator=generator)
        return self.WB_lb + (self.WB_ub - self.WB_lb) * U

    def logpdf_p_wb(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=self.dtype, device=self.device)
        inside = torch.all((X >= self.WB_lb) & (X <= self.WB_ub), dim=1)
        vol = torch.prod(self.WB_ub - self.WB_lb).item()
        logc = -math.log(vol)
        return torch.where(
            inside,
            torch.full((X.shape[0],), logc, dtype=self.dtype, device=self.device),
            torch.full((X.shape[0],), -1e12, dtype=self.dtype, device=self.device),
        )

    def pdf(self, X: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.logpdf_p_wb(X))

    def welded_beam_components(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X = X.to(dtype=self.dtype, device=self.device)
        h, length, t, b = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        tp = self.P_wb / math.sqrt(2.0) / (h * length)
        R = torch.sqrt(0.25 * (length * length + (h + t) ** 2))
        J = h * length * (length * length / 12.0 + 0.25 * (h + t) ** 2)
        tpp = (self.P_wb / math.sqrt(2.0)) * (self.L_wb + 0.5 * length) * R / (h * length * J)
        tau = torch.sqrt(tp * tp + tpp * tpp + (length * tp * tpp) / R)
        sigma = 5.04e5 / (t * t * b)
        Pc = 64746.022 * (1.0 - 0.028236 * t) * t * (b ** 3)
        delta = 4.0 * self.P_wb * (self.L_wb ** 3) / (self.E_wb * b * (t ** 3))
        return tau, sigma, Pc, delta

    def eval(self, X: torch.Tensor) -> torch.Tensor:
        tau, sigma, Pc, delta = self.welded_beam_components(X)
        g1 = self.tau_max - tau
        g2 = self.sigma_max - sigma
        g3 = Pc - self.P_wb
        g4 = self.delta_max - delta
        y = -torch.minimum(torch.minimum(g1, g2), torch.minimum(g3, g4))
        return -y if self.negate else y


class RoundShaftBT:
    def __init__(self, negate: bool = False, dtype: torch.dtype = DTYPE, device: torch.device = DEVICE):
        self.negate = negate
        self.dtype = dtype
        self.device = torch.device(device)
        self.dim = 5
        self.L = 1.2
        self.SF = 1.5
        self.theta_max = 0.06
        self.d_nom = 0.038
        self.bounds = torch.tensor(
            [
                [250.0, 150.0, self.d_nom - 0.002, 2.5e8, 7.0e10],
                [800.0, 700.0, self.d_nom + 0.002, 5.0e8, 9.0e10],
            ],
            dtype=dtype,
            device=self.device,
        ).T

    def ln_params_from_median_cv(self, median: float, cv: float) -> Tuple[float, float]:
        s2 = math.log(1.0 + cv * cv)
        return math.log(median), math.sqrt(s2)

    def _truncnorm_logpdf(self, x: torch.Tensor, mean: float, sd: float, low: float, high: float) -> torch.Tensor:
        dist = torch.distributions.Normal(
            loc=torch.tensor(mean, dtype=self.dtype, device=self.device),
            scale=torch.tensor(sd, dtype=self.dtype, device=self.device),
        )
        x = x.to(dtype=self.dtype, device=self.device)
        low_t = torch.tensor(low, dtype=self.dtype, device=self.device)
        high_t = torch.tensor(high, dtype=self.dtype, device=self.device)
        Z = (dist.cdf(high_t) - dist.cdf(low_t)).clamp_min(torch.finfo(self.dtype).eps)
        lp = dist.log_prob(x) - torch.log(Z)
        in_support = (x >= low_t) & (x <= high_t)
        return torch.where(in_support, lp, torch.full_like(x, -1e12))

    def _rejection_truncnorm(
        self,
        n: int,
        mean: float,
        sd: float,
        low: float,
        high: float,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        out = torch.empty(n, dtype=self.dtype, device=self.device)
        k = 0
        mean_t = torch.tensor(mean, dtype=self.dtype, device=self.device)
        sd_t = torch.tensor(sd, dtype=self.dtype, device=self.device)
        while k < n:
            size = max(2 * (n - k), 100)
            cand = torch.normal(mean=mean_t, std=sd_t, size=(size,), generator=generator)
            cand = cand[(cand >= low) & (cand <= high)]
            take = min(cand.numel(), n - k)
            if take > 0:
                out[k : k + take] = cand[:take]
                k += take
        return out

    def sampler(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        mM, sM = self.ln_params_from_median_cv(450.0, 0.25)
        M = torch.distributions.LogNormal(
            loc=torch.tensor(mM, dtype=self.dtype, device=self.device),
            scale=torch.tensor(sM, dtype=self.dtype, device=self.device),
        ).sample((n,))

        mT, sT = self.ln_params_from_median_cv(300.0, 0.30)
        T = torch.distributions.LogNormal(
            loc=torch.tensor(mT, dtype=self.dtype, device=self.device),
            scale=torch.tensor(sT, dtype=self.dtype, device=self.device),
        ).sample((n,))

        d = self._rejection_truncnorm(n, self.d_nom, 5e-4, self.d_nom - 0.002, self.d_nom + 0.002, generator)
        sig_y = self._rejection_truncnorm(n, 370e6, 30e6, 250e6, 500e6, generator)
        G = self._rejection_truncnorm(n, 80e9, 3e9, 70e9, 90e9, generator)
        return torch.stack([M, T, d, sig_y, G], dim=1)

    def logpdf_p_shaft(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=self.dtype, device=self.device)
        M, T, d, sig_y, G = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

        def logpdf_logn(x: torch.Tensor, median: float, cv: float) -> torch.Tensor:
            m, s = self.ln_params_from_median_cv(median, cv)
            return -torch.log(x * s) - 0.5 * math.log(2 * math.pi) - ((torch.log(x) - m) ** 2) / (2 * s * s)

        lM = logpdf_logn(M, 450.0, 0.25)
        lT = logpdf_logn(T, 300.0, 0.30)
        ld = self._truncnorm_logpdf(d, mean=self.d_nom, sd=5e-4, low=self.d_nom - 0.002, high=self.d_nom + 0.002)
        lsigy = self._truncnorm_logpdf(sig_y, mean=370e6, sd=30e6, low=250e6, high=500e6)
        lG = self._truncnorm_logpdf(G, mean=80e9, sd=3e9, low=70e9, high=90e9)
        return lM + lT + ld + lsigy + lG

    def pdf(self, X: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.logpdf_p_shaft(X))

    def eval(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=self.dtype, device=self.device)
        M, T, d, sig_y, G = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        sig_b = 32.0 * M / (math.pi * d ** 3)
        tau = 16.0 * T / (math.pi * d ** 3)
        sig_vm = torch.sqrt(sig_b ** 2 + 3.0 * tau ** 2)
        sig_allow = sig_y / self.SF
        theta = 32.0 * T * self.L / (G * math.pi * d ** 4)
        y = torch.maximum(sig_vm / sig_allow, theta / self.theta_max)
        return -y if self.negate else y
