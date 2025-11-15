import torch

from utils import *
from kdemixture import KDEMixture
from typing import Optional, Callable, Tuple
# from typing import Any, Dict, List, Optional, Sequence

class TestFunctions():

    def __init__(self, negate=False):
        # general attributes
        self.negate = negate
        # self.herbie_bounds = torch.tensor([[-2.0, 2.0], [-2.0, 2.0]])
        # self.herbie_dim = 2

        return

class Herbie():

    def __init__(self, negate=False, dtype=torch.double):
        self.negate = negate
        self.dtype  = dtype
        self.dim    = 2
        self.bounds = torch.tensor([[-2.0, 2.0], [-2.0, 2.0]], dtype=dtype)
    def eval(self, X: torch.Tensor) -> torch.Tensor:
        """
        Torch version of the provided numpy herbie_2d.
        X: (n, 2) tensor, dtype should be torch.double
        Returns: (n,) tensor
        """
        # ensure computations happen in double (to match the rest of your code)
        X = X.to(dtype=self.dtype)
        term = torch.exp(-(X - 1.0) ** 2) \
             + torch.exp(-0.8 * (X + 1.0) ** 2) \
             - 0.05 * torch.sin(8.0 * (X + 0.1))
        return term.sum(dim=1)

    def eval_g(self, X: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
        """
        Limit-state function for Herbie.
        g(x) = threshold - herbie(x)
        (You can tune `threshold` to control difficulty; 2.0–2.5 are common.)
        """
        return threshold - self.herbie(X)

    def f_eval(self, X: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
        """
        Failure score used elsewhere in your code: failure if f > 0.
        f(x) = -g(x) = herbie(x) - threshold
        """
        return -self.herbie_g(X, threshold=threshold)

    @staticmethod
    def sampler_p_herbie(n: int) -> torch.Tensor:
        """
        Proposal / input distribution p(x): standard normal N(0, I) in 2D,
        mirroring your four-branch setup.
        """
        return torch.randn(n, 2, dtype=torch.double, device=device)

    @staticmethod
    def logpdf_p_herbie(X: torch.Tensor) -> torch.Tensor:
        """
        Log-density of N(0, I) in 2D.
        """
        return -0.5 * torch.sum(X * X, dim=1) - math.log(2.0 * math.pi)

class FourBranch():
    def __init__(self, negate=False, dtype=torch.double):
        self.negate = negate
        self.dtype = dtype
        self.dim = 2
        self.bounds = torch.tensor([[-8.0, 8.0], [-8.0, 8.0]], dtype=dtype)

    def eval(self, X: torch.Tensor, pconst: float = 6.0) -> torch.Tensor:
        x1, x2 = X[:, 0], X[:, 1]
        a = 3 + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / math.sqrt(2)
        b = 3 + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / math.sqrt(2)
        c = (x1 - x2)  + (pconst / math.sqrt(2))
        d = (x2 - x1)  + (pconst / math.sqrt(2))
        g = torch.minimum(torch.minimum(a, b), torch.minimum(c, d))
        return g

    def f_eval(self, X: torch.Tensor) -> torch.Tensor:
        return -self.four_branch_g(X)  # failure if f>0

    def sampler_p_fb(n: int) -> torch.Tensor:
        return torch.randn(n, 2, dtype=torch.double, device=device)

    def logpdf_p_fb(X: torch.Tensor) -> torch.Tensor:
        # standard normal N(0,I) in 2D
        return -0.5 * torch.sum(X * X, dim=1) - math.log(2 * math.pi)

class CantileverBeam():
    def __init__(self, negate=False, dtype=torch.double):
        self.negate = negate
        self.dtype = dtype
        self.dim = 4
        self.bounds = torch.tensor([[9.0807e+03, 3.0000e+00, 1.7096e+11, 1.0000e-01],
                                    [1.0934e+04, 3.1000e+00, 2.5658e+11, 2.0000e-01]],
                                   dtype=torch.double).T
        self.b_fixed = 0.30  # width (m), fixed
        self.Dmax = 0.05  # allowable deflection (m)


    # Distributions:
    # P ~ N(1e4, (2e2)^2), L ~ U[3.0, 3.1] m, E ~ Lognormal(mean=2.1e11, cv=0.05), t ~ U[0.10, 0.20] m

    def ln_params_from_mean_cv(self, mean, cv):
        s2 = math.log(1.0 + cv * cv)
        m = math.log(mean) - 0.5 * s2
        return m, math.sqrt(s2)

    def sampler_p_cantilever(self, n: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        device = "cpu"

        # order: [P, L, E, t]
        P = torch.normal(mean=tensor(1e4), std=tensor(2e2), size=(n,), dtype=torch.double, device=device)
        L = torch.empty(n, dtype=torch.double, device=device).uniform_(3.0, 3.1)
        mE, sE = self.ln_params_from_mean_cv(2.1e11, 0.05)
        E = torch.distributions.LogNormal(loc=tensor(mE), scale=tensor(sE)).sample((n,)).to(device, torch.double)
        t = torch.empty(n, dtype=torch.double, device=device).uniform_(0.10, 0.20)
        return torch.stack([P, L, E, t], dim=1)

    def logpdf_p_cantilever(self, X: torch.Tensor) -> torch.Tensor:
        P, L, E, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        # P normal
        lp = -0.5 * ((P - 1e4) / 2e2) ** 2 - math.log(2 * math.pi * (2e2) ** 2) / 2
        # L uniform [3.0, 3.1]
        lL = torch.where((L >= 3.0) & (L <= 3.1), torch.full_like(L, -math.log(0.1)), torch.full_like(L, -1e12))
        # E lognormal with (mE, sE)
        mE, sE = self.ln_params_from_mean_cv(2.1e11, 0.05)
        lE = -torch.log(E * sE) - ((torch.log(E) - mE) ** 2) / (2 * sE * sE)
        # t uniform [0.10,0.20]
        lt = torch.where((t >= 0.10) & (t <= 0.20), torch.full_like(t, -math.log(0.10)), torch.full_like(t, -1e12))
        return lp + lL + lE + lt

    def pdf_p_cantilever(self, X: torch.Tensor) -> torch.Tensor:

        return torch.exp(self.logpdf_p_cantilever(X))

    def eval(self, X: torch.Tensor) -> torch.Tensor:
        P, L, E, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        I = self.b_fixed * (t ** 3) / 12.0
        delta = P * (L ** 3) / (3.0 * E * I)
        return delta #- self.Dmax  # failure if > 0

class WeldedBeam():

    def __init__(self, negate=False, dtype=torch.double):
        self.negate = negate
        self.dtype = dtype
        self.dim = 4
        self.WB_lb = torch.tensor([0.125, 0.1, 0.1, 0.125], dtype=torch.double)
        self.WB_ub = torch.tensor([5.0, 10.0, 10.0, 5.0], dtype=torch.double)
        self.bounds = torch.row_stack([self.WB_lb, self.WB_ub])

        self.P_wb = 6000.0
        self.L_wb = 14.0
        self.E_wb = 30e6
        self.tau_max = 13600.0
        self.sigma_max = 30000.0
        self.delta_max = 0.25

    def sampler_p_wb(self, n: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        U = torch.rand(n, 4, dtype=torch.double, device=device)
        return self.WB_lb + (self.WB_ub - self.WB_lb) * U

    def logpdf_p_wb(self, X: torch.Tensor) -> torch.Tensor:
        inside = torch.all((X >= self.WB_lb) & (X <= self.WB_ub), dim=1)
        vol = torch.prod(self.WB_ub - self.WB_lb).item()
        logc = -math.log(vol)
        return torch.where(inside, torch.full((X.shape[0],), logc, dtype=torch.double, device=device),
                           torch.full((X.shape[0],), -1e12, dtype=torch.double, device=device))

    def welded_beam_components(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h, l, t, b = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        # Shear stress components (MathWorks / classical formulas)
        tp = self.P_wb / math.sqrt(2.0) / (h * l)
        R = torch.sqrt(0.25 * (l * l + (h + t) ** 2))
        J = h * l * (l * l / 12.0 + 0.25 * (h + t) ** 2)
        tpp = (self.P_wb / math.sqrt(2.0)) * (self.L_wb + 0.5 * l) * R / (h * l * J)

        tau = torch.sqrt(tp * tp + tpp * tpp + (l * tp * tpp) / R)  # resultant shear stress
        sigma = 5.04e5 / (t * t * b)  # bending stress (psi), matches MW example
        Pc = 64746.022 * (1.0 - 0.028236 * t) * t * (b ** 3)  # buckling load (lb), MW closed form

        # End deflection under P at the tip for rectangular cross-section:
        # δ = 4 P L^3 / (E b t^3)  (derived for this welded-beam setup -> equals 2.1952/(t^3 b))
        delta = 4.0 * self.P_wb * (self.L_wb ** 3) / (self.E_wb * b * (t ** 3))
        return tau, sigma, Pc, delta

    def eval(self, X: torch.Tensor) -> torch.Tensor:
        tau, sigma, Pc, delta = self.welded_beam_components(X)
        g1 = self.tau_max - tau
        g2 = self.sigma_max - sigma
        g3 = Pc - self.P_wb
        g4 = self.delta_max - delta
        g = torch.minimum(torch.minimum(g1, g2), torch.minimum(g3, g4))
        return -g  # failure if > 0

class RoundShaftBT():
    """
    Solid round shaft under combined bending + torsion.
    Inputs order: [M, T, d, sigma_y, G]
      M        : bending moment (N·m), LogNormal (median=450, cv=0.25)
      T        : torque (N·m),        LogNormal (median=300, cv=0.30)
      d        : diameter (m),        Truncated Normal(mean=d_nom, sd=0.0005, bounds [d_nom-0.002, d_nom+0.002])
      sigma_y  : yield (Pa),          Truncated Normal(mean=370e6, sd=30e6, bounds [250e6, 500e6])
      G        : shear modulus (Pa),  Truncated Normal(mean=80e9,  sd=3e9,  bounds [70e9, 90e9])

    eval(X) returns f(x) = max(σ_vm/σ_allow, θ/θ_max). Failure if f(x) > 1.
    """

    def __init__(self, negate: bool = False, dtype=torch.double):
        self.negate = negate
        self.dtype = dtype
        self.dim = 5

        # Geometry / limits
        self.L = 1.2  # length [m]
        self.SF = 1.5  # safety factor on yield
        self.theta_max = 0.06  # allowable twist [rad] (~3.4 deg)

        # Nominal diameter to tune rarity (see earlier note)
        self.d_nom = 0.038  # 38 mm default (≈ Pf ~ 5e-4 with given loads)

        # Support bounds (min row; max row), then transpose to shape (dim, 2)
        # [M, T, d, sigma_y, G]
        self.bounds = torch.tensor([
            [250.0, 150.0, self.d_nom - 0.002, 2.5e8, 7.0e10],
            [800.0, 700.0, self.d_nom + 0.002, 5.0e8, 9.0e10],
        ], dtype=self.dtype).T

    # ---------- Utilities ----------
    def ln_params_from_median_cv(self, median: float, cv: float):
        """Lognormal parameters (mu, sigma) from median and coefficient of variation."""
        s2 = math.log(1.0 + cv * cv)
        m = math.log(median)  # median = exp(mu)
        return m, math.sqrt(s2)

    def _truncnorm_logpdf(self, x, mean, sd, low, high):
        """Logpdf of a truncated Normal(mean, sd) on [low, high]. Returns -1e12 outside support."""
        dist = torch.distributions.Normal(loc=torch.tensor(mean, dtype=self.dtype),
                                          scale=torch.tensor(sd, dtype=self.dtype))
        x = x.to(self.dtype)
        low_t = torch.tensor(low, dtype=self.dtype)
        high_t = torch.tensor(high, dtype=self.dtype)
        Z = (dist.cdf(high_t) - dist.cdf(low_t)).clamp_min(torch.finfo(self.dtype).eps)
        lp = dist.log_prob(x) - torch.log(Z)
        in_support = (x >= low_t) & (x <= high_t)
        return torch.where(in_support, lp, torch.full_like(x, -1e12))

    # ---------- Base measure p ----------
    def sampler_p_shaft(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        device = "cpu"

        # Order: [M, T, d, sigma_y, G]
        # Lognormal(M) with median=450, cv=0.25
        mM, sM = self.ln_params_from_median_cv(450.0, 0.25)
        M = torch.distributions.LogNormal(loc=torch.tensor(mM, dtype=self.dtype),
                                          scale=torch.tensor(sM, dtype=self.dtype)).sample((n,)).to(device,
                                                                                                    self.dtype)

        # Lognormal(T) with median=300, cv=0.30
        mT, sT = self.ln_params_from_median_cv(300.0, 0.30)
        T = torch.distributions.LogNormal(loc=torch.tensor(mT, dtype=self.dtype),
                                          scale=torch.tensor(sT, dtype=self.dtype)).sample((n,)).to(device,
                                                                                                    self.dtype)

        # d ~ Truncated Normal(d_nom, 0.0005) to [d_nom-0.002, d_nom+0.002] (simple rejection)
        mean_d, sd_d = self.d_nom, 5e-4
        low_d, high_d = self.d_nom - 0.002, self.d_nom + 0.002
        d = torch.empty(n, dtype=self.dtype, device=device)
        k = 0
        while k < n:
            size = max(2 * (n - k), 100)  # oversample for fewer loops
            cand = torch.normal(mean=torch.tensor(mean_d, dtype=self.dtype),
                                std=torch.tensor(sd_d, dtype=self.dtype),
                                size=(size,),
                                generator=generator).to(device, self.dtype)
            cand = cand[(cand >= low_d) & (cand <= high_d)]
            take = min(cand.numel(), n - k)
            if take > 0:
                d[k:k + take] = cand[:take]
                k += take

        # sigma_y ~ TruncN(370e6, 30e6) to [250e6, 500e6]
        sigy_mean, sigy_sd, sigy_low, sigy_high = 370e6, 30e6, 250e6, 500e6
        sig_y = torch.empty(n, dtype=self.dtype, device=device)
        k = 0
        while k < n:
            size = max(2 * (n - k), 100)
            cand = torch.normal(mean=torch.tensor(sigy_mean, dtype=self.dtype),
                                std=torch.tensor(sigy_sd, dtype=self.dtype),
                                size=(size,),
                                generator=generator).to(device, self.dtype)
            cand = cand[(cand >= sigy_low) & (cand <= sigy_high)]
            take = min(cand.numel(), n - k)
            if take > 0:
                sig_y[k:k + take] = cand[:take]
                k += take

        # G ~ TruncN(80e9, 3e9) to [70e9, 90e9]
        G_mean, G_sd, G_low, G_high = 80e9, 3e9, 70e9, 90e9
        G = torch.empty(n, dtype=self.dtype, device=device)
        k = 0
        while k < n:
            size = max(2 * (n - k), 100)
            cand = torch.normal(mean=torch.tensor(G_mean, dtype=self.dtype),
                                std=torch.tensor(G_sd, dtype=self.dtype),
                                size=(size,),
                                generator=generator).to(device, self.dtype)
            cand = cand[(cand >= G_low) & (cand <= G_high)]
            take = min(cand.numel(), n - k)
            if take > 0:
                G[k:k + take] = cand[:take]
                k += take

        return torch.stack([M, T, d, sig_y, G], dim=1)

    def logpdf_p_shaft(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.dtype)
        M, T, d, sig_y, G = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

        # M, T: Lognormal
        def logpdf_logn(x, median, cv):
            m, s = self.ln_params_from_median_cv(median, cv)
            # exact lognormal log-pdf (includes -0.5*log(2π))
            return -torch.log(x * s) - 0.5 * math.log(2 * math.pi) - ((torch.log(x) - m) ** 2) / (2 * s * s)

        lM = logpdf_logn(M, 450.0, 0.25)
        lT = logpdf_logn(T, 300.0, 0.30)

        # d, sigma_y, G: truncated normals
        ld = self._truncnorm_logpdf(d, mean=self.d_nom, sd=5e-4,
                                    low=self.d_nom - 0.002, high=self.d_nom + 0.002)
        lsigy = self._truncnorm_logpdf(sig_y, mean=370e6, sd=30e6,
                                       low=250e6, high=500e6)
        lG = self._truncnorm_logpdf(G, mean=80e9, sd=3e9,
                                    low=70e9, high=90e9)

        return lM + lT + ld + lsigy + lG

    def pdf_p_shaft(self, X: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.logpdf_p_shaft(X))

    # ---------- Limit-state evaluation ----------
    def eval(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.dtype)
        M, T, d, sig_y, G = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

        # Section relations
        sig_b = 32.0 * M / (math.pi * d ** 3)  # bending normal stress
        tau = 16.0 * T / (math.pi * d ** 3)  # torsional shear
        sig_vm = torch.sqrt(sig_b ** 2 + 3.0 * tau ** 2)  # von Mises
        sig_allow = sig_y / self.SF

        # Twist
        theta = 32.0 * T * self.L / (G * math.pi * d ** 4)

        # Failure metric f(x): failure if > 1
        f = torch.maximum(sig_vm / sig_allow, theta / self.theta_max)
        return -f if self.negate else f