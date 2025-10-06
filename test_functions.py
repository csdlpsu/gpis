from utils import *
from run_expt import *
from kdemixture import KDEMixture
class TestFunctions():

    def __init__(self):
        self.herbie_bounds = torch.tensor([[-2.0, 2.0], [-2.0, 2.0]])
        self.herbie_dim = 2

        return

    def four_branch_g(self, X: torch.Tensor, pconst: float = 6.0) -> torch.Tensor:
        x1, x2 = X[:, 0], X[:, 1]
        a = 3 + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / math.sqrt(2)
        b = 3 + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / math.sqrt(2)
        c = (x1 - x2) / math.sqrt(2) - pconst
        d = (x2 - x1) / math.sqrt(2) - pconst
        g = torch.minimum(torch.minimum(a, b), torch.minimum(c, d))
        return g

    def f_four_branch(self, X: torch.Tensor) -> torch.Tensor:
        return -self.four_branch_g(X)  # failure if f>0

    def sampler_p_fb(n: int) -> torch.Tensor:
        return torch.randn(n, 2, dtype=torch.double, device=device)

    def logpdf_p_fb(X: torch.Tensor) -> torch.Tensor:
        # standard normal N(0,I) in 2D
        return -0.5 * torch.sum(X * X, dim=1) - math.log(2 * math.pi)

    def herbie(self, X: torch.Tensor) -> torch.Tensor:
        """
        Torch version of the provided numpy herbie_2d.
        X: (n, 2) tensor, dtype should be torch.double
        Returns: (n,) tensor
        """
        # ensure computations happen in double (to match the rest of your code)
        X = X.to(dtype=torch.double)
        term = torch.exp(-(X - 1.0) ** 2) \
             + torch.exp(-0.8 * (X + 1.0) ** 2) \
             - 0.05 * torch.sin(8.0 * (X + 0.1))
        return term.sum(dim=1)

    def herbie_g(self, X: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
        """
        Limit-state function for Herbie.
        g(x) = threshold - herbie(x)
        (You can tune `threshold` to control difficulty; 2.0–2.5 are common.)
        """
        return threshold - self.herbie(X)

    def f_herbie(self, X: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
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