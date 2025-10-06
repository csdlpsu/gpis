from utils import *
from kdemixture import KDEMixture
from test_functions import TestFunctions

@torch.no_grad()
def gp_posterior_pi(model: SingleTaskGP, X: torch.Tensor, threshold: float) -> torch.Tensor:
    post = model.posterior(X)
    mu = post.mean.squeeze(-1)
    var = post.variance.squeeze(-1).clamp_min(1e-16)
    std = var.sqrt()
    # P(f > t) = 1 - Phi((t - mu)/std)
    z = (threshold - mu) / std
    return 1.0 - STD_NORMAL.cdf(z)

def fit_gp(X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
    model = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model

def run_experiment(
    name: str,
    f_and_t: Tuple[Callable[[torch.Tensor], torch.Tensor], float],  # f(x), t with failure if f>t
    p_sampler: Callable[[int], torch.Tensor],
    p_logpdf: Callable[[torch.Tensor], torch.Tensor],
    d: int,
    m_pilot: int = 10000,
    n_init: int = 20,
    q_batch: int = 4,
    n_iters: int = 50,
    alpha: float = 0.7,
    h: float = None,
    eta_schedule: Callable[[int], float] = lambda n: min(1.0, 1.0 / math.sqrt(max(1,n))),
    seed: int = 1234,
):
    set_seed(seed)
    print(f"\n=== {name} ===")
    f, t = f_and_t

    # Pilot and init
    pilot_X = p_sampler(m_pilot)                  # (m,d)
    if h is None:
        # Silverman-like: m^{-1/(4+d)} times a global scale ~ 1.0 (assuming inputs scaled)
        h = float(m_pilot**(-1.0/(4.0+d)))
    sampler = KDEMixture(pilot_X, p_sampler, p_logpdf, bandwidth=h, eta_schedule=eta_schedule)

    X = p_sampler(n_init)
    Y = f(X).unsqueeze(-1)                        # (n_init,1)

    # Storage for MIS (balance heuristic): keep each iteration's (weights, eta_n)
    past = []   # list of dicts: {"weights": (m,), "eta": float}
    prop_counts = []  # counts per proposal: q used each iter (for Q_N)
    total_counts = n_init

    # Precompute prior density constant for reporting ESS, etc., if needed
    # (Not essential; p(X) via p_logpdf is used directly below)

    for it in range(1, n_iters+1):
        # === Fit GP
        model = fit_gp(X, Y)

        # === Compute pi on pilot and weights
        pi_pilot = gp_posterior_pi(model, pilot_X, t).clamp(1e-9, 1-1e-9)
        w = (pi_pilot**alpha)
        w = w / w.sum()
        # === Draw new batch from mixture
        X_new, eta_n = sampler.sample_mixture(it, q_batch, w)
        # Evaluate oracle
        Y_new = f(X_new).unsqueeze(-1)

        # Update data
        X = torch.cat([X, X_new], dim=0)
        Y = torch.cat([Y, Y_new], dim=0)
        total_counts += q_batch

        # Store proposal for MIS
        past.append({"weights": w.detach(), "eta": eta_n})
        prop_counts.append(q_batch)

        # === Online MIS (balance heuristic) over ALL evaluated points so far
        # Build Q_N(x_i) = [n_init * p(x_i) + sum_k N_k * q_k(x_i)] / N_tot
        Xi = X
        logpXi = p_logpdf(Xi)
        num = n_init * torch.exp(logpXi)  # (N,)

        # add each proposal density q_k(x_i) scaled by its batch count
        for k, info in enumerate(past):
            qk = sampler.q_density(Xi, info["weights"], info["eta"])  # (N,)
            num = num + prop_counts[k] * qk

        QN = num / float(total_counts)              # (N,)
        # MIS weights and indicator
        ind_fail = (Y.squeeze(-1) > t).double()
        wMIS = torch.exp(logpXi) / QN
        pf_hat = (ind_fail * wMIS).mean().item()
        print(f"iter {it:02d} | N={X.shape[0]:5d} | pf^MIS ≈ {pf_hat:.3e} | eta={eta_n:.3g}")

    return {"X": X, "Y": Y, "pilot_X": pilot_X, "past": past, "model": model, "t": t, "pf_hat": pf_hat}
