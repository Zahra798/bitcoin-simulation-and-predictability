r"""Four ways to simulate Bitcoin returns, from simplest to richest.

1. Geometric Brownian motion (GBM), the Black-Scholes model:
       dS_t = μ S_t dt + σ S_t dW_t   ⇒   S_t = S_0 exp((μ − σ²/2)t + σ W_t),
   so daily log returns are i.i.d. N(μ − σ²/2, σ²). Thin tails, no memory.

2. i.i.d. bootstrap: resample historical daily returns with replacement.
   Keeps the true (heavy-tailed) distribution, destroys all time dependence.

3. Stationary block bootstrap (Politis & Romano, 1994): copy whole blocks of
   consecutive days, with block lengths ~ Geometric(1/L). Keeps the heavy tails
   AND short-range dependence such as volatility clustering.

4. GARCH(1,1) with Student-t innovations (Bollerslev, 1986):
       r_t = μ + ε_t,   ε_t = σ_t z_t,   z_t ~ t_ν scaled to unit variance,
       σ_t² = ω + α ε_{t−1}² + β σ_{t−1}²,
   a parametric model of volatility clustering. Parameters are estimated by
   maximum likelihood; α + β < 1 is required for a finite long-run variance
   σ̄² = ω / (1 − α − β).
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, signal, special, stats


# ---------------------------------------------------------------------------
# 1. GBM
# ---------------------------------------------------------------------------
class GBM:
    name = "GBM (normal, i.i.d.)"

    def fit(self, r):
        r = np.asarray(r, dtype=float)
        self.sigma_ = float(r.std(ddof=1))
        self.mu_ = float(r.mean() + self.sigma_**2 / 2)  # drift of the price, per day
        return self

    def simulate(self, n_paths, horizon, rng):
        return rng.normal(self.mu_ - self.sigma_**2 / 2, self.sigma_, size=(n_paths, horizon))


# ---------------------------------------------------------------------------
# 2-3. Bootstraps
# ---------------------------------------------------------------------------
class IIDBootstrap:
    name = "i.i.d. bootstrap"

    def fit(self, r):
        self.r_ = np.asarray(r, dtype=float)
        return self

    def simulate(self, n_paths, horizon, rng):
        return rng.choice(self.r_, size=(n_paths, horizon), replace=True)


class StationaryBootstrap:
    name = "block bootstrap"

    def __init__(self, mean_block: int = 20):
        self.mean_block = mean_block

    def fit(self, r):
        self.r_ = np.asarray(r, dtype=float)
        return self

    def simulate(self, n_paths, horizon, rng):
        """Each day: with prob. 1/L jump to a random start, otherwise take the next day (wrapping around)."""
        n = len(self.r_)
        p = 1 / self.mean_block
        idx = np.empty((n_paths, horizon), dtype=int)
        idx[:, 0] = rng.integers(0, n, n_paths)
        jump = rng.random((n_paths, horizon)) < p
        fresh = rng.integers(0, n, (n_paths, horizon))
        for t in range(1, horizon):
            idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
        return self.r_[idx]


# ---------------------------------------------------------------------------
# 4. GARCH(1,1)-t
# ---------------------------------------------------------------------------
SCALE = 100.0  # estimate on percentage returns: much better conditioned


def std_t_logpdf(z, nu):
    """Log-density of a Student-t with ν d.o.f. rescaled to unit variance."""
    return (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
            - (nu + 1) / 2 * np.log1p(z**2 / (nu - 2)))


def std_t_ppf(q, nu):
    return stats.t.ppf(q, nu) * np.sqrt((nu - 2) / nu)


class GARCHt:
    name = "GARCH(1,1)-t"

    def _filter(self, params, x):
        mu, omega, alpha, beta, _ = params
        eps = x - mu
        s2 = np.empty_like(x)
        s2[0] = np.var(x)
        # s2[t] = β s2[t-1] + (ω + α ε²[t-1]): a first-order linear recursion, run in C by lfilter
        s2[1:], _ = signal.lfilter([1.0], [1.0, -beta], omega + alpha * eps[:-1] ** 2, zi=[beta * s2[0]])
        return eps, s2

    def _nll(self, params, x):
        nu = params[4]
        eps, s2 = self._filter(params, x)
        return -np.sum(std_t_logpdf(eps / np.sqrt(s2), nu) - 0.5 * np.log(s2))

    def fit(self, r):
        """Maximum likelihood with *variance targeting* (Engle & Mezrich, 1996):
        ω is tied to the sample variance v by ω = v(1 − α − β), so the model's
        long-run variance equals the observed one and only (μ, α, β, ν) are estimated."""
        x = np.asarray(r, dtype=float) * SCALE
        v = x.var()
        full = lambda q: np.array([q[0], v * (1 - q[1] - q[2]), q[1], q[2], q[3]])  # noqa: E731
        res = optimize.minimize(lambda q: self._nll(full(q), x), [x.mean(), 0.08, 0.88, 5.0], method="SLSQP",
                                bounds=[(-5, 5), (1e-6, 0.999), (1e-6, 0.999), (2.1, 100)],
                                constraints=[{"type": "ineq", "fun": lambda q: 0.999 - q[1] - q[2]}],
                                options={"maxiter": 500, "ftol": 1e-10})
        res.x = full(res.x)
        self.params_ = res.x
        self.converged_ = bool(res.success)
        # log-likelihood of the unscaled returns: f_r(r) = SCALE · f_x(SCALE · r)
        self.loglik_ = float(-res.fun + len(x) * np.log(SCALE))
        mu, omega, alpha, beta, nu = res.x
        self.mu_, self.omega_, self.alpha_, self.beta_, self.nu_ = mu / SCALE, omega / SCALE**2, alpha, beta, nu
        eps, s2 = self._filter(res.x, x)
        self.last_eps_, self.last_s2_ = eps[-1] / SCALE, s2[-1] / SCALE**2
        return self

    @property
    def persistence(self):
        return self.alpha_ + self.beta_

    @property
    def half_life_days(self):
        """Days for a volatility shock to decay by half:  ln(0.5) / ln(α + β)."""
        return float(np.log(0.5) / np.log(self.persistence))

    @property
    def long_run_vol(self):
        return float(np.sqrt(self.omega_ / (1 - self.persistence)))

    def conditional_variance(self, r):
        """σ_t² for every day of r (uses only returns before t), with fitted parameters."""
        x = np.asarray(r, dtype=float) * SCALE
        _, s2 = self._filter(self.params_, x)
        return s2 / SCALE**2

    def next_variance(self, r):
        """One-step-ahead forecast σ²_{T+1} given returns r_1..r_T."""
        x = np.asarray(r, dtype=float) * SCALE
        eps, s2 = self._filter(self.params_, x)
        return float((self.params_[1] + self.params_[2] * eps[-1] ** 2 + self.params_[3] * s2[-1]) / SCALE**2)

    def simulate(self, n_paths, horizon, rng, start_from_last: bool = True, burn: int = 500):
        nu = self.nu_
        z = rng.standard_t(nu, size=(n_paths, horizon + (0 if start_from_last else burn))) * np.sqrt((nu - 2) / nu)
        out = np.empty_like(z)
        if start_from_last:
            s2 = np.full(n_paths, self.omega_ + self.alpha_ * self.last_eps_**2 + self.beta_ * self.last_s2_)
        else:
            s2 = np.full(n_paths, self.omega_ / (1 - self.persistence))
        for t in range(z.shape[1]):
            eps = np.sqrt(s2) * z[:, t]
            out[:, t] = self.mu_ + eps
            s2 = self.omega_ + self.alpha_ * eps**2 + self.beta_ * s2
        return out if start_from_last else out[:, burn:]


def simulate_garch(omega, alpha, beta, nu, mu, n, rng):
    """Draw one GARCH(1,1)-t series with known parameters (used in the unit tests)."""
    g = GARCHt()
    g.mu_, g.omega_, g.alpha_, g.beta_, g.nu_ = mu, omega, alpha, beta, nu
    return g.simulate(1, n, rng, start_from_last=False)[0]
