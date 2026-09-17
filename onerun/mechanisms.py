"""
Mechanisms, as (P, Q) pairs represented by their log-likelihood ratio.

A channel is described by two arrays on a common index set:

    log_lr[j]   log dQ/dP at outcome j   (+inf allowed: a delta-atom)
    mass[j]     the outcome's probability under the mixture M = (P+Q)/2

Everything else -- the score profile, delta(eps), eps(delta) -- is derived from
these generically, so a new mechanism only has to supply the pair.
"""
import numpy as np
from scipy.stats import norm
from .profile import profile_from_loglr


class Channel:
    def __init__(self, log_lr, mass, name="", param=None, param_range=(0.01, 32.0)):
        
        self.log_lr = np.asarray(log_lr, dtype=float)
        m = np.asarray(mass, dtype=float)
        self.mass = m / m.sum()
        
        self.name, self.param, self.param_range = name, param, param_range
        # How far into the tail of log(dQ/dP) this representation actually goes.
        # A +inf atom represents the tail exactly, so nothing is truncated and
        # the reach is unbounded; otherwise it is the largest finite value the
        # discretisation attains.
        if np.any(np.isposinf(self.log_lr)):
            self._reach = np.inf
        else:
            finite = self.log_lr[np.isfinite(self.log_lr)]
            self._reach = float(finite.max()) if finite.size else -np.inf

    def profile(self):
        return profile_from_loglr(self.log_lr, self.mass, name=self.name)

    # -- privacy accounting, generic in (log_lr, mass) -----------------------
    def delta_of_eps(self, eps):
        """
        we use an equivalent form to compute delta(eps), this is because
        we have the following fact:
        an dp algorithm is (eps, delta)-DP if and only 
        
        int (q - e^eps p)_+ < delta
        
        as know as the hockey-stick divergence representation of (eps, delta)-DP
        
        where p and q are base pairs corresponding to neighboring distributions.
        in our context, we have"
        
        delta(eps) = int (q - e^eps p)_+ = E_M[ 2 (L - e^eps)_+ / (1 + L) ].

        Derivation: with m = (p+q)/2 and L = q/p we have q = 2mL/(1+L) and
        p = 2m/(1+L), so q - e^eps p = 2m (L - e^eps)/(1+L).  The expression is
        evaluated as 2 (1 - e^{eps - logL}) / (1 + e^{-logL}) so that an
        infinite log_lr (a delta-atom) contributes its full 2*mass.
        """
        L = self.log_lr
        # delta(eps) draws mass only from outcomes with log_lr > eps.  If the
        # discretisation does not reach that far, every term is clipped to zero
        # and the result is a truncation artefact, not a small number.  Say so
        # rather than returning 0 silently.
        if eps > self._reach and not np.any(np.isposinf(L)):
            raise ValueError(
                f"delta({eps:g}) is not resolved: the grid reaches log(dQ/dP) = "
                f"{self._reach:.4g}, so no outcome contributes and the result "
                f"would truncate to 0. Rebuild the channel with a larger `span`.")
            
        with np.errstate(over="ignore", invalid="ignore"):
            num = 1.0 - np.exp(np.minimum(eps - L, 700.0))   # 1 - e^eps/L
            den = 1.0 + np.exp(-np.clip(L, -700.0, 700.0))   # 1 + 1/L
            term = 2.0 * np.where(np.isposinf(L), 1.0, num / den)
        
        '''np.maximum ensure that we only retain point where log_lr > eps, otherwise we set it to 0'''
        return float(np.sum(self.mass * np.maximum(term, 0.0)))

    def eps_from_delta(self, delta, lo=0.0, hi=None, iters=200):
        """Smallest eps with delta(eps) <= delta.  delta(.) is decreasing.

        The search is capped at the tail reach of the representation: beyond it
        delta(eps) is not resolved (see `delta_of_eps`), so a bisection allowed
        to wander past it would converge on a truncation artefact.
        """
        if delta >= 1.0:
            return 0.0
        
        cap = self._reach if np.isfinite(self._reach) else 200.0
        hi = min(200.0, cap) if hi is None else min(hi, cap)
        
        if self.delta_of_eps(hi) > delta * (1.0 + 1e-9) + 1e-300:
            '''should increase hi, otherwise we cannot find a eps such that delta(eps) <= delta'''
            raise ValueError(
                f"delta({hi:.4g}) = {self.delta_of_eps(hi):.3e} is still above the "
                f"target {delta:g} at the edge of what this representation "
                f"resolves. Rebuild the channel with a larger `span`.")
            
        if self.delta_of_eps(lo) <= delta:
            return lo
        
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self.delta_of_eps(mid) > delta:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def _grid(lo, hi, n):
    """
    Midpoint grid: each cell carries the density times its own width, so the
    masses sum to 1 without a separate quadrature rule.
    
    return the midpoints and the width (dx) of each cell
    """
    edges = np.linspace(lo, hi, n + 1)
    return 0.5 * (edges[:-1] + edges[1:]), edges[1] - edges[0]

'''
concrete mechanisms:
For a user defined DP mechanism, must provide a function that returns the log-likelihood ratio and the mass of each outcome
e.g., for Gaussian mechanism, the log-likelihood ratio is log(dQ/dP) = mu x - mu^2/2, 
and the mass of each outcome is the probability density function of the mixture distribution M = (P+Q)/2.
In this manner, we are operating in the discretized space.
'''

# ---------------------------------------------------------------------------

def gaussian(mu, n_grid = 400_000, span = 14.0):
    """mu-GDP: P = N(0,1), Q = N(mu,1).  log dQ/dP = mu x - mu^2/2."""
    x, dx = _grid(-span - abs(mu), abs(mu) + span, n_grid)
    
    '''the interface, log ratio and the mass of each outcome'''
    log_lr = - mu * x + mu * mu / 2.0
    mass = 0.5 * (norm.pdf(x) + norm.pdf(x - mu)) * dx
    
    return Channel(
        log_lr, 
        mass, 
        f"Gaussian(mu={mu})", 
        mu, 
        (0.01, 32.0)
    )


def laplace(mu, b=1.0, n_grid=400_000, span=40.0):
    """P = Laplace(0,b), Q = Laplace(mu,b).  Pure DP with eps = mu/b."""
    x, dx = _grid(-span * b - abs(mu), abs(mu) + span * b, n_grid)
    lp = -np.abs(x) / b
    lq = -np.abs(x - mu) / b
    log_lr = lq - lp
    mass = 0.5 * (np.exp(lp) + np.exp(lq)) / (2 * b) * dx
    
    return Channel(
        log_lr, 
        mass, 
        f"Laplace(mu={mu},b={b})", 
        mu, 
        (0.01, 32.0)
    )


def subsampled_gaussian(mu, q, n_grid=400_000, span=14.0):
    """P = N(0,1), Q = (1-q) N(0,1) + q N(mu,1)."""
    x, dx = _grid(-span - abs(mu), abs(mu) + span, n_grid)
    # log dQ/dP = log[(1-q) + q e^{mu x - mu^2/2}], formed with logaddexp so it
    # stays finite in the far tails where both densities underflow to zero.
    log_lr = np.logaddexp(
        np.log1p(-q) if q < 1 else -np.inf,
        np.log(q) + mu * x - mu * mu / 2.0
    )
    p = norm.pdf(x)
    qd = p * np.exp(np.clip(log_lr, -700.0, 700.0))
    mass = 0.5 * (p + qd) * dx
    
    return Channel(
        log_lr, 
        mass, 
        f"SGM(mu={mu},q={q})", 
        mu, 
        (0.01, 32.0)
    )


def eps_delta(eps, delta):
    """The tight (eps,delta)-DP pair: four atoms, no discretisation error.

        z_inf : P=0,   Q=delta          -> L = +inf
        z_0   : P=delta, Q=0            -> L = -inf
        z_+   : L = +eps
        z_-   : L = -eps
    """
    c = (1 - delta) / (1 + np.exp(eps))
    P = np.array([0.0, delta, c, c * np.exp(eps)])
    Q = np.array([delta, 0.0, c * np.exp(eps), c])
    log_lr = np.array([np.inf, -np.inf, eps, -eps])
    mass = 0.5 * (P + Q)
    
    return Channel(
        log_lr, 
        mass, 
        f"(eps={eps},delta={delta})-DP", 
        eps, 
        (0.01, 32.0)
    )
