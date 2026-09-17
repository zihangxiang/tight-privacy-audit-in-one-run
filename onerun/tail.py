"""The tail bound.

    P[ T <= u ]  =  E_W[ BinCDF(u; r, gammabar(W)) ],   W ~ Beta(n-r, r+1)

where T is the number of mistakes among the r released guesses and W is the
score quantile of the CUT -- the highest-scoring guess that is NOT released.
This is an equality, not a bound, and assumes nothing about independence
between the released error bits.
"""
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import binom


def log_binom_cdf(u, r, p):
    """log P[Binomial(r,p) <= u], vectorised over p, accurate into deep tails.

    scipy's binom.logcdf is a regularised incomplete beta underneath --
    BinCDF(u;r,p) = I_{1-p}(r-u, u+1) -- so it stays accurate where a direct
    sum of pmf terms would underflow.  The two guards below cover the corners
    that arise when gammabar(w) hits 0 (an atom with no error mass) or when
    u >= r (the event is certain).
    """
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    certain = (u >= r)
    if certain:
        return np.zeros_like(p)
    zero_p = p <= 0.0
    out[zero_p] = 0.0                                   # Binomial(r,0) = 0 <= u
    ok = ~zero_p
    if np.any(ok):
        out[ok] = binom.logcdf(int(u), int(r), p[ok])
    return out


def _cut_grid_beta(n, r, n_points = int(5e4), n_sd = 18.0):
    """Quadrature nodes for W ~ Beta(n-r, r+1), in log space.

    The nodes are placed on the Beta's own scale -- mean +/- n_sd standard
    deviations -- and NOT on the score profile's nodes.  The profile's nodes are
    placed where the channel needs resolution, which is a different place: using
    them makes the outer integral both far slower and, once the Beta becomes
    narrow (large n), numerically empty.
    """
    a, b = float(n - r), float(r + 1)
    mean = a / (a + b)
    sd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    
    lo = max(1e-15, mean - n_sd * sd)
    hi = min(1 - 1e-15, mean + n_sd * sd)
    
    w = np.linspace(lo, hi, n_points)
    beta_log_dens = ((a - 1) * np.log(w) + (b - 1) * np.log1p(-w)
                + gammaln(a + b) - gammaln(a) - gammaln(b))
    return w, beta_log_dens


def _integrate(x_points, log_integrand):
    """int exp(log_integrand) dw, done with one log-sum-exp shift.

    Two ways the shift can fail have to be told apart.

      * The log-integrand is +inf or NaN.  That is a numerical failure and is
        raised.
      * The log-integrand is -inf everywhere.  That is not a failure: the
        integrand has underflowed, so the integral is zero to double precision.
        This happens legitimately whenever `epsilon_lower_bound` probes a very
        private null, where P[T <= u] falls below exp(-745); returning 0 is the
        correct answer and lets the bisection proceed.
    """
    M = np.max(log_integrand)
    if np.isnan(M) or M == np.inf:
        raise ValueError("the integrand is not resolved: the log-integrand "
                         f"contains {'NaN' if np.isnan(M) else '+inf'}")
    if M == -np.inf:
        return 0.0
    return float(
        np.exp(M) * np.trapezoid(np.exp(log_integrand - M), x_points)
    )


def exact_tail(profile, u, r, n, n_points=20001, n_sd=18.0):
    """P[T <= u] under the Theorem 2."""
    if not (0 <= u <= r <= n):
        raise ValueError(f"need 0 <= u <= r <= n, got u={u}, r={r}, n={n}")
    if r == n:
        # nothing is discarded: the cut is degenerate, W == 0
        return float(np.exp(log_binom_cdf(u, r, np.array([profile.e_opt]))[0]))
    
    w, beta_log_dens = _cut_grid_beta(n, r, n_points, n_sd)
    if not np.any(np.isfinite(beta_log_dens)):
        # the quadrature nodes missed the Beta entirely -- a grid failure, as
        # distinct from a tail that merely underflows below.
        raise ValueError(f"the cut density is not resolved at n={n}, r={r}: "
                         f"no quadrature node carries finite mass")
    '''this is key quantity by order statistics: the mean error rate above quantile w'''
    gbar_at_w = profile.gammabar(w)
    
    return min(
        1.0, 
        _integrate(
            w, 
            beta_log_dens + log_binom_cdf(u, r, gbar_at_w)
        )
    )
