"""Steinke, Nasr and Jagielski's one-run audit bound, as a baseline.

Faithful port of `p_value_DP_audit` / `get_eps_audit` from

    Steinke, Nasr, Jagielski, "Privacy Auditing with One (1) Training Run",
    NeurIPS 2023,

as distributed in that paper's reference implementation.  Reproduced here only
so the two methods can be run side by side on identical inputs; the algorithm is
theirs.
"""
import math

import numpy as np
import scipy.stats


def p_value(m, r, v, eps, delta):
    """P[at least v of r guesses correct] under the (eps,delta)-DP null.

    m = number of examples, r = number of released guesses, v = number correct.
    The binomial term uses the accuracy of eps-DP randomised response,
    q = 1/(1+e^-eps); the second term is the delta correction, which carries the
    O(m) factor.
    """
    assert 0 <= v <= r <= m and eps >= 0 and 0 <= delta <= 1
    q = 1 / (1 + math.exp(-eps))
    beta = scipy.stats.binom.sf(v - 1, r, q)
    if v >= 1:
        i = np.arange(1, v + 1)
        alpha = float(
            np.max(
                (
                    scipy.stats.binom.cdf(v, r, q)
                    - scipy.stats.binom.cdf(v - i, r, q)
                ) / i
            )
        )
    else:
        alpha = 0.0
    return min(beta + alpha * delta * 2 * m, 1.0)


def epsilon_lower_bound(m, r, v, delta, significance=0.05):
    """Largest eps whose p-value stays below `significance`."""
    assert 0 <= v <= r <= m
    lo, hi = 0.0, 1.0
    while p_value(m, r, v, hi, delta) < significance:
        hi += 1.0
        if hi > 200:
            return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if p_value(m, r, v, mid, delta) < significance:
            lo = mid
        else:
            hi = mid
    return lo
