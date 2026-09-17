"""Brute-force Monte Carlo, used to check the closed forms.

Nothing here uses the order-statistic identity, That is why
agreement with `exact_tail` is evidence of soundness.
"""
import numpy as np


def simulate(profile, r, n, reps=int(1e6), chunk=2000, seed=0):
    """Return `reps` independent draws of the released error count T.

    One simulation serves every threshold u, which is why this is exposed
    separately: checking a whole tail costs one run, not one run per point.
    """
    if not (0 < r <= n):
        raise ValueError(f"need 0 < r <= n, got r={r}, n={n}")
    rng = np.random.default_rng(seed)
    edges = profile.w_hi
    out = []
    for start in range(0, reps, chunk):
        m = min(chunk, reps - start)
        # the score quantiles of n i.i.d. channels are n i.i.d. Uniform(0,1)
        w = rng.random((m, n))
        top = w if r == n else np.partition(w, n - r, axis=1)[:, n - r:]
        idx = np.clip(
            np.searchsorted(edges, top.ravel(), side="right"),
            0, 
            len(profile.gamma) - 1
        )
        gam = profile.gamma[idx].reshape(top.shape)
        '''
        coin flips for each of the r released guesses, with success probability = gamma
        '''
        out.append((rng.random(top.shape) < gam).sum(axis=1))
    return np.concatenate(out)


def monte_carlo(profile, u, r, n, reps=200_000, chunk=2000, seed=0):
    """Estimate P[T <= u] directly.  Returns (estimate, standard error)."""
    return tail_from_samples(
        simulate(profile, r, n, reps, chunk, seed), 
        u
    )


def tail_from_samples(samples, u):
    """
    P[T <= u] is what we estimate, return
    P[T <= u] and its standard error, from an existing simulation.
    """
    p = float((samples <= u).mean())
    return p, float(np.sqrt(max(p * (1 - p), 1e-12) / samples.size))
