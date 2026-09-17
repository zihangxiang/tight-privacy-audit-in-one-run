"""Score profiles.

Everything downstream of a mechanism is a function of one object: the joint law
of (score, error) for a SINGLE bit passed through the mechanism's f-DP channel.
We represent that law by two arrays,

    gamma[j]   the error probability of an outcome,   in [0, 1/2]
    mass[j]    its probability under the mixture M = (P+Q)/2,   sums to 1

sorted by INCREASING score, equivalently by DECREASING gamma (Lemma: the error
probability of the optimal decoder is 1/(1+e^s), a decreasing function of the
score s).  A continuous channel is discretised onto a fine grid; a channel with
atoms -- pure DP, or the delta-atom of (eps,delta)-DP -- is represented exactly
by a handful of entries.  The rest of the library never needs to know which.

The only quantity the tail bound consumes is

    G(w) = P[error AND score above quantile w] = int_w^1 gamma(w') dw'

and gammabar(w) = G(w)/(1-w), the mean error rate above quantile w.
"""
import numpy as np


class ScoreProfile:
    """handles the order statistics of the score profile."""

    def __init__(self, gamma, mass, name=""):
        '''gamma is the error probability of an outcome, in [0, 1/2]'''
        ''' mass is the probability of an outcome under the mixture M = (P+Q)/2, sums to 1'''
        
        gamma = np.asarray(gamma, dtype=float)
        mass = np.asarray(mass, dtype=float)
        self.name = name
        
        '''check abnormal values'''
        if gamma.shape != mass.shape:
            raise ValueError("gamma and mass must have the same shape")
        if np.any(mass < 0):
            raise ValueError("mass must be non-negative")
        if np.any((gamma < -1e-12) | (gamma > 0.5 + 1e-12)):
            raise ValueError("gamma must lie in [0, 1/2]")
        
        mass = mass / mass.sum()                     # normalise the mixture
        gamma = np.clip(gamma, 0.0, 0.5)

        '''
        Sort by DECREASING gamma = increasing score.  This is the order the
        filtering uses: the top of the score range is the tail of this array.
        
        gamma can be viewed as the error probability.
        score is the log-likelihood ratio of the two distributions, which is 
        a decreasing function of gamma large ratio means large score, which 
        also means easy to guess, which means small error probability
        '''
        order = np.argsort(-gamma, kind="stable")
        self.gamma = gamma[order]
        self.mass = mass[order]
        
        '''
        now mass[0] is the mass of the entry with the lowest score
        mass[-1] is the mass of the entry with the highest score
        '''

        '''
        w[j] = mixture mass strictly below entry j = the entry's lower quantile.
        Entry j occupies the quantile band [w[j], w[j] + mass[j]).
        '''
        
        '''construct cdf'''
        '''can be seen as < '''
        self.w_lo = np.concatenate([[0.0], np.cumsum(self.mass)[:-1]])
        '''can be seen as <= '''
        self.w_hi = self.w_lo + self.mass

        # G_above[j] = error mass carried by entries strictly above j.
        err = self.gamma * self.mass
        self.G_above = err[::-1].cumsum()[::-1] - err

        # e_opt = E[gamma] = G(0): the optimal single-bit error probability.
        self.e_opt = float(err.sum())

    # -- the two quantities the tail bound needs -----------------------------
    def G(self, w):
        """
        P[error, score above quantile w].  Exact, including inside an atom.

        Within entry j the quantile is uniform over [w_lo[j], w_hi[j]) and the
        error probability is gamma[j], so the part of entry j lying above w
        contributes gamma[j] * (w_hi[j] - w).
        """
        w = np.asarray(w, dtype=float)
        '''inverse cdf'''
        j = np.clip(
            np.searchsorted(self.w_hi, w, side="right"), 
            0, 
            len(self.gamma) - 1
        )
        partial = self.gamma[j] * np.maximum(self.w_hi[j] - w, 0.0)
        return partial + self.G_above[j]

    def gammabar(self, w):
        """
        Mean error rate above quantile w.

        Divide by the QUERY's own 1-w rather than storing gammabar per entry:
        the last entry has w_hi = 1, so a stored gammabar would be 0/0 there and
        would poison every interpolation that touches it.
        """
        w = np.asarray(w, dtype=float)
        return self.G(w) / np.maximum(1.0 - w, np.finfo(float).tiny)

    def __repr__(self):
        return (f"ScoreProfile({self.name!r}, entries={len(self.gamma)}, "
                f"e_opt={self.e_opt:.6f})")


def profile_from_loglr(log_lr, mass, name=""):
    """Build a profile from a log-likelihood-ratio grid.

    gamma = 1/(1+exp|log_lr|).
    """
    a = np.abs(np.asarray(log_lr, dtype=float))
    with np.errstate(over="ignore"):
        # gamma = np.exp(-a) / (1.0 + np.exp(-a)) 
        log_gamma = -a - np.logaddexp(0.0, -a)
        gamma = np.exp(log_gamma)                   # = 1/(1+e^{|L|})
    return ScoreProfile(gamma, mass, name=name)
