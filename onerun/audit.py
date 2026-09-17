"""From an observed audit outcome to a privacy lower bound.

    p_value(...)               the probability a null f-DP mechanism produces
                               at most u mistakes among r released guesses
    epsilon_lower_bound(...)   the largest eps we can reject at the given
                               confidence, reported at the given delta
"""
import numpy as np
from .tail import exact_tail


def p_value(channel, u, r, n, **kw):
    """P[T <= u] under the null that the mechanism is `channel`."""
    return exact_tail(channel.profile(), u, r, n, **kw)


def epsilon_lower_bound(
    set_dp_mechanism, 
    *, 
    errors, 
    r, 
    n, 
    delta,
    confidence = 0.95, 
    param_lo = None, 
    param_hi = None, 
    iters = 40, 
    warn = True
):
    """Largest eps we can reject at `confidence`, given `errors` mistakes.

    `set_dp_mechanism(theta)` must build the null channel at privacy parameter
    theta, with theta increasing = LESS private.  We bisect on theta for the
    largest null whose p-value still falls below the significance level, then
    convert that null to (eps, delta).

    The bisection range must match the family being searched.  A range that does
    not contain the crossing returns an end point, which is a floor/ceiling
    artefact and not a bound; `warn` reports that.
    """
    probe = set_dp_mechanism(1.0)
    '''lo and hi are the lower and upper bounds of the parameter range for the channel.'''
    '''eg, for Gaussian mechanism, the paramter is mu, and lo could be 0 (perfectly private), hi could be 30 (almost not private at all).'''
    lo = probe.param_range[0] if param_lo is None else param_lo
    hi = probe.param_range[1] if param_hi is None else param_hi
    target_p_value = 1.0 - confidence

    '''
    pval(theta) returns the p-value of the null hypothesis at parameter theta.
    e.g., when we set mu = 1, mu is treated as a upper bound, this means that the true
    private algorithm will only be more private than mu=1, when we query pval(theta)
    we compute the probability of observing the number of errors less than or equal 
    to the observed errors r, under the assumption that the mu is treated as a upper bound
    once the p-value is less than the target p-value, we can reject the null hypothesis
    which give us a lower bound on the privacy parameter.
    
    to give some intuition, if we set mu very small, the p-value we get will be very large, 
    which means we cannot reject the null hypothesis;
    then we increase mu, the p-value will decrease, and at some point it will be less than the target p-value,
    which means we can reject the null hypothesis, and we can say that the true privacy parameter 
    is greater than the mu we set, which gives us a lower bound on the privacy parameter.
    '''
    
    def pval(theta):    
        return p_value(set_dp_mechanism(theta), int(errors), int(r), int(n))

    if pval(lo) >= target_p_value:
        '''
        this means the true lower bouns is smaller than lo
        which means that we should increass lo
        '''
        if warn:
            print(f"  [warning] even theta={lo} is not rejected; nothing to report")
        return 0.0, lo
    if pval(hi) < target_p_value:
        '''
        this means the true lower bound is larger than hi
        which means that we should increase hi so that
        we can continue to search for a larger lower bound
        '''
        if warn:
            print(f"  [warning] search hit the ceiling theta={hi}; the returned "
                  f"epsilon is an artefact of the range, not a bound")
        return set_dp_mechanism(hi).eps_from_delta(delta), hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if pval(mid) < target_p_value:
            '''
            it means we still can reject the null, 
            let's increase the lower bound set here to
            get a stronger lower bound
            '''
            lo = mid
        else:
            '''
            it means we cannot reject the null,
            let's decrease it
            '''
            hi = mid
    theta = 0.5 * (lo + hi)
    return set_dp_mechanism(theta).eps_from_delta(delta), theta
