"""Smallest complete example: one mechanism, one observed audit outcome."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import onerun as oa

n, r, errors = 10_000, 2_500, 350        # canaries, released guesses, mistakes
delta, confidence = 1e-5, 0.95

eps, mu = oa.epsilon_lower_bound(
    lambda t: oa.gaussian(t),            # null family: mu-GDP, searched over mu
    errors = errors, 
    r = r, 
    n = n, 
    delta = delta, 
    confidence = confidence
)

print(f"observed {errors} mistakes in {r} released guesses out of {n} canaries")

''' observe p-value under different privacy parameters, e.g. 1-GDP (mu=1) '''
print(f"p-value under 1-GDP : {oa.p_value(oa.gaussian(1.0), errors, r, n):.4e}")

''' report the lower bound on eps at the given delta and confidence '''
print(f"lower bound         : eps >= {eps:.4f} at delta = {delta:g} "
      f"with {confidence:.0%} confidence   (fitted mu = {mu:.4f})")
