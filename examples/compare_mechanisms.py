"""
Lower bounds for different DP mechanism families at a common operating point.
The common operating point is the expected number of errors to be made under an optimal decoder.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import beta as Beta
import onerun as oa

n, r, delta, conf = 10_000, 2_500, 1e-5, 0.95
FAMILIES = {
    "Gaussian":            (lambda t: oa.gaussian(t),                    oa.gaussian(1.0)),
    "Laplace":             (lambda t: oa.laplace(t),                     oa.laplace(1.0)),
    "Sub-sampled Gaussian":(lambda t: oa.subsampled_gaussian(t, 0.5),    oa.subsampled_gaussian(1.0, 0.5)),
    "(eps,delta)-DP":      (lambda t: oa.eps_delta(t, 1e-6),             oa.eps_delta(1.0, 1e-6)),
}

def expected_errors(prof, n, r):
    w = np.linspace(1e-12, 1 - 1e-12, 200_001)
    return r * np.trapezoid(Beta.pdf(w, n - r, r + 1) * prof.gammabar(w), w)

print(f"n = {n}, r = {r}, delta = {delta:g}, confidence = {conf:.0%}\n")
print(f"{'family':<22} {'true eps':>9} {'E[errors]':>10} {'observed u':>11} "
      f"{'eps lower bound':>16}")
for name, (make, truth) in FAMILIES.items():
    prof = truth.profile()
    E = expected_errors(prof, n, r)
    u = int(round(E))                       # a typical, unlucky-free observation
    eps, _ = oa.epsilon_lower_bound(
        make, 
        errors=u, 
        r=r, 
        n=n,
        delta=delta, 
        confidence=conf, 
        warn=False
    )
    print(f"{name:<22} {truth.eps_from_delta(delta):9.4f} {E:10.1f} {u:11d} {eps:16.4f}")
