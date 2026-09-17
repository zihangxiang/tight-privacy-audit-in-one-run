"""Validate the exact tail against brute-force simulation, on every mechanism.

    python examples/validate_all.py

For each mechanism and each (n, r) setting the audit is simulated once, and the
closed form is then checked at several thresholds u against that one simulation.
The comparison is meaningful because the simulator shares no machinery with the
formula: it draws n i.i.d. quantiles, keeps the top r by score, and flips one
coin per released guess.

Two regimes are reported separately, because one statistic cannot serve both.

  RESOLVED  -- the simulation saw at least MIN_EVENTS events on each side, so
               the normal approximation to its error holds and the z-score
               (exact - MC)/se is meaningful.  These should look like standard
               normal draws: we report their spread as well as their maximum,
               since all-near-zero would be as suspicious as an outlier.

  UNRESOLVED -- the tail is below what the simulation can see (typically 0 hits
               in `reps` draws).  A z-score here is meaningless -- the standard
               error collapses to zero and any discrepancy divides by it -- so
               instead we check the closed form against an exact Clopper-Pearson
               99.9% interval, which stays valid at zero counts.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import beta as Beta
import onerun as oa

MIN_EVENTS = 25          # below this the normal approximation to the MC error fails

REPS = 200_000
for i, a in enumerate(sys.argv):
    if a == "--reps":
        REPS = int(sys.argv[i + 1])

MECHANISMS = [
    ("Gaussian mu=0.8",      oa.gaussian(0.8)),
    ("Gaussian mu=1.5",      oa.gaussian(1.5)),
    ("Laplace mu=1",         oa.laplace(1.0)),
    ("Laplace mu=2",         oa.laplace(2.0)),
    ("SGM mu=1 q=0.5",       oa.subsampled_gaussian(1.0, 0.5)),
    ("SGM mu=2 q=0.2",       oa.subsampled_gaussian(2.0, 0.2)),
    ("(1,1e-3)-DP",          oa.eps_delta(1.0, 1e-3)),
    ("(2,1e-2)-DP",          oa.eps_delta(2.0, 1e-2)),
]
SETTINGS = [(2000, 500), (2000, 2000), (4000, 1000)]   # includes r = n
FRACTIONS = [0.80, 0.90, 1.00, 1.10]                   # of the expected errors


def expected_errors(prof, n, r):
    """E[T] under the exact law, used only to place the probes sensibly."""
    if r == n:
        return n * prof.e_opt
    w = np.linspace(1e-12, 1 - 1e-12, 100_001)
    return r * np.trapezoid(Beta.pdf(w, n - r, r + 1) * prof.gammabar(w), w)


def clopper_pearson(k, N, alpha=1e-3):
    """Exact binomial interval; valid at k = 0 and k = N, unlike the normal one."""
    lo = 0.0 if k == 0 else Beta.ppf(alpha / 2, k, N - k + 1)
    hi = 1.0 if k == N else Beta.ppf(1 - alpha / 2, k + 1, N - k)
    return lo, hi


def main():
    print(f"exact tail vs brute-force Monte Carlo   ({REPS:,} reps per setting)\n")
    print(f"{'mechanism':<18} {'n':>5} {'r':>5} {'u':>5} | {'exact':>10} " f"{'Monte Carlo':>19} | {'z':>6}")
    print("-" * 78)
    
    zs, flagged, unresolved, unresolved_bad = [], [], 0, []
    for name, ch in MECHANISMS:
        prof = ch.profile()
        for (n, r) in SETTINGS:
            '''simulate once for this (mechanism, n, r) setting, and reuse the samples for all u'''
            samples = oa.simulate(prof, r=r, n=n, reps=REPS, seed=abs(hash((name, n, r))) % 2**31)
            '''expected number of errors under optimal decoder in the idealized case'''
            E = expected_errors(prof, n, r)
            for frac in FRACTIONS:
                u = int(frac * E)
                if not (0 <= u <= r):
                    continue
                
                exact = oa.exact_tail(prof, u, r, n)
                mc, se = oa.tail_from_samples(samples, u)
                k, N = int((samples <= u).sum()), samples.size
                
                if min(k, N - k) >= MIN_EVENTS:                 # RESOLVED
                    z = (exact - mc) / se
                    zs.append(z)
                    if abs(z) > 3:
                        flagged.append((name, n, r, u, exact, mc, se, z))
                    print(f"{name:<18} {n:>5} {r:>5} {u:>5} | {exact:10.6f} "
                          f"{mc:12.6f}+-{2*se:.5f} | {z:6.2f}")
                else:                                            # UNRESOLVED
                    lo, hi = clopper_pearson(k, N)
                    ok = lo <= exact <= hi
                    unresolved += 1
                    if not ok:
                        unresolved_bad.append((name, n, r, u, exact, lo, hi))
                    print(f"{name:<18} {n:>5} {r:>5} {u:>5} | {exact:10.6f} "
                          f"{'  '+str(k)+'/'+str(N)+' events':>19} | {'  in CI' if ok else 'OUT OF CI'}")
    zs = np.asarray(zs)
    print("-" * 78)
    print(f"\nRESOLVED: {len(zs)} comparisons where the simulation has power")
    print(f"  |z| <= 2 : {int((np.abs(zs) <= 2).sum()):>3}  ({(np.abs(zs)<=2).mean():.0%}, "
          f"expected 95%)")
    print(f"  |z| <= 3 : {int((np.abs(zs) <= 3).sum()):>3}  ({(np.abs(zs)<=3).mean():.0%}, "
          f"expected 99.7%)")
    print(f"  mean z   : {zs.mean():+.3f}   (expected 0; a systematic offset would "
          f"indicate bias)")
    print(f"  sd of z  : {zs.std(ddof=1):.3f}   (expected 1; much below 1 would "
          f"suggest the two are not independent)")
    print(f"  max |z|  : {np.abs(zs).max():.2f}")
    if flagged:
        print("\n  beyond 3 sigma:")
        for name, n, r, u, ex, mc, se, z in flagged:
            print(f"    {name} n={n} r={r} u={u}: exact {ex:.6f} vs MC {mc:.6f} (z={z:+.2f})")
    else:
        print("\n  nothing beyond 3 sigma")

    print(f"\nUNRESOLVED: {unresolved} comparisons below the simulation's resolution")
    print(f"  checked against an exact Clopper-Pearson 99.9% interval instead;")
    print(f"  outside it: {len(unresolved_bad)}")
    for name, n, r, u, ex, lo, hi in unresolved_bad:
        print(f"    {name} n={n} r={r} u={u}: exact {ex:.3e} not in [{lo:.3e}, {hi:.3e}]")

    bad = len(flagged) + len(unresolved_bad)
    print(f"\n{'VALIDATION PASSED' if bad == 0 else f'{bad} DISCREPANCIES'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
